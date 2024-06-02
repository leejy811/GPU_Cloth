#include "PBD_ClothCuda.h"
#include "include/CUDA_Custom/DeviceManager.cuh"

__constant__ ClothParam clothParam;

__global__ void InitSaturation(Face face)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	face.d_fSaturation[idx] = clothParam._maxsaturation;
}

__global__ void CompGravity_kernel(Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL3 v = ver.d_Vel[idx];
	REAL mass = ver.d_SatMass[idx] + (1.0 / ver.d_InvMass[idx]);
	REAL3 gravity = make_REAL3(0.0, clothParam._gravity, 0.0);
	REAL3 visc = -1 * v * mass;
	v += gravity * clothParam._subdt;
	v += visc * (1.0 / mass) * clothParam._subdt;
	v += ver.d_Adhesion[idx] * (1.0 / mass) * clothParam._subdt;
	v *= clothParam._linearDamping;
	ver.d_Vel[idx] = v;

	ver.d_Pos1[idx] = ver.d_Pos[idx] + (ver.d_Vel[idx] * clothParam._subdt);
}

__global__ void CompWind_kernel(Face face, Vertex ver, REAL3 wind)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	uint iv0 = face.d_faceIdx[idx].x;
	uint iv1 = face.d_faceIdx[idx].y;
	uint iv2 = face.d_faceIdx[idx].z;

	REAL3 v0 = ver.d_Pos1[iv0];
	REAL3 v1 = ver.d_Pos1[iv1];
	REAL3 v2 = ver.d_Pos1[iv2];

	REAL3 normal = Cross(v1 - v0, v2 - v0);
	Normalize(normal);
	REAL3 force = normal * Dot(normal, wind);
	ver.d_Vel[iv0] += force;
	ver.d_Vel[iv1] += force;
	ver.d_Vel[iv2] += force;
}

__global__ void CompIntergrate_kernel(Vertex ver)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	//if (ver.d_Pos1[idx].x > 0.495 && ver.d_Pos1[idx].x < 0.515 && ver.d_Pos1[idx].y > 0.49)
	//	return;

	if (ver.d_Pos1[idx].x > 0.9)
		return;

	ver.d_Vel[idx] = (ver.d_Pos1[idx] - ver.d_Pos[idx]) * clothParam._subInvdt;

	REAL speed = Length(ver.d_Vel[idx]);
	REAL maxSpeed = clothParam._thickness * clothParam._subInvdt;

	if (speed > maxSpeed)
		ver.d_Vel[idx] *= (maxSpeed / speed);

	REAL dt = 1.0 / clothParam._subInvdt;
	ver.d_Pos[idx] += ver.d_Vel[idx] * dt;

	//ver.d_Pos[idx] = ver.d_Pos1[idx];
}

__global__ void CompFaceNorm_kernel(Face face, Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	uint iv0 = face.d_faceIdx[idx].x;
	uint iv1 = face.d_faceIdx[idx].y;
	uint iv2 = face.d_faceIdx[idx].z;

	REAL3 v0 = ver.d_Pos[iv0];
	REAL3 v1 = ver.d_Pos[iv1];
	REAL3 v2 = ver.d_Pos[iv2];

	REAL3 norm = Cross(v1 - v0, v2 - v0);
	Normalize(norm);
	face.d_fNormal[idx] = norm;
}

__global__ void CompVertexNorm_kernel(Face face, Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	uint numNbFaces = ver.d_nbFaces._index[idx + 1] - ver.d_nbFaces._index[idx];

	for (int i = 0; i < numNbFaces; i++)
	{
		uint fIdx = ver.d_nbFaces._array[ver.d_nbFaces._index[idx] + i];
		ver.d_vNormal[idx] += face.d_fNormal[fIdx];
	}
	ver.d_vNormal[idx] /= numNbFaces;
	Normalize(ver.d_vNormal[idx]);
}

__device__ int3 calculateGridPos(REAL3 pos, REAL gridSize)
{
	int3 intPos = make_int3(floorf(pos.x / gridSize), floorf(pos.y / gridSize), floorf(pos.z / gridSize));
	return intPos;
}

__device__ uint calculateGridHash(int3 pos, uint gridRes)
{
	pos.x = pos.x &
		(gridRes - 1);  // wrap grid, assumes size is power of 2
	pos.y = pos.y & (gridRes - 1);
	pos.z = pos.z & (gridRes - 1);

	return __umul24(__umul24(pos.z, gridRes), gridRes) +
		__umul24(pos.y, gridRes) + pos.x;
}

__global__ void Colide_PP(Vertex ver, Device_Hash hash, REAL* impulse)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL cellSize = 1.0 / clothParam._gridRes;
	int3 gridPos = calculateGridPos(ver.d_Pos1[idx], cellSize);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);

				uint neighHash = calculateGridHash(neighbourPos, clothParam._gridRes);
				uint startIdx = hash.d_cellStart[neighHash];

				if (startIdx != 0xffffffff)
				{
					uint endIdx = hash.d_cellEnd[neighHash];

					for (uint i = startIdx; i < endIdx; i++)
					{
						uint id0 = idx;
						uint id1 = hash.d_gridIdx[i];

						if (id0 != id1)
						{

							REAL3 diffPos = ver.d_Pos1[id0] - ver.d_Pos1[id1];
							REAL dist2 = LengthSquared(diffPos);
							REAL thickness2 = clothParam._thickness * clothParam._thickness;

							if (dist2 > thickness2) continue;

							REAL restDist2 = LengthSquared(ver.d_restPos[id0] - ver.d_restPos[id1]);
							REAL minDist = clothParam._thickness;

							if (dist2 > restDist2) continue;

							if (restDist2 < thickness2)
								minDist = sqrt(restDist2);

							REAL dist = sqrt(dist2);
							diffPos = diffPos * ((minDist - dist) / dist);

							atomicAdd_REAL(impulse + id0 * 3u + 0u, diffPos.x * 0.5);
							atomicAdd_REAL(impulse + id0 * 3u + 1u, diffPos.y * 0.5);
							atomicAdd_REAL(impulse + id0 * 3u + 2u, diffPos.z * 0.5);

							atomicAdd_REAL(impulse + id1 * 3u + 0u, diffPos.x * -0.5);
							atomicAdd_REAL(impulse + id1 * 3u + 1u, diffPos.y * -0.5);
							atomicAdd_REAL(impulse + id1 * 3u + 2u, diffPos.z * -0.5);

							REAL3 v0 = (ver.d_Pos1[id0] - ver.d_Pos[id0]);
							REAL3 v1 = (ver.d_Pos1[id1] - ver.d_Pos[id1]);

							REAL3 Vavg = (v0 + v1) * 0.5;

							REAL3 dp0 = (Vavg - v0);
							REAL3 dp1 = (Vavg - v1);

							atomicAdd_REAL(impulse + id0 * 3u + 0u, dp0.x);
							atomicAdd_REAL(impulse + id0 * 3u + 1u, dp0.y);
							atomicAdd_REAL(impulse + id0 * 3u + 2u, dp0.z);

							atomicAdd_REAL(impulse + id1 * 3u + 0u, dp1.x);
							atomicAdd_REAL(impulse + id1 * 3u + 1u, dp1.y);
							atomicAdd_REAL(impulse + id1 * 3u + 2u, dp1.z);

						}
					}
				}
			}
		}
	}
}

__device__ REAL SDFCalculate(REAL x, REAL y, REAL z, BOOL isPlane)
{
	if (isPlane)
	{
		//ax + by + cz + d = 0 평면 방정식
		REAL a = 0.0;
		REAL b = 2.0;
		REAL c = 0.0;
		REAL d = -0.7;

		return (a * x + b * y + c * z + d) / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
	}
	else
	{
		REAL x0 = 0.5f, y0 = 0.4f, z0 = 0.5f; //sphere 중심 좌표
		REAL r = 0.2f;
		return sqrt(pow(x - x0, 2) + pow(y - y0, 2) + pow(z - z0, 2)) - r; //구 방정식
	}
}

__device__ REAL GetCoefficientFriction(REAL sat)
{
	REAL gamma = 1;
	REAL sigma = 5;
	REAL mu = 30;

	REAL zeta = 23;
	REAL logistic = (1 / (1 + exp(-1 * (zeta - mu) / sigma)));
	REAL alpha = gamma * logistic;

	if (sat <= zeta) 
	{
		return alpha;
	}
	else
		return gamma * (1 / (1 + exp(-1 * (sat - mu) / sigma)));
}

__global__ void LevelSetCollision_D(Vertex ver, Face face, BOOL isPlane)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL deltaT = 0.01f;
	REAL h = 0.1f;
	REAL coefficientFriction = GetCoefficientFriction(ver.d_vSaturation[idx]);
	//REAL coefficientFriction = 100 ;

	REAL x = ver.d_Pos[idx].x;
	REAL y = ver.d_Pos[idx].y;
	REAL z = ver.d_Pos[idx].z;

	REAL3 N = make_REAL3(SDFCalculate(x + h, y, z, isPlane) - SDFCalculate(x, y, z, isPlane),
		SDFCalculate(x, y + h, z, isPlane) - SDFCalculate(x, y, z, isPlane),
		SDFCalculate(x, y, z + h, isPlane) - SDFCalculate(x, y, z, isPlane));
	//N.print();

	N /= h; //법선 벡터 계산 (오일러 방법 이용) = Gradient PI
	Normalize(N);

	REAL pi = SDFCalculate(x, y, z, isPlane); //PI, newPI 계산
	REAL newPI = pi + Dot((ver.d_Vel[idx] * deltaT), N);

	ver.d_vSaturation[idx] = 0;
	if (newPI < 0)
	{
		REAL vpN = Dot(ver.d_Vel[idx], N); //원래의 법선 방향 속력
		REAL3 vpNN = N * vpN; //원래의 법선 방향 속도
		REAL3 vpT = ver.d_Vel[idx] - vpNN; //원래의 접선 방향 속도

		REAL newVpN = vpN - (newPI / deltaT); //새로운 법선 방향 속력
		REAL3 newVpNN = N * newVpN; // 새로운 법선 방향 속도


		REAL friction = (coefficientFriction * (newVpN - vpN) / Length(vpT));
		REAL3 newVpT = vpT * (1 - friction);

		if (1 - friction <= 0.05)
			newVpT = make_REAL3(0, 0, 0);

		ver.d_Vel[idx] = newVpNN; //속도 업데이트

		ver.d_vSaturation[idx] += 0.000001f;
	}
}

__global__ void UpdateFaceAABB(Face face, Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	uint iv0 = face.d_faceIdx[idx].x;
	uint iv1 = face.d_faceIdx[idx].y;
	uint iv2 = face.d_faceIdx[idx].z;

	REAL3 v0 = ver.d_Pos1[iv0];
	REAL3 v1 = ver.d_Pos1[iv1];
	REAL3 v2 = ver.d_Pos1[iv2];

	setAABB(face.d_faceAABB[idx], make_REAL3(100.0f, 100.0f, 100.0f), make_REAL3(-100.0f, -100.0f, -100.0f));
	addAABB(face.d_faceAABB[idx], v0);
	addAABB(face.d_faceAABB[idx], v1);
	addAABB(face.d_faceAABB[idx], v2);
}

__global__ void Colide_PT(Face face, Vertex ver, Device_Hash hash, REAL* impulse)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	uint iv0 = face.d_faceIdx[idx].x;
	uint iv1 = face.d_faceIdx[idx].y;
	uint iv2 = face.d_faceIdx[idx].z;

	REAL3 p0 = ver.d_Pos1[iv0];
	REAL3 p1 = ver.d_Pos1[iv1];
	REAL3 p2 = ver.d_Pos1[iv2];

	REAL cellSize = 1.0 / clothParam._gridRes;

	int3 gridMinPos = calculateGridPos(face.d_faceAABB[idx]._min, cellSize);
	int3 gridMaxPos = calculateGridPos(face.d_faceAABB[idx]._max, cellSize);

	for (int xi = gridMinPos.x - 1; xi <= gridMaxPos.x + 1; xi++)
	{
		for (int yi = gridMinPos.y - 1; yi <= gridMaxPos.y + 1; yi++)
		{
			for (int zi = gridMinPos.z - 1; zi <= gridMaxPos.z + 1; zi++)
			{
				int3 neighbourPos = make_int3(xi, yi, zi);
				uint gridHash = calculateGridHash(neighbourPos, clothParam._gridRes);
				uint startIndex = hash.d_cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					uint endIndex = hash.d_cellEnd[gridHash];

					REAL u, v, w;
					for (uint j = startIndex; j < endIndex; j++)
					{
						uint sortedIdx = hash.d_gridIdx[j];

						if (sortedIdx == iv0 || sortedIdx == iv1 || sortedIdx == iv2) continue;

						REAL3 p = make_REAL3(ver.d_Pos1[sortedIdx]);

						REAL u, v, w;

						//제약 조건 투영
						REAL3 v0 = p1 - p0;
						REAL3 v1 = p2 - p0;
						REAL3 v2 = p - p0;

						REAL dot00 = Dot(v0, v0);
						REAL dot01 = Dot(v0, v1);
						REAL dot02 = Dot(v0, v2);
						REAL dot11 = Dot(v1, v1);
						REAL dot12 = Dot(v1, v2);

						REAL invDenom = 1 / (dot00 * dot11 - dot01 * dot01);

						REAL tempU = (dot11 * dot02 - dot01 * dot12) * invDenom;
						REAL tempV = (dot00 * dot12 - dot01 * dot02) * invDenom;

						if (!(tempU >= 0 && tempV >= 0 && tempU + tempV <= 1))
							continue;

						u = tempU;
						v = tempV;
						w = 1 - u - v;

						REAL3 pc = p0 + v0 * u + v1 * v;

						REAL distance = Length(p - pc);

						if (distance > clothParam._thickness) // 설정 거리보다 멀리 있을 경우 투영 X
							continue;

						REAL3 n = (p - pc);
						Normalize(n);

						//barycentric coord 이용 보간
						REAL c = Dot(n, p - p0) - clothParam._thickness;

						REAL a0 = w;
						REAL a1 = u;
						REAL a2 = v;
						REAL scale = c / (pow(a0, 2) + pow(a1, 2) + pow(a2, 2) + 1); //정점의 무게가 다 같다고 가정

						REAL3 gP = n;
						REAL3 gP0 = n * a0 * (-1);
						REAL3 gP1 = n * a1 * (-1);
						REAL3 gP2 = n * a2 * (-1);

						REAL3 dP = gP * scale * (-1);
						REAL3 dP0 = gP0 * scale * (-1);
						REAL3 dP1 = gP1 * scale * (-1);
						REAL3 dP2 = gP2 * scale * (-1);

						atomicAdd_REAL(impulse + sortedIdx * 3u + 0u, dP.x);
						atomicAdd_REAL(impulse + sortedIdx * 3u + 1u, dP.y);
						atomicAdd_REAL(impulse + sortedIdx * 3u + 2u, dP.z);

						atomicAdd_REAL(impulse + iv0 * 3u + 0u, dP0.x);
						atomicAdd_REAL(impulse + iv0 * 3u + 1u, dP0.y);
						atomicAdd_REAL(impulse + iv0 * 3u + 2u, dP0.z);

						atomicAdd_REAL(impulse + iv1 * 3u + 0u, dP1.x);
						atomicAdd_REAL(impulse + iv1 * 3u + 1u, dP1.y);
						atomicAdd_REAL(impulse + iv1 * 3u + 2u, dP1.z);

						atomicAdd_REAL(impulse + iv2 * 3u + 0u, dP2.x);
						atomicAdd_REAL(impulse + iv2 * 3u + 1u, dP2.y);
						atomicAdd_REAL(impulse + iv2 * 3u + 2u, dP2.z);
					}
				}
			}
		}
	}
}

__global__ void ApplyImpulse_kernel(Vertex ver, REAL* impulse)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL3 im;
	im.x = impulse[idx * 3 + 0u] * clothParam._selfColliDamping;
	im.y = impulse[idx * 3 + 1u] * clothParam._selfColliDamping;
	im.z = impulse[idx * 3 + 2u] * clothParam._selfColliDamping;

	ver.d_Pos1[idx] += im;
}

__global__ void WaterAbsorption_Kernel(Face face, Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	uint iv0 = face.d_faceIdx[idx].x;
	uint iv1 = face.d_faceIdx[idx].y;
	uint iv2 = face.d_faceIdx[idx].z;

	REAL3 v0 = ver.d_Pos[iv0];
	REAL3 v1 = ver.d_Pos[iv1];
	REAL3 v2 = ver.d_Pos[iv2];

	REAL area = (Length(Cross(v1 - v0, v2 - v0))) / 2;
	REAL mass = ver.d_vSaturation[iv0] + ver.d_vSaturation[iv1] + ver.d_vSaturation[iv2];
	REAL abMass = clothParam._absorptionK * (clothParam._maxsaturation - face.d_fSaturation[idx]) * area;
	REAL resiMass = mass - abMass;

	if (resiMass > 0)
	{
		face.d_fSaturation[idx] += abMass / area;
		//Particle 질량 resiMass로 변경
	}
	else
	{
		face.d_fSaturation[idx] += mass / area;
		//Particle 시뮬레이션에서 제외
	}
}

__global__ void WaterDiffusion_Kernel(Face face, Vertex ver, REAL* delS)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	uint iv0 = face.d_faceIdx[idx].x;
	uint iv1 = face.d_faceIdx[idx].y;
	uint iv2 = face.d_faceIdx[idx].z;

	REAL3 v0 = ver.d_Pos[iv0];
	REAL3 v1 = ver.d_Pos[iv1];
	REAL3 v2 = ver.d_Pos[iv2];

	REAL3 cl = (v0 + v1 + v2) / 3;
	REAL sSum = 0;

	uint numNbFaces = face.d_nbFace._index[idx + 1] - face.d_nbFace._index[idx];
	REAL sln[3];
	REAL sl = face.d_fSaturation[idx];

	for (int i = 0; i < numNbFaces; i++)
	{
		uint fIdx = face.d_nbFace._array[face.d_nbFace._index[idx] + i];

		uint nbiv0 = face.d_faceIdx[fIdx].x;
		uint nbiv1 = face.d_faceIdx[fIdx].y;
		uint nbiv2 = face.d_faceIdx[fIdx].z;

		REAL3 nbv0 = ver.d_Pos[nbiv0];
		REAL3 nbv1 = ver.d_Pos[nbiv1];
		REAL3 nbv2 = ver.d_Pos[nbiv2];

		REAL3 cn = (nbv0 + nbv1 + nbv2) / 3;
		REAL sn = face.d_fSaturation[fIdx];
		REAL cosln = Dot(cl, cn) / (Length(cl) * Length(cn));
		sln[i] = min(0.0, clothParam._diffusK * (sl - sn) + clothParam._gravityDiffusK * cosln);
		sSum += sln[i];

		if (cosln > 0)
			atomicAdd_REAL(face.d_fDripbuf + fIdx, face.d_fDripbuf[idx] * cosln);

		//Squeeze Cloth
		//REAL3 clr = (ver.d_restPos[idx] + ver.d_restPos[idx] + ver.d_restPos[idx]) / 3;
		//REAL3 cnr = (ver.d_restPos[fIdx] + ver.d_restPos[fIdx] + ver.d_restPos[fIdx]) / 3;

		//REAL reducethres = max(0.0, Length(clr - cnr) - Length(cl - cn));
		//face.d_fDripThres[idx] = max(0.0, face.d_fDripThres[idx] - reducethres * 0.01f);
	}

	REAL norm = 1;
	if (sSum > sl)
		norm = sl / sSum;

	REAL areal = (Length(Cross(v1 - v0, v2 - v0))) / 2;
	for (int i = 0; i < numNbFaces; i++)
	{
		uint fIdx = face.d_nbFace._array[face.d_nbFace._index[idx] + i];

		uint nbiv0 = face.d_faceIdx[fIdx].x;
		uint nbiv1 = face.d_faceIdx[fIdx].y;
		uint nbiv2 = face.d_faceIdx[fIdx].z;

		REAL3 nbv0 = ver.d_Pos[nbiv0];
		REAL3 nbv1 = ver.d_Pos[nbiv1];
		REAL3 nbv2 = ver.d_Pos[nbiv2];

		REAL arean = (Length(Cross(nbv1 - nbv0, nbv2 - nbv0))) / 2;

		delS[idx] -= norm * sln[i];
		atomicAdd_REAL(delS + fIdx, norm * sln[i] * (areal / arean));
	}
}

__global__ void ApplyDeltaSaturation_Kernel(Face face, REAL* delS)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	face.d_fSaturation[idx] += delS[idx];
}

__global__ void WaterDripping_Kernel(Face face, Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numFaces)
		return;

	if (clothParam._maxsaturation < face.d_fSaturation[idx])
	{
		uint iv0 = face.d_faceIdx[idx].x;
		uint iv1 = face.d_faceIdx[idx].y;
		uint iv2 = face.d_faceIdx[idx].z;

		REAL3 v0 = ver.d_Pos[iv0];
		REAL3 v1 = ver.d_Pos[iv1];
		REAL3 v2 = ver.d_Pos[iv2];

		REAL area = (Length(Cross(v1 - v0, v2 - v0))) / 2;
		REAL exMass = (face.d_fSaturation[idx] - clothParam._maxsaturation) * area;
		face.d_fDripbuf[idx] += exMass;

		while (face.d_fDripbuf[idx] > face.d_fDripThres[idx])
		{
			//Particle 생성 및 초기화 추가
			face.d_fDripbuf[idx] -= 0.01;
		}

		face.d_fSaturation[idx] = clothParam._maxsaturation;
	}
}

__global__ void UpdateVertexMass_Kernel(Face face, Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	uint numNbFaces = ver.d_nbFaces._index[idx + 1] - ver.d_nbFaces._index[idx];
	REAL addMass = 0.0;
	REAL vSat = 0.0;

	for (int i = 0; i < numNbFaces; i++)
	{
		uint fIdx = ver.d_nbFaces._array[ver.d_nbFaces._index[idx] + i];

		uint iv0 = face.d_faceIdx[fIdx].x;
		uint iv1 = face.d_faceIdx[fIdx].y;
		uint iv2 = face.d_faceIdx[fIdx].z;

		REAL3 v0 = ver.d_Pos[iv0];
		REAL3 v1 = ver.d_Pos[iv1];
		REAL3 v2 = ver.d_Pos[iv2];

		REAL area = (Length(Cross(v1 - v0, v2 - v0))) / 2;

		addMass += (face.d_fSaturation[fIdx] * area) / 3;
		vSat += face.d_fSaturation[fIdx] / 3;
	}

	ver.d_SatMass[idx] = addMass * 20;
	ver.d_vSaturation[idx] = vSat;
}

__global__ void UpdateAdhesionForce_kernel(Vertex ver, Device_Hash hash, REAL* adhes)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL cellSize = 1.0 / clothParam._gridRes;
	int3 gridPos = calculateGridPos(ver.d_Pos1[idx], cellSize);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);

				uint neighHash = calculateGridHash(neighbourPos, clothParam._gridRes);
				uint startIdx = hash.d_cellStart[neighHash];

				if (startIdx != 0xffffffff)
				{
					uint endIdx = hash.d_cellEnd[neighHash];

					for (uint i = startIdx; i < endIdx; i++)
					{
						uint id0 = idx;
						uint id1 = hash.d_gridIdx[i];

						if (id0 != id1)
						{

							REAL3 diffPos = ver.d_Pos1[id0] - ver.d_Pos1[id1];
							REAL dist2 = LengthSquared(diffPos);
							REAL thickness2 = clothParam._adhesionThickness * clothParam._adhesionThickness;

							if (dist2 > thickness2) continue;

							REAL restDist2 = LengthSquared(ver.d_restPos[id0] - ver.d_restPos[id1]);
							REAL minDist = clothParam._adhesionThickness;

							if (dist2 > restDist2) continue;

							if (restDist2 < thickness2)
								minDist = sqrt(restDist2);

							REAL dist = sqrt(dist2);

							REAL pi = 3.14159265358979;
							REAL theta = (dist / minDist) * 90;
							REAL rad = theta * (pi / 180);
							REAL r0 = 0;
							REAL r1 = 0;

							uint numNbVer0 = ver.d_nbVertices._index[id0 + 1] - ver.d_nbVertices._index[id0];
							uint numNbVer1 = ver.d_nbVertices._index[id1 + 1] - ver.d_nbVertices._index[id1];

							for (int i = 0; i < numNbVer0; i++)
							{
								uint nid0 = ver.d_nbVertices._array[ver.d_nbVertices._index[id0] + i];
								r0 += Length(ver.d_Pos1[nid0] - ver.d_Pos1[id0]) / 2;
							}

							r0 /= numNbVer0;

							for (int i = 0; i < numNbVer1; i++)
							{
								uint nid1 = ver.d_nbVertices._array[ver.d_nbVertices._index[id1] + i];
								r1 += Length(ver.d_Pos1[nid1] - ver.d_Pos1[id1]) / 2;
							}

							r1 /= numNbVer1;

							Normalize(diffPos);
							REAL3 adhesion0 = -1 * diffPos * (pi * r0 * r0 * clothParam._surfTension * (((cos(rad) + cos(rad)) / minDist) - (sin(rad) / r0)) + (2 * pi * r0 * clothParam._surfTension * sin(rad)));
							REAL3 adhesion1 = diffPos * (pi * r1 * r1 * clothParam._surfTension * (((cos(rad) + cos(rad)) / minDist) - (sin(rad) / r0)) + (2 * pi * r1 * clothParam._surfTension * sin(rad)));
							atomicAdd_REAL(adhes + id0 * 3u + 0u, adhesion0.x);
							atomicAdd_REAL(adhes + id0 * 3u + 1u, adhesion0.y);
							atomicAdd_REAL(adhes + id0 * 3u + 2u, adhesion0.z);

							atomicAdd_REAL(adhes + id1 * 3u + 0u, adhesion1.x);
							atomicAdd_REAL(adhes + id1 * 3u + 1u, adhesion1.y);
							atomicAdd_REAL(adhes + id1 * 3u + 2u, adhesion1.z);

						}
					}
				}
			}
		}
	}
}

__global__ void ApplyAdhesion_kernel(Vertex ver, REAL* adhes)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	if (ver.d_vSaturation[idx] < 0.1) return;

	REAL3 adhesion;
	adhesion.x = adhes[idx * 3 + 0u];
	adhesion.y = adhes[idx * 3 + 1u];
	adhesion.z = adhes[idx * 3 + 2u];

	ver.d_Adhesion[idx] = adhesion;
}

__device__ REAL GetInvm(REAL invm, REAL satm)
{
	return (1.0 / ((1.0 / invm) + satm));
}

__device__ REAL GetCot(REAL3 dir0, REAL3 dir1)
{
	Normalize(dir0);
	Normalize(dir1);
	REAL sin_x = Length(Cross(dir0, dir1));
	REAL cos_x = Dot(dir0, dir1);
	return cos_x / sin_x;
}

__device__ REAL ApplyLogisticFunc(REAL value, REAL gamma, REAL mu, REAL sigma)
{
	return gamma / (1 + exp(((value - mu) * -1) / sigma));
}

__global__ void SolveAngleConstraint_kernel(Edge edge, Vertex ver, Face face, REAL* dp)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numEdges)
		return;

	if (edge.d_nbEFaces._index[idx + 1] - edge.d_nbEFaces._index[idx] != 2)
		return;

	uint edgeSum = edge.d_edgeIdx[idx].x + edge.d_edgeIdx[idx].y;
	uint fId0 = edge.d_nbEFaces._array[edge.d_nbEFaces._index[idx]];
	uint fId1 = edge.d_nbEFaces._array[edge.d_nbEFaces._index[idx] + 1];

	uint vId0 = face.d_faceIdx[fId0].x + face.d_faceIdx[fId0].y + face.d_faceIdx[fId0].z - edgeSum;
	uint vId1 = face.d_faceIdx[fId1].x + face.d_faceIdx[fId1].y + face.d_faceIdx[fId1].z - edgeSum;
	uint vId2 = edge.d_edgeIdx[idx].x;
	uint vId3 = edge.d_edgeIdx[idx].y;

	REAL3 p0 = ver.d_Pos1[vId0];
	REAL3 p1 = ver.d_Pos1[vId1];
	REAL3 p2 = ver.d_Pos1[vId2];
	REAL3 p3 = ver.d_Pos1[vId3];

	REAL invm0 = GetInvm(ver.d_InvMass[vId0], ver.d_SatMass[vId0]);
	REAL invm1 = GetInvm(ver.d_InvMass[vId1], ver.d_SatMass[vId1]);
	REAL invm2 = GetInvm(ver.d_InvMass[vId2], ver.d_SatMass[vId2]);
	REAL invm3 = GetInvm(ver.d_InvMass[vId3], ver.d_SatMass[vId3]);

	REAL3 e = p3 - p2;
	REAL length = Length(e);
	if (length < 0.001)
		return;

	REAL invlength = 1.0 / length;
	REAL3 n1 = Cross(p2 - p0, p3 - p0);
	REAL3 n2 = Cross(p3 - p1, p2 - p1);
	n1 /= LengthSquared(n1);
	n2 /= LengthSquared(n2);

	REAL3 d0 = n1 * length;
	REAL3 d1 = n2 * length;
	REAL3 d2 = n1 * (Dot(p0 - p3, e) * invlength) + n2 * (Dot(p1 - p3, e) * invlength);
	REAL3 d3 = n1 * (Dot(p2 - p0, e) * invlength) + n2 * (Dot(p2 - p1, e) * invlength);

	Normalize(n1);
	Normalize(n2);
	REAL dot = Dot(n1, n2);

	if (dot < -1.0)
		dot = -1.0;
	if (dot > 1.0)
		dot = 1.0;

	REAL phi = acos(dot);

	REAL lambda = invm0 * LengthSquared(d0) +
		invm1 * LengthSquared(d1) +
		invm2 * LengthSquared(d2) +
		invm3 * LengthSquared(d3);

	if (lambda == 0.0)
	{
		return;
	}

	REAL stiffness = 0.05;
	REAL pi = 3.14159265358979;

	REAL cotA = GetCot(p2 - p0, p3 - p0);
	REAL cotB = GetCot(p2 - p1, p3 - p1);
	REAL weight = 0.5 * (cotA + cotB);

	REAL satSum = face.d_fSaturation[fId0] + face.d_fSaturation[fId1];
	REAL ratio = ApplyLogisticFunc(satSum / (clothParam._maxsaturation * 2), 1, 0.4, 0.04);
	//REAL ratio = satSum / (clothParam._maxsaturation * 2);
	REAL restAngle = phi;

	restAngle += phi * 0.05 * ratio;
	//restAngle *= weight * ratio;

	lambda = (phi - restAngle) / lambda * stiffness;

	if (Dot(e, Cross(n1, n2))> 0.0)
	{
		lambda = -lambda;
	}

	atomicAdd_REAL(dp + vId0 * 3u + 0u, (d0 * (-invm0 * lambda)).x);
	atomicAdd_REAL(dp + vId0 * 3u + 1u, (d0 * (-invm0 * lambda)).y);
	atomicAdd_REAL(dp + vId0 * 3u + 2u, (d0 * (-invm0 * lambda)).z);

	atomicAdd_REAL(dp + vId1 * 3u + 0u, (d1 * (-invm1 * lambda)).x);
	atomicAdd_REAL(dp + vId1 * 3u + 1u, (d1 * (-invm1 * lambda)).y);
	atomicAdd_REAL(dp + vId1 * 3u + 2u, (d1 * (-invm1 * lambda)).z);

	atomicAdd_REAL(dp + vId2 * 3u + 0u, (d2 * (-invm2 * lambda)).x);
	atomicAdd_REAL(dp + vId2 * 3u + 1u, (d2 * (-invm2 * lambda)).y);
	atomicAdd_REAL(dp + vId2 * 3u + 2u, (d2 * (-invm2 * lambda)).z);

	atomicAdd_REAL(dp + vId3 * 3u + 0u, (d3 * (-invm3 * lambda)).x);
	atomicAdd_REAL(dp + vId3 * 3u + 1u, (d3 * (-invm3 * lambda)).y);
	atomicAdd_REAL(dp + vId3 * 3u + 2u, (d3 * (-invm3 * lambda)).z);
}

__global__ void ApplyConstraintDeltaPos_kernel(Vertex ver, REAL* dp)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL3 delta;
	delta.x = dp[idx * 3 + 0u];
	delta.y = dp[idx * 3 + 1u];
	delta.z = dp[idx * 3 + 2u];

	ver.d_Pos1[idx] += delta;
}

__global__ void ComputeAngle_kernel(Edge edge, Vertex ver, Face face)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numEdges)
		return;

	if (edge.d_nbEFaces._index[idx + 1] - edge.d_nbEFaces._index[idx] != 2)
		return;

	uint edgeSum = edge.d_edgeIdx[idx].x + edge.d_edgeIdx[idx].y;
	uint fId0 = edge.d_nbEFaces._array[edge.d_nbEFaces._index[idx]];
	uint fId1 = edge.d_nbEFaces._array[edge.d_nbEFaces._index[idx] + 1];

	uint vId0 = face.d_faceIdx[fId0].x + face.d_faceIdx[fId0].y + face.d_faceIdx[fId0].z - edgeSum;
	uint vId1 = face.d_faceIdx[fId1].x + face.d_faceIdx[fId1].y + face.d_faceIdx[fId1].z - edgeSum;
	uint vId2 = edge.d_edgeIdx[idx].x;
	uint vId3 = edge.d_edgeIdx[idx].y;

	REAL3 p0 = ver.d_Pos1[vId0];
	REAL3 p1 = ver.d_Pos1[vId1];
	REAL3 p2 = ver.d_Pos1[vId2];
	REAL3 p3 = ver.d_Pos1[vId3];

	REAL invm0 = GetInvm(ver.d_InvMass[vId0], ver.d_SatMass[vId0]);
	REAL invm1 = GetInvm(ver.d_InvMass[vId1], ver.d_SatMass[vId1]);
	REAL invm2 = GetInvm(ver.d_InvMass[vId2], ver.d_SatMass[vId2]);
	REAL invm3 = GetInvm(ver.d_InvMass[vId3], ver.d_SatMass[vId3]);

	REAL3 e = p3 - p2;
	REAL length = Length(e);
	if (length < 0.001)
		return;

	REAL invlength = 1.0 / length;
	REAL3 n1 = Cross(p2 - p0, p3 - p0);
	REAL3 n2 = Cross(p3 - p1, p2 - p1);
	n1 /= LengthSquared(n1);
	n2 /= LengthSquared(n2);

	REAL3 d0 = n1 * length;
	REAL3 d1 = n2 * length;
	REAL3 d2 = n1 * (Dot(p0 - p3, e) * invlength) + n2 * (Dot(p1 - p3, e) * invlength);
	REAL3 d3 = n1 * (Dot(p2 - p0, e) * invlength) + n2 * (Dot(p2 - p1, e) * invlength);

	Normalize(n1);
	Normalize(n2);
	REAL dot = Dot(n1, n2);

	if (dot < -1.0)
		dot = -1.0;
	if (dot > 1.0)
		dot = 1.0;

	REAL angle = acos(dot);

	atomicAdd_REAL(ver.d_vAngle + vId2, angle);
	atomicAdd_REAL(ver.d_vAngle + vId3, angle);
}

__global__ void ComputeLaplacian_kernel(Vertex ver, Face face)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL lambda = 0.06;

	REAL3 pos = ver.d_Pos[idx];
	REAL3 sumPos = make_REAL3(0.0, 0.0, 0.0);

	uint fid0 = ver.d_nbFaces._index[idx];
	uint fid1 = ver.d_nbFaces._index[idx + 1];
	uint numnbF = fid1 - fid0;
	REAL satSum = 0.0f;
	for (int j = 0; j < fid1 - fid0; j++)
	{
		uint fIdx = ver.d_nbFaces._array[ver.d_nbFaces._index[idx] + j];
		satSum += face.d_fSaturation[fIdx];
	}

	uint id0 = ver.d_nbVertices._index[idx];
	uint id1 = ver.d_nbVertices._index[idx + 1];
	uint numnbV = id1 - id0;
	ver.d_vAngle[idx] /= numnbV;
	REAL weight = ver.d_vAngle[idx] * (satSum / (clothParam._maxsaturation * numnbF));
	for (int j = 0; j < numnbV; j++)
	{
		uint vIdx = ver.d_nbVertices._array[ver.d_nbVertices._index[idx] + j];
		sumPos += ver.d_Pos[vIdx];
	}
	REAL3 finalPos = (sumPos / numnbV - pos) * -1 * lambda * weight;

	ver.d_Vel[idx] += finalPos / clothParam._subdt;
}