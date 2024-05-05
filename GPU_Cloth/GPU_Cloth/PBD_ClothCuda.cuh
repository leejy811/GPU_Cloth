#include "PBD_ClothCuda.h"
#include "include/CUDA_Custom/DeviceManager.cuh"

__constant__ ClothParam clothParam;

__global__ void CompGravity_kernel(Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL3 v = ver.d_Vel[idx];
	REAL3 gravity = make_REAL3(0.0, clothParam._gravity, 0.0);
	v += gravity * clothParam._subdt;
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

	//if (ver.d_Pos1[idx].y > 0.9   && ver.d_Pos1[idx].x > 0.9)
	//	return;

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

__device__ double SDFCalculate(double x, double y, double z)
{
	//ax + by + cz + d = 0 평면 방정식
	double a = 0.0;
	double b = 2.0;
	double c = 0.0;
	double d = -1.0;

	return (a * x + b * y + c * z + d) / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
}

__device__ double SDFCalculate(REAL3 p)
{
	return SDFCalculate(p.x, p.y, p.z);
}

__global__ void LevelSetCollision_D(Vertex ver)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= clothParam._numVertices)
		return;

	REAL deltaT = 0.01f;
	REAL h = 0.1f;
	REAL coefficientFriction = 0.1f;

	REAL x = ver.d_Pos[idx].x;
	REAL y = ver.d_Pos[idx].y;
	REAL z = ver.d_Pos[idx].z;

	REAL3 N = make_REAL3(SDFCalculate(x + h, y, z) - SDFCalculate(x, y, z),
		SDFCalculate(x, y + h, z) - SDFCalculate(x, y, z),
		SDFCalculate(x, y, z + h) - SDFCalculate(x, y, z));
	//N.print();

	N /= h; //법선 벡터 계산 (오일러 방법 이용) = Gradient PI
	Normalize(N);

	REAL pi = SDFCalculate(x, y, z); //PI, newPI 계산
	REAL newPI = pi + Dot((ver.d_Vel[idx] * deltaT), N);

	if (newPI < 0)
	{
		REAL vpN = Dot(ver.d_Vel[idx], N); //원래의 법선 방향 속력
		REAL3 vpNN = N * vpN; //원래의 법선 방향 속도
		REAL3 vpT = ver.d_Vel[idx] - vpNN; //원래의 접선 방향 속도

		double newVpN = vpN - (newPI / deltaT); //새로운 법선 방향 속력
		REAL3 newVpNN = N * newVpN; // 새로운 법선 방향 속도


		double friction = (coefficientFriction * (newVpN - vpN) / Length(vpT));
		REAL3 newVpT = vpT * (1 - friction);

		if (1 - friction < 0)
			newVpT = make_REAL3(0, 0, 0);

		ver.d_Vel[idx] = newVpNN + newVpT; //속도 업데이트
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