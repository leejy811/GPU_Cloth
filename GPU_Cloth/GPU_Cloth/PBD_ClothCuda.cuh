#include "PBD_ClothCuda.h"
#include "include/CUDA_Custom/DeviceManager.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void CompExternlaForce_kernel(REAL3* pos, REAL3* pos1, REAL3* vel, REAL* invm, REAL3 gravity, REAL3 ext, REAL damp, uint numVer, REAL dt, REAL thickness)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	REAL3 v = vel[idx];
	v += gravity * dt;
	v += ext * invm[idx] * dt;
	v *= damp;
	vel[idx] = v;

	REAL invdt = 1.0 / dt;
	REAL speed = Length(vel[idx]);
	REAL maxSpeed = thickness * invdt;

	if (speed > maxSpeed)
		vel[idx] *= (maxSpeed / speed);

	pos1[idx] = pos[idx] + (vel[idx] * dt);
}

__global__ void CompWind_kernel(uint3* fIdx, REAL3* pos1, REAL3* vel, REAL3 wind, uint numFace)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numFace)
		return;

	uint iv0 = fIdx[idx].x;
	uint iv1 = fIdx[idx].y;
	uint iv2 = fIdx[idx].z;

	REAL3 v0 = pos1[iv0];
	REAL3 v1 = pos1[iv1];
	REAL3 v2 = pos1[iv2];

	REAL3 normal = Cross(v1 - v0, v2 - v0);
	Normalize(normal);
	REAL3 force = normal * Dot(normal, wind);
	vel[iv0] += force;
	vel[iv1] += force;
	vel[iv2] += force;
}

__global__ void CompIntergrate_kernel(REAL3* pos, REAL3* pos1, REAL3* vel, uint numVer, REAL thickness, REAL invdt)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	//if (pos1[idx].y > 0.9   && pos1[idx].x > 0.9)
	//	return;

	vel[idx] = (pos1[idx] - pos[idx]) * invdt;

	REAL speed = Length(vel[idx]);
	REAL maxSpeed = thickness * invdt;

	if (speed > maxSpeed)
		vel[idx] *= (maxSpeed / speed);

	REAL dt = 1.0 / invdt;
	pos[idx] += vel[idx] * dt;

	//pos[idx] = pos1[idx];

	//if (pos[idx].x > 1.0)	pos[idx].x = 1.0;
	//if (pos[idx].y > 1.0)	pos[idx].y = 1.0;
	//if (pos[idx].z > 1.0)	pos[idx].z = 1.0;

	//if (pos[idx].x < -1.0)	pos[idx].x = -1.0;
	//if (pos[idx].y < -1.0)	pos[idx].y = -1.0;
	//if (pos[idx].z < -1.0)	pos[idx].z = -1.0;
}

__global__ void CompFaceNorm_kernel(uint3* fIdx, REAL3* pos, REAL3* fNorm, uint numFace)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numFace)
		return;

	uint iv0 = fIdx[idx].x;
	uint iv1 = fIdx[idx].y;
	uint iv2 = fIdx[idx].z;

	REAL3 v0 = pos[iv0];
	REAL3 v1 = pos[iv1];
	REAL3 v2 = pos[iv2];

	REAL3 norm = Cross(v1 - v0, v2 - v0);
	Normalize(norm);
	fNorm[idx] = norm;
}

__global__ void CompVertexNorm_kernel(uint* nbFIdx, uint* nbFArray, REAL3* fNorm, REAL3* vNorm, uint numVer)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	uint numNbFaces = nbFIdx[idx + 1] - nbFIdx[idx];

	for (int i = 0; i < numNbFaces; i++)
	{
		uint fIdx = nbFArray[nbFIdx[idx] + i];
		vNorm[idx] += fNorm[fIdx];
	}
	vNorm[idx] /= numNbFaces;
	Normalize(vNorm[idx]);
}

__device__ int3 calcGridPos(REAL3 pos, REAL gridSize)
{
	int3 intPos = make_int3(floorf(pos.x / gridSize), floorf(pos.y / gridSize), floorf(pos.z / gridSize));
	return intPos;
}

__device__ uint calcGridHash(int3 pos, uint gridRes)
{
	pos.x = pos.x &
		(gridRes - 1);  // wrap grid, assumes size is power of 2
	pos.y = pos.y & (gridRes - 1);
	pos.z = pos.z & (gridRes - 1);

	return __umul24(__umul24(pos.z, gridRes), gridRes) +
		__umul24(pos.y, gridRes) + pos.x;
}

__global__ void CalculateHash_D(uint* gridHash, uint* gridIdx, REAL3* pos, uint gridRes, uint numVer)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);
	uint hash = calcGridHash(gridPos, gridRes);

	gridHash[idx] = hash;
	gridIdx[idx] = idx;
}

__global__ void FindCellStart_D(uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL3* pos, REAL3* vel, REAL3* sortedPos, REAL3* sortedVel, uint numParticles)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[];
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	uint hash;

	if (idx < numParticles)
	{
		hash = gridHash[idx];
		sharedHash[threadIdx.x + 1] = hash;

		if (idx > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = gridHash[idx - 1];
		}
	}

	cg::sync(cta);

	if (idx < numParticles)
	{

		if (idx == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = idx;

			if (idx > 0) cellEnd[sharedHash[threadIdx.x]] = idx;
		}

		if (idx == numParticles - 1)
		{
			cellEnd[hash] = idx + 1;
		}

		uint sortedIndex = gridIdx[idx];
		REAL3 newPos = pos[sortedIndex];
		REAL3 newVel = vel[sortedIndex];

		sortedPos[idx] = newPos;
		sortedVel[idx] = newVel;
	}
}

__device__ REAL3 ColideSphere(REAL3 restPosA, REAL3 restPosB, REAL3 posA, REAL3 posB, REAL3 velA, REAL3 velB, REAL thickness, REAL damping)
{
	// calculate relative position
	REAL3 relPos = posB - posA;

	float dist = Length(relPos);
	float collideDist = thickness + thickness;
	float restDist = Length(restPosA - restPosB);

	REAL3 force = make_REAL3(0.0f);

	REAL spring = 0.5f;
	REAL shear = 0.1f;
	REAL attraction = 0.0f;

	if (dist < collideDist)
	{
		if (dist > restDist) return force;

		if (restDist < collideDist)
			collideDist = restDist;

		REAL3 norm = relPos / dist;

		// relative velocity
		REAL3 relVel = velB - velA;

		// relative tangential velocity
		REAL3 tanVel = relVel - (Dot(relVel, norm) * norm);

		// spring force
		force = -spring * (collideDist - dist) * norm;
		// dashpot (damping) force
		force += damping * relVel;
		// tangential shear force
		force += shear * tanVel;
		// attraction
		force += attraction * relPos;
	}

	return force;
}

__device__ REAL3 collideCell(REAL thickness, REAL damping, uint gridRes, int3 gridPos, uint index, REAL3 pos, REAL3 vel, REAL3* oldPos, REAL3* oldVel, uint* cellStart, uint* cellEnd)
{
	uint gridHash = calcGridHash(gridPos, gridRes);

	// get start of bucket for this cell
	uint startIndex = cellStart[gridHash];

	REAL3 force = make_REAL3(0.0f);

	if (startIndex != 0xffffffff)  // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = cellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)  // check not colliding with self
			{
				REAL3 pos2 = make_REAL3(oldPos[j]);
				REAL3 vel2 = make_REAL3(oldVel[j]);

				//force += ColideSphere(pos, pos2, vel, vel2, thickness, damping);
			}
		}
	}

	return force;
}

__global__ void Colide_PP(REAL3* restPos, REAL3* pos1, REAL3* pos, REAL3* sortedVel, REAL3* vel, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL* impulse, REAL thickness, uint gridRes, uint numVer, BOOL* flag)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	REAL cellSize = 1.0 / gridRes;
	int3 gridPos = calcGridPos(pos1[idx], cellSize);
	uint hash = calcGridHash(gridPos, gridRes);
	int cnt = 0;
	REAL3 force = make_REAL3(0.0f);

	flag[idx] = false;

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = make_int3(gridPos.x + x, gridPos.y + y, gridPos.z + z);

				uint neighHash = calcGridHash(neighbourPos, gridRes);
				uint startIdx = cellStart[neighHash];

				if (startIdx != 0xffffffff)
				{
					uint endIdx = cellEnd[neighHash];

					for (uint i = startIdx; i < endIdx; i++)
					{
						uint id0 = idx;
						uint id1 = gridIdx[i];

						if (id0 != id1)
						{

							REAL3 diffPos = pos1[id0] - pos1[id1];
							REAL dist2 = LengthSquared(diffPos);
							REAL thickness2 = thickness * thickness;

							if (dist2 > thickness2) continue;

							REAL restDist2 = LengthSquared(restPos[id0] - restPos[id1]);
							REAL minDist = thickness;

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

							REAL3 v0 = (pos1[id0] - pos[id0]);
							REAL3 v1 = (pos1[id1] - pos[id1]);

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

				//uint gridHash = calcGridHash(neighbourPos, gridRes);
				//uint startIndex = cellStart[gridHash];

				//if (startIndex != 0xffffffff)
				//{
				//	uint endIndex = cellEnd[gridHash];

				//	for (uint j = startIndex; j < endIndex; j++)
				//	{
				//		if (j != idx)
				//		{
				//			REAL3 pos2 = make_REAL3(pos[j]);
				//			REAL3 vel2 = make_REAL3(sortedVel[j]);

				//			uint id0 = gridIdx[idx];
				//			uint id1 = gridIdx[j];

				//			force += ColideSphere(restPos[id0], restPos[id1], pos[idx], pos2, sortedVel[idx], vel2, thickness, damping);
				//		}
				//	}
				//}
			}
		}
	}

	//uint sortedIdx = gridIdx[idx];
	//vel[sortedIdx] = sortedVel[idx] + force;
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

__global__ void LevelSetCollision_D(REAL3* pos, REAL3* vel, uint numVer)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	REAL deltaT = 0.01f;
	REAL h = 0.1f;
	REAL coefficientFriction = 0.1f;

	REAL x = pos[idx].x;
	REAL y = pos[idx].y;
	REAL z = pos[idx].z;

	REAL3 N = make_REAL3(SDFCalculate(x + h, y, z) - SDFCalculate(x, y, z),
		SDFCalculate(x, y + h, z) - SDFCalculate(x, y, z),
		SDFCalculate(x, y, z + h) - SDFCalculate(x, y, z));
	//N.print();

	N /= h; //법선 벡터 계산 (오일러 방법 이용) = Gradient PI
	Normalize(N);

	REAL pi = SDFCalculate(x, y, z); //PI, newPI 계산
	REAL newPI = pi + Dot((vel[idx] * deltaT), N);

	if (newPI < 0)
	{
		REAL vpN = Dot(vel[idx], N); //원래의 법선 방향 속력
		REAL3 vpNN = N * vpN; //원래의 법선 방향 속도
		REAL3 vpT = vel[idx] - vpNN; //원래의 접선 방향 속도

		double newVpN = vpN - (newPI / deltaT); //새로운 법선 방향 속력
		REAL3 newVpNN = N * newVpN; // 새로운 법선 방향 속도


		double friction = (coefficientFriction * (newVpN - vpN) / Length(vpT));
		REAL3 newVpT = vpT * (1 - friction);

		if (1 - friction < 0)
			newVpT = make_REAL3(0, 0, 0);

		vel[idx] = newVpNN + newVpT; //속도 업데이트
	}
}

__global__ void UpdateFaceAABB(uint3* fIdx, REAL3* pos, AABB* fAABB, uint numFace)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numFace)
		return;

	uint iv0 = fIdx[idx].x;
	uint iv1 = fIdx[idx].y;
	uint iv2 = fIdx[idx].z;

	REAL3 v0 = pos[iv0];
	REAL3 v1 = pos[iv1];
	REAL3 v2 = pos[iv2];

	setAABB(fAABB[idx], make_REAL3(100.0f, 100.0f, 100.0f), make_REAL3(-100.0f, -100.0f, -100.0f));
	addAABB(fAABB[idx], v0);
	addAABB(fAABB[idx], v1);
	addAABB(fAABB[idx], v2);
}

__global__ void Colide_PT(uint3* fIdx, REAL3* pos, AABB* fAABB, uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd, REAL* impulse, REAL thickness, uint gridRes, uint numFace, BOOL* flag)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numFace)
		return;

	uint iv0 = fIdx[idx].x;
	uint iv1 = fIdx[idx].y;
	uint iv2 = fIdx[idx].z;

	REAL3 p0 = pos[iv0];
	REAL3 p1 = pos[iv1];
	REAL3 p2 = pos[iv2];

	REAL cellSize = 1.0 / gridRes;
	
	int3 gridMinPos = calcGridPos(fAABB[idx]._min, cellSize);
	int3 gridMaxPos = calcGridPos(fAABB[idx]._max, cellSize);

	for (int xi = gridMinPos.x - 1; xi <= gridMaxPos.x + 1; xi++)
	{
		for (int yi = gridMinPos.y - 1; yi <= gridMaxPos.y + 1; yi++)
		{
			for (int zi = gridMinPos.z - 1; zi <= gridMaxPos.z + 1; zi++)
			{
				int3 neighbourPos = make_int3(xi, yi, zi);
				uint gridHash = calcGridHash(neighbourPos, gridRes);
				uint startIndex = cellStart[gridHash];

				if (startIndex != 0xffffffff)
				{
					uint endIndex = cellEnd[gridHash];

					REAL u, v, w;
					for (uint j = startIndex; j < endIndex; j++)
					{
						uint sortedIdx = gridIdx[j];

						if (sortedIdx == iv0 || sortedIdx == iv1 || sortedIdx == iv2) continue;
						
						REAL3 p = make_REAL3(pos[sortedIdx]);

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

						if (distance > thickness) // 설정 거리보다 멀리 있을 경우 투영 X
							continue;

						REAL3 n = (p - pc);
						Normalize(n);

						//barycentric coord 이용 보간
						REAL c = Dot(n, p - p0) - thickness;

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

						REAL damping = 1.0;
						//sortedPos[j] += dP * damping;
						//sortedPos[iv0] += dP0 * damping;
						//sortedPos[iv1] += dP1 * damping;
						//sortedPos[iv2] += dP2 * damping;
						//pos[sortedIdx] += dP * damping;
						//pos[iv0] += dP0 * damping;
						//pos[iv1] += dP1 * damping;
						//pos[iv2] += dP2 * damping;

						atomicAdd_REAL(impulse + sortedIdx * 3u + 0u, dP.x * damping);
						atomicAdd_REAL(impulse + sortedIdx * 3u + 1u, dP.y * damping);
						atomicAdd_REAL(impulse + sortedIdx * 3u + 2u, dP.z * damping);

						atomicAdd_REAL(impulse + iv0 * 3u + 0u, dP0.x * damping);
						atomicAdd_REAL(impulse + iv0 * 3u + 1u, dP0.y * damping);
						atomicAdd_REAL(impulse + iv0 * 3u + 2u, dP0.z * damping);

						atomicAdd_REAL(impulse + iv1 * 3u + 0u, dP1.x * damping);
						atomicAdd_REAL(impulse + iv1 * 3u + 1u, dP1.y * damping);
						atomicAdd_REAL(impulse + iv1 * 3u + 2u, dP1.z * damping);

						atomicAdd_REAL(impulse + iv2 * 3u + 0u, dP2.x * damping);
						atomicAdd_REAL(impulse + iv2 * 3u + 1u, dP2.y * damping);
						atomicAdd_REAL(impulse + iv2 * 3u + 2u, dP2.z * damping);

						//REAL3 cor = n * (thickness - distance);

						//atomicAdd_REAL(impulse + sortedIdx * 3u + 0u, cor.x * 0.5);
						//atomicAdd_REAL(impulse + sortedIdx * 3u + 1u, cor.y * 0.5);
						//atomicAdd_REAL(impulse + sortedIdx * 3u + 2u, cor.z * 0.5);

						//atomicAdd_REAL(impulse + iv0 * 3u + 0u, cor.x * -0.5);
						//atomicAdd_REAL(impulse + iv0 * 3u + 1u, cor.y * -0.5);
						//atomicAdd_REAL(impulse + iv0 * 3u + 2u, cor.z * -0.5);

						//atomicAdd_REAL(impulse + iv1 * 3u + 0u, cor.x * -0.5);
						//atomicAdd_REAL(impulse + iv1 * 3u + 1u, cor.y * -0.5);
						//atomicAdd_REAL(impulse + iv1 * 3u + 2u, cor.z * -0.5);

						//atomicAdd_REAL(impulse + iv2 * 3u + 0u, cor.x * -0.5);
						//atomicAdd_REAL(impulse + iv2 * 3u + 1u, cor.y * -0.5);
						//atomicAdd_REAL(impulse + iv2 * 3u + 2u, cor.z * -0.5);
					}
				}
			}
		}
	}
}

__global__ void ApplyImpulse_kernel(REAL3* pos, REAL* impulse, uint numVer, REAL damping)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	REAL3 im;
	im.x = impulse[idx * 3 + 0u] * damping;
	im.y = impulse[idx * 3 + 1u] * damping;
	im.z = impulse[idx * 3 + 2u] * damping;

	pos[idx] += im;
}