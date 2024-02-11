#include "PBD_ClothCuda.h"

__global__ void ComputeGravity_kernel(REAL3* pos, REAL3* pos1, REAL3* vel, REAL3 gravity, REAL damp, uint numVer, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;
	REAL3 v = vel[idx];
	v += gravity * dt;
	v *= damp;
	vel[idx] = v;

	pos1[idx] = pos[idx] + (vel[idx] * dt);
}

__global__ void ComputeIntergrate_kernel(REAL3* pos, REAL3* pos1, REAL3* vel, uint numVer, REAL invdt)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	vel[idx] = (pos1[idx] - pos[idx]) * invdt;
	pos[idx] = pos1[idx];
}

__global__ void ComputeNormal_kernel(REAL3* fIdx, REAL3* pos, REAL3* fNorm, REAL3* vNorm, uint numFace)
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