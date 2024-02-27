#include "PBD_ClothCuda.h"

__global__ void CompExternlaForce_kernel(REAL3* pos, REAL3* pos1, REAL3* vel, REAL* invm, REAL3 gravity, REAL3 ext, REAL damp, uint numVer, REAL dt)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;
	REAL3 v = vel[idx];
	v += gravity * dt;
	v += ext * invm[idx] * dt;
	v *= damp;
	vel[idx] = v;

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

__global__ void CompIntergrate_kernel(REAL3* pos, REAL3* pos1, REAL3* vel, uint numVer, REAL invdt)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numVer)
		return;

	if (pos1[idx].y > 0.7)
		return;

	vel[idx] = (pos1[idx] - pos[idx]) * invdt;
	pos[idx] = pos1[idx];
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