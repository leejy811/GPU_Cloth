#include "ClothRenderer.h"

__global__ void CopyVecToVBO(REAL3* vbo, REAL3* device, uint num)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= num)
		return;

	vbo[idx] = device[idx];
}

__global__ void CopyIdxToIBO(uint3* ibo, uint3* device, uint num)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= num)
		return;

	ibo[idx] = device[idx];
}