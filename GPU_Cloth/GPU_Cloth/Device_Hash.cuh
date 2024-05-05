#include "Device_Hash.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__constant__ HashParam hashParam;

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

__global__ void CalculateHash_D(uint* gridHash, uint* gridIdx, REAL3* pos)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= hashParam._numParticle)
		return;

	REAL cellSize = 1.0 / hashParam._gridRes;
	int3 gridPos = calcGridPos(pos[idx], cellSize);
	uint hash = calcGridHash(gridPos, hashParam._gridRes);

	gridHash[idx] = hash;
	gridIdx[idx] = idx;
}

__global__ void FindCellStart_D(uint* gridHash, uint* gridIdx, uint* cellStart, uint* cellEnd)
{
	cg::thread_block cta = cg::this_thread_block();
	extern __shared__ uint sharedHash[];
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	uint hash;

	if (idx < hashParam._numParticle)
	{
		hash = gridHash[idx];
		sharedHash[threadIdx.x + 1] = hash;

		if (idx > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = gridHash[idx - 1];
		}
	}

	cg::sync(cta);

	if (idx < hashParam._numParticle)
	{

		if (idx == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = idx;

			if (idx > 0) cellEnd[sharedHash[threadIdx.x]] = idx;
		}

		if (idx == hashParam._numParticle - 1)
		{
			cellEnd[hash] = idx + 1;
		}
	}
}