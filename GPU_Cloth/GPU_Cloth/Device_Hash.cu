#include "Device_Hash.cuh"

Device_Hash::Device_Hash()
{

}

Device_Hash::~Device_Hash()
{

}

void Device_Hash::InitParam(ClothParam clothParam)
{
	_param._gridRes = clothParam._gridRes;
	_param._numParticle = clothParam._numVertices;

	cudaMemcpyToSymbol(hashParam, &_param, sizeof(HashParam));
}

void Device_Hash::SetHashTable_kernel(REAL3* pos)
{
	CalculateHash_Kernel(pos);
	SortParticle_Kernel();
	FindCellStart_Kernel();
}

void Device_Hash::CalculateHash_Kernel(REAL3* pos)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_param._numParticle, 256, numBlocks, numThreads);

	CalculateHash_D << <numBlocks, numThreads >> >
		(d_gridHash, d_gridIdx, pos);
}

void Device_Hash::SortParticle_Kernel()
{
	thrust::sort_by_key(thrust::device_ptr<uint>(d_gridHash),
		thrust::device_ptr<uint>(d_gridHash + _param._numParticle),
		thrust::device_ptr<uint>(d_gridIdx));
}

void Device_Hash::FindCellStart_Kernel()
{
	uint numThreads, numBlocks;
	ComputeGridSize(_param._numParticle, 256, numBlocks, numThreads);

	uint smemSize = sizeof(uint) * (numThreads + 1);
	FindCellStart_D << <numBlocks, numThreads, smemSize >> >
		(d_gridHash, d_gridIdx, d_cellStart, d_cellEnd);
}

void Device_Hash::InitDeviceMem(uint numParticle)
{
	uint numCell = _param._gridRes * _param._gridRes * _param._gridRes;

	cudaMalloc(&d_gridHash, sizeof(uint) * numParticle);	cudaMemset(d_gridHash, 0, sizeof(uint) * numParticle);
	cudaMalloc(&d_gridIdx, sizeof(uint) * numParticle);	cudaMemset(d_gridIdx, 0, sizeof(uint) * numParticle);
	cudaMalloc(&d_cellStart, sizeof(uint) * numCell);	cudaMemset(d_cellStart, 0, sizeof(uint) * numCell);
	cudaMalloc(&d_cellEnd, sizeof(uint) * numCell);	cudaMemset(d_cellEnd, 0, sizeof(uint) * numCell);
}

void Device_Hash::FreeDeviceMem(void)
{
	cudaFree(d_gridHash);
	cudaFree(d_gridIdx);
	cudaFree(d_cellStart);
	cudaFree(d_cellEnd);
}