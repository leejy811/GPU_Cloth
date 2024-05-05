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

	cudaMemcpyToSymbol(&hashParam, &_param, sizeof(HashParam));
}

void Device_Hash::SetHashTable_kernel(Dvector<REAL3>& pos)
{
	CalculateHash_Kernel(pos);
	SortParticle_Kernel();
	FindCellStart_Kernel();
}

void Device_Hash::CalculateHash_Kernel(Dvector<REAL3>& pos)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_param._numParticle, 256, numBlocks, numThreads);

	CalculateHash_D << <numBlocks, numThreads >> >
		(d_gridHash(), d_gridIdx(), pos());
}

void Device_Hash::SortParticle_Kernel()
{
	thrust::sort_by_key(thrust::device_ptr<uint>(d_gridHash()),
		thrust::device_ptr<uint>(d_gridHash() + _param._numParticle),
		thrust::device_ptr<uint>(d_gridIdx()));
}

void Device_Hash::FindCellStart_Kernel()
{
	uint numThreads, numBlocks;
	ComputeGridSize(_param._numParticle, 256, numBlocks, numThreads);

	uint smemSize = sizeof(uint) * (numThreads + 1);
	FindCellStart_D << <numBlocks, numThreads, smemSize >> >
		(d_gridHash(), d_gridIdx(), d_cellStart(), d_cellEnd());
}

void Device_Hash::InitDeviceMem(uint numParticle)
{
	d_gridHash.resize(numParticle);			d_gridHash.memset(0);
	d_gridIdx.resize(numParticle);			d_gridIdx.memset(0);
	d_cellStart.resize(pow(_param._gridRes, 3));			d_cellStart.memset(0);
	d_cellEnd.resize(pow(_param._gridRes, 3));			d_cellEnd.memset(0);

	_param._numParticle = numParticle;

	cudaMemcpyToSymbol(&hashParam, &_param, sizeof(HashParam));
}

void Device_Hash::FreeDeviceMem(void)
{
	d_gridHash.free();
	d_gridIdx.free();
	d_cellStart.free();
	d_cellEnd.free();
}