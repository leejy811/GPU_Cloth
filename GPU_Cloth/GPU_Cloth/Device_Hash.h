#ifndef __DEVICE_HASH_H__
#define __DEVICE_HASH_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "Parameter.h"
#include "thrust/sort.h"
#include <vector>

using namespace std;

class Device_Hash
{
public: //Device
	uint* d_gridHash;
	uint* d_gridIdx;
	uint* d_cellStart;
	uint* d_cellEnd;
public:
	HashParam _param;
public:
	Device_Hash();
	~Device_Hash();
public:
	void InitParam(ClothParam clothParam);
	void SetHashTable_kernel(REAL3* pos);
	void CalculateHash_Kernel(REAL3* pos);
	void SortParticle_Kernel();
	void FindCellStart_Kernel();
	void ComputeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = divup(n, numThreads);
	}
public:
	void InitDeviceMem(uint numVertex);
	void FreeDeviceMem(void);
};

#endif