#ifndef __CONSTRAINT_H__
#define __CONSTRAINT_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include <vector>

#define CONST_BLOCK_SIZE 128

using namespace std;

class Constraint
{
public:		//Device
	Dvector<uint2> d_EdgeIdx;
	Dvector<REAL> d_RestLength;
public:		//Host
	vector<uint2> h_EdgeIdx;
	vector<REAL> h_RestLength;
public:	//const
	uint _numConstraint;
	uint _numColor;
public:
	Constraint();
	~Constraint();
public:		//init
	void Init();
public:		//Update
public:		//Cuda
	void InitDeviceMem(void);
	void	copyToDevice(void);
	void	copyToHost(void);
	void FreeDeviceMem(void);
};

#endif
