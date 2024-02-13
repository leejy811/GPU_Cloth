#ifndef __CONSTRAINT_H__
#define __CONSTRAINT_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include <vector>

#define CONST_BLOCK_SIZE 128

using namespace std;

class Constraint
{
public:		//Device
	Dvector<uint2> d_EdgeIdx;
	Dvector<REAL> d_RestLength;
	DPrefixArray<uint> d_ColorIdx;
public:		//Host
	vector<uint2> h_EdgeIdx;
	vector<REAL> h_RestLength;
	PrefixArray<uint> h_ColorIdx;
public:	//const
	uint _numConstraint;
	uint _numColor;
	uint _iteration;
	REAL _springK;
public:
	Constraint();
	Constraint(uint iter, REAL stiff)
	{
		_iteration = iter;
		_springK = stiff;
		_numConstraint = 0;
		_numColor = 0;
	}
	~Constraint();
public:		//Init
	void InitConstraintColor(void);
public:		//Update
	void IterateConstraint(Dvector<REAL3>& pos1, Dvector<REAL>& invm);
	void SolveDistanceConstraint_kernel(uint numConst, uint idx, Dvector<REAL3>& pos1, Dvector<REAL>& invm);
public:		//Cuda
	void InitDeviceMem(void);
	void	copyToDevice(void);
	void	copyToHost(void);
	void FreeDeviceMem(void);
};

#endif
