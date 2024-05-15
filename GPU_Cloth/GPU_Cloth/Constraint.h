#ifndef __CONSTRAINT_H__
#define __CONSTRAINT_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "Vertex.h"
#include "Parameter.h"
#include <vector>

#define CONST_BLOCK_SIZE 32

using namespace std;

class Constraint
{
public:		//Device
	Dvector<uint2> d_EdgeIdx;
	Dvector<REAL> d_RestLength;
	DPrefixArray<uint> d_ColorIdx;
public:		//Host
	vector<uint2> h_EdgeIdx;
	vector<uint2> h_GraphIdx;
	vector<REAL> h_RestLength;
	PrefixArray<uint> h_ColorIdx;
	PrefixArray<uint> h_nbCEdges;
	PrefixArray<uint> h_nbGVertices;
public:	//const
	ConstParam _param;
public:
	Constraint();
	Constraint(uint iter, REAL stiff)
	{
		_param._iteration = iter;
		_param._springK = stiff;
		_param._numConstraint = 0;
		_param._numColor = 0;
	}
	~Constraint();
public:		//Init
	void Init(int numVertices);
	void InitGraphEdge(int numVertices);
	void InitGraphAdjacency();
	void InitConstraintColor(void);
public:		//Update
	void IterateConstraint(REAL3* pos, REAL* invm, REAL* satm);
	void SolveDistanceConstraint_kernel(uint numConst, uint idx, REAL3* pos, REAL* invm, REAL* satm);
public:		//Cuda
	void InitDeviceMem(void);
	void	copyToDevice(void);
	void	copyToHost(void);
	void FreeDeviceMem(void);
};

#endif
