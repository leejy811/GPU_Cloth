#ifndef __EDGE_H__
#define __EDGE_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "Mesh.h"
#include <vector>

using namespace std;

class Edge
{
public:		//Device
	uint2* d_edgeIdx;
	REAL* d_restAngle;
	REAL* d_cotWeight;
	DPrefixArray<uint> d_nbEFaces;
public:
	Edge();
	~Edge();
public:
	void InitDeviceMem(uint numEdge);
	void copyToDevice(const Mesh& mesh);
	void copyToHost(Mesh& mesh);
	void FreeDeviceMem(void);
};

#endif