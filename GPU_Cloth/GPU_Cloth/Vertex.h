#ifndef __VERTEX_H__
#define __VERTEX_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "Mesh.h"
#include <vector>

using namespace std;

class Vertex
{
public:		//Device
	REAL3* d_restPos;
	REAL3* d_Pos;
	REAL3* d_Pos1;
	REAL3* d_Vel;
	REAL3* d_vNormal;
	REAL* d_InvMass;
	DPrefixArray<uint> d_nbFaces;
	DPrefixArray<uint> d_nbVertices;
public:
	Vertex();
	~Vertex();
public:
	void InitDeviceMem(uint numVertex);
	void copyToDevice(const Mesh& mesh);
	void copyToHost(Mesh& mesh);
	void FreeDeviceMem(void);
};

#endif