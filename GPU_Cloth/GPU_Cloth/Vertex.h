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
	Dvector<REAL3> d_restPos;
	Dvector<REAL3> d_Pos;
	Dvector<REAL3> d_Pos1;
	Dvector<REAL3> d_Vel;
	Dvector<REAL3> d_vNormal;
	Dvector<REAL> d_InvMass;
	DPrefixArray<uint> d_nbFaces;
	DPrefixArray<uint> d_nbVertices;
public:		//Host
	vector<REAL3> h_pos;
	vector<REAL3> h_pos1;
	vector<REAL3> h_vel;
	vector<REAL3> h_vNormal;
	vector<REAL> h_invMass;
	PrefixArray<uint> h_nbVFaces;
	PrefixArray<uint> h_nbVertices;
public:
	Vertex();
	Vertex(const Mesh& mesh)
	{
		copyHostValue(mesh);
	}
	~Vertex();
public:
	void InitDeviceMem(uint numVertex);
	void copyToDevice(void);
	void copyToHost(void);
	void copyHostValue(const Mesh& mesh);
	void FreeDeviceMem(void);
};

#endif