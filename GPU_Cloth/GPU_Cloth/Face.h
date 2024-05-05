#ifndef __FACE_H__
#define __FACE_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "Mesh.h"
#include <vector>

using namespace std;

class Face
{
public:		//Device
	Dvector<uint3> d_faceIdx;
	Dvector<AABB> d_faceAABB;
	Dvector<REAL3> d_fNormal;
public:		//Host
	vector<uint3> h_faceIdx;
	vector<REAL3> h_fNormal;
public:
	Face();
	Face(const Mesh& mesh)
	{
		copyHostValue(mesh);
	}
	~Face();
public:
	void InitDeviceMem(uint numFace);
	void copyToDevice(void);
	void copyHostValue(const Mesh& mesh);
	void FreeDeviceMem(void);
};

#endif