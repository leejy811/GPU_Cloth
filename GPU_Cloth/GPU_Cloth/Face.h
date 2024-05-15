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
	uint3* d_faceIdx;
	AABB* d_faceAABB;
	REAL3* d_fNormal;
	REAL* d_fSaturation;
	REAL* d_fDripbuf;
	REAL* d_fDripThres;
	DPrefixArray<uint> d_nbFace;
public:
	Face();
	~Face();
public:
	void InitDeviceMem(uint numFace);
	void copyToDevice(const Mesh& mesh);
	void copyToHost(Mesh& mesh);
	void FreeDeviceMem(void);
};

#endif