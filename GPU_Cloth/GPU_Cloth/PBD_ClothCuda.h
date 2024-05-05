#ifndef __PBD_CUDA_H__
#define __PBD_CUDA_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "thrust/sort.h"
#include "Constraint.h"
#include "Vertex.h"
#include "Face.h"
#include "Device_Hash.h"
#include "Mesh.h"
#include "Parameter.h"
#include <GL/freeglut.h>
#include <vector>
#include <fstream>
#include <string>

#define BLOCK_SIZE 1024

using namespace std;

class PBD_ClothCuda
{
public:
	//Parameter
	ClothParam _param;
	AABB _boundary;
public:	//Constraint
	Constraint* _strechSpring;
	Constraint* _bendSpring;
public:	//Mesh
	Vertex* d_Vertex;
	Face* d_Face;
	Device_Hash* d_Hash;
	Mesh* h_Mesh;
public:
	PBD_ClothCuda();
	PBD_ClothCuda(char* filename, REAL gravity, REAL dt)
	{
		InitParam(gravity, dt);
		Init(filename);
	}
	~PBD_ClothCuda();
public:		//init
	void Init(char* filename);
	void InitParam(REAL gravity, REAL dt);
	void buildConstraint(void);
public:		//Update
	void ComputeGravity_kernel(void);
	void ComputeWind_kernel(REAL3 wind);
	void Intergrate_kernel(void);
	void ComputeFaceNormal_kernel(void);
	void ComputeVertexNormal_kernel(void);
	void ProjectConstraint_kernel(void);
	void SetHashTable_kernel(void);
	void UpdateFaceAABB_Kernel(void);
	void Colide_kernel();
	void LevelSetCollision_kernel(void);
public:
	void draw(void);
	void drawWire(void);
public:		//Cuda
	void InitDeviceMem(void);
	void copyToDevice(void);
	void copyToHost(void);
	void FreeDeviceMem(void);
	void ComputeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = divup(n, numThreads);
	}
};

#endif
