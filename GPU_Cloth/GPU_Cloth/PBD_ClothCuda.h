#ifndef __PBD_CUDA_H__
#define __PBD_CUDA_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"1
#include "thrust/sort.h"
#include "Constraint.h"
#include "Vertex.h"
#include "Face.h"
#include "Edge.h"
#include "Device_Hash.h"
#include "Mesh.h"
#include "ClothRenderer.h"
#include "Parameter.h"
#include "Camera.h"
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

	int frame = 0;
public:	//Constraint
	Constraint* _strechSpring;
	Constraint* _bendSpring;
public:	//Mesh
	Vertex d_Vertex;
	Face d_Face;
	Edge d_Edge;
	Device_Hash d_Hash;
	Mesh* h_Mesh;
	ClothRenderer* _clothRenderer;
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
	void ComputeRestAngle(void);
	void ComputeLaplacian(void);
public:		//Update
	void Simulation();
	void ComputeGravity_kernel(void);
	void ComputeWind_kernel(REAL3 wind);
	void Intergrate_kernel(void);
	void ComputeFaceNormal_kernel(void);
	void ComputeVertexNormal_kernel(void);
	void ProjectConstraint_kernel(void);
	void AngleConstraint_kernel(void);
	void SetHashTable_kernel(void);
	void UpdateFaceAABB_Kernel(void);
	void Colide_kernel(void);
	void SelfCollision_kernel(void);
	void AdhesionForce_kernel(void);
	void WetCloth_Kernel(void);
	void Absorption_Kernel(void);
	void Diffusion_Kernel(void);
	void Dripping_Kernel(void);
	void UpdateMass_Kernel(void);
	void LevelSetCollision_kernel(void);
	void ComputeWrinkCloth_kernel(void);
public:
	void draw(void);
	void drawBO(const Camera& camera);
	void drawWire(void);
	void drawFlat(void);
	REAL3 ScalarToColor(REAL val);
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
