#ifndef __PBD_CUDA_H__
#define __PBD_CUDA_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"
#include "Constraint.h"
#include <GL/freeglut.h>
#include <vector>
#include <fstream>
#include <string>

#define BLOCK_SIZE 1024

using namespace std;

class PBD_ClothCuda
{
public:		//Device
	Dvector<uint3> d_faceIdx;
	Dvector<REAL3> d_Pos;
	Dvector<REAL3> d_Pos1;
	Dvector<REAL3> d_Vel;
	Dvector<REAL3> d_fNormal;
	Dvector<REAL3> d_vNormal;
	Dvector<REAL> d_InvMass;
	DPrefixArray<uint> d_nbFaces;
	DPrefixArray<uint> d_nbVertices;
public:		//Host
	vector<uint3> h_faceIdx;
	vector<REAL3> h_pos;
	vector<REAL3> h_pos1;
	vector<REAL3> h_vel;
	vector<REAL3> h_fNormal;
	vector<REAL3> h_vNormal;
	vector<REAL> h_invMass;
	PrefixArray<uint> h_nbFaces;
	PrefixArray<uint> h_nbVertices;
public:	
	uint _numVertices;
	uint _numFaces;
	uint _iteration;
	REAL _linearDamping;
	REAL _springK;
	REAL3 _externalForce;
	AABB _boundary;
public:	//Constraint
	Constraint* _strechSpring;
	Constraint* _bendSpring;
public:
	PBD_ClothCuda();
	PBD_ClothCuda(char* filename, uint iter, REAL damp, REAL stiff)
	{
		LoadObj(filename);
		Init(iter, damp, stiff);
	}
	~PBD_ClothCuda();
public:		//init
	void Init(uint iter, REAL damp, REAL stiff);
	void	LoadObj(char* filename);
	void moveCenter(REAL scale);
	void buildAdjacency(void);
	void computeNormal(void);
	void SetMass(void);
public:		//Update
	void ComputeExternalForce_kernel(REAL3& extForce, REAL dt);
	void ComputeWind_kernel(REAL3 wind);
	void Intergrate_kernel(REAL dt);
	void ComputeFaceNormal_kernel(void);
	void ComputeVertexNormal_kernel(void);
	void ProjectConstraint_kernel(void);
public:
	void draw(void);
public:		//Cuda
	void InitDeviceMem(void);
	void	copyToDevice(void);
	void	copyToHost(void);
	void	copyNbToDevice(void);
	void	copyNbToHost(void);
	void FreeDeviceMem(void);
};

#endif
