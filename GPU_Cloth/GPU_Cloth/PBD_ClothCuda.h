#ifndef __PBD_CUDA_H__
#define __PBD_CUDA_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
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
public:		//Host
	vector<uint3> h_faceIdx;
	vector<REAL3> h_pos;
	vector<REAL3> h_pos1;
	vector<REAL3> h_vel;
	vector<REAL3> h_fNormal;
	vector<REAL3> h_vNormal;
	vector<REAL> h_invMass;
public:	//const
	uint _numVertices;
	uint _numFaces;
	uint _iteration;
	REAL _linearDamping;
	REAL _springK;
	AABB _boundary;
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
	void computeNormal(void);
public:		//Update
	void ComputeGravityForce_kernel(REAL3& gravity, REAL dt);
	void Intergrate_kernel(REAL dt);
	void computeNormal_kernel(void);
public:
	void draw(void);
public:		//Cuda
	void InitDeviceMem(void);
	void	copyToDevice(void);
	void	copyToHost(void);
	void FreeDeviceMem(void);
};

#endif
