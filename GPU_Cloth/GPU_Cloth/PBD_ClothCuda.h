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
#include <GL/freeglut.h>
#include <vector>
#include <fstream>
#include <string>

#define BLOCK_SIZE 1024

using namespace std;

class PBD_ClothCuda
{
public:		//Device
	Dvector<BOOL> d_flag;
	Dvector<uint> d_gridHash;
	Dvector<uint> d_gridIdx;
	Dvector<uint> d_cellStart;
	Dvector<uint> d_cellEnd;
	Dvector<uint3> d_faceIdx;
	Dvector<AABB> d_faceAABB;
	Dvector<REAL3> d_sortedPos;
	Dvector<REAL3> d_restPos;
	Dvector<REAL3> d_Pos;
	Dvector<REAL3> d_Pos1;
	Dvector<REAL3> d_sortedVel;
	Dvector<REAL3> d_Vel;
	Dvector<REAL3> d_fNormal;
	Dvector<REAL3> d_vNormal;
	Dvector<REAL> d_InvMass;
	DPrefixArray<uint> d_nbFaces;
	DPrefixArray<uint> d_nbVertices;
public:		//Host
	vector<BOOL> h_flag;
	vector<uint3> h_faceIdx;
	vector<uint2> h_edgeIdx;
	vector<REAL3> h_pos;
	vector<REAL3> h_pos1;
	vector<REAL3> h_vel;
	vector<REAL3> h_fNormal;
	vector<REAL3> h_vNormal;
	vector<REAL> h_invMass;
	PrefixArray<uint> h_nbEFaces;
	PrefixArray<uint> h_nbVFaces;
	PrefixArray<uint> h_nbVertices;
	PrefixArray<uint> h_nbEVertices;
public:	
	uint _numVertices;
	uint _numEdges;
	uint _numFaces;
	uint _iteration;
	uint _gridRes;
	REAL _gridSize;
	REAL _thickness;
	REAL _linearDamping;
	REAL _selfColliDamping;
	REAL _springK;
	REAL3 _externalForce;
	AABB _boundary;
public:	//Constraint
	Constraint* _strechSpring;
	Constraint* _bendSpring;
public:
	PBD_ClothCuda();
	PBD_ClothCuda(char* filename, uint iter, REAL liDamp, REAL stiff, uint grid, REAL thick, REAL selfDamp)
		: _iteration(iter), _linearDamping(liDamp), _springK(stiff), _gridRes(grid), _thickness(thick), _selfColliDamping(selfDamp)
	{
		LoadObj(filename);
		Init();
	}
	~PBD_ClothCuda();
public:		//init
	void Init();
	void LoadObj(char* filename);
	void moveCenter(REAL scale);
	void buildAdjacency(void);
	void buildAdjacency_VF(void);
	void buildAdjacency_EF(void);
	void buildConstraint(void);
	void buildGraph(void);
	void buildEdges(void);
	void computeNormal(void);
	void SetMass(void);
public:		//Update
	void ComputeExternalForce_kernel(REAL3& extForce, REAL dt);
	void ComputeWind_kernel(REAL3 wind);
	void Intergrate_kernel(REAL dt);
	void ComputeFaceNormal_kernel(void);
	void ComputeVertexNormal_kernel(void);
	void ProjectConstraint_kernel(void);
	void SetHashTable_kernel(void);
	void UpdateFaceAABB_Kernel(void);
	void Colide_kernel();
public:		//Hash
	void CalculateHash_Kernel(void);
	void SortParticle_Kernel(void);
	void FindCellStart_Kernel(void);

	void LevelSetCollision_kernel(void);
public:
	void draw(void);
	void drawWire(void);
public:		//Cuda
	void InitDeviceMem(void);
	void copyToDevice(void);
	void copyToHost(void);
	void copyNbToDevice(void);
	void copyNbToHost(void);
	void FreeDeviceMem(void);
	void ComputeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = divup(n, numThreads);
	}
};

#endif
