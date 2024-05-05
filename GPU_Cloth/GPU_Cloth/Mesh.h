#ifndef __MESH_H__
#define __MESH_H__

#pragma once
#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/PrefixArray.h"
#include <vector>
#include <fstream>
#include <string>

using namespace std;

class Mesh
{
public:	//Host
	//Face
	vector<uint3> h_faceIdx;
	vector<REAL3> h_fNormal;
	//Edge
	vector<uint2> h_edgeIdx;
	PrefixArray<uint> h_nbEFaces;
	PrefixArray<uint> h_nbEVertices;
	//Vertex
	vector<REAL3> h_pos;
	vector<REAL3> h_pos1;
	vector<REAL3> h_vel;
	vector<REAL3> h_vNormal;
	vector<REAL> h_invMass;
	PrefixArray<uint> h_nbVFaces;
	PrefixArray<uint> h_nbVertices;
public:
	Mesh();
	Mesh(char* filename, AABB boundary)
	{
		LoadObj(filename, boundary);
	}
	~Mesh();
private:		//init
	void LoadObj(char* filename, AABB boundary);
	void moveCenter(REAL scale, AABB boundary);
	void buildAdjacency(void);
	void buildAdjacency_VF(void);
	void buildAdjacency_EF(void);
	void buildEdges(void);
	void computeNormal(void);
	void SetMass(void);
};

#endif