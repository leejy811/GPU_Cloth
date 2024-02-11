#ifndef __DEVICE_VERTEX_H__
#define __DEVICE_VERTEX_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vec3.h"
#include "Vertex.h"
#include "Face.h"
#include <vector>

using namespace std;

class DeviceVertex
{
public:		//Device
	float3* _dPos;
	float3* _dPos1;
	float3* _dVel;
	float3* _dNormal;
	float* _dInvMass;
public:		//Host
	vector<float3> _pos;
	vector<float3> _pos1;
	vector<float3> _vel;
	vector<float3> _normal;
	vector<float> _invMass;
public:
	int _numVertices;
public:
	DeviceVertex();
	~DeviceVertex();
public:
	void InitMem();
	void FreeMem();
	void CopyToDevice();
	void CopyToHost();
};

#endif