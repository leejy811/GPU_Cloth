#ifndef __PBD_CUDA_H__
#define __PBD_CUDA_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vec3.h"
#include "Vertex.h"
#include "Face.h"
#include <vector>

using namespace std;

class PBD_Cuda
{
public:
	double3* dPos;
	double3* dPos1;
	double* dInvMass;
public:
	PBD_Cuda();
	~PBD_Cuda();
public:

};

#endif
