#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#pragma once

#include "CUDA_Custom/DeviceManager.h"
#include "CUDA_Custom/Dvector.h"
#include "CUDA_Custom/PrefixArray.h"

struct ClothParam
{
	uint _numVertices;
	uint _numEdges;
	uint _numFaces;

	uint _gridRes;
	REAL _thickness;
	REAL _selfColliDamping;

	uint _iteration;
	REAL _springK;
	REAL _linearDamping;

	REAL _gravity;
	REAL _subdt;
	REAL _subInvdt;

	REAL _maxsaturation;
	REAL _absorptionK;

	REAL _diffusK;
	REAL _gravityDiffusK;

	REAL _surfTension;
	REAL _adhesionThickness;
};

struct ConstParam
{
	uint _numConstraint;
	uint _numColor;
	uint _iteration;
	REAL _springK;
};

struct HashParam
{
	uint _gridRes;
	uint _numParticle;
};

#endif