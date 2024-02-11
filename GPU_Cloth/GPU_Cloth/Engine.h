#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#pragma once
#include "PBD_ClothCuda.h"

class Engine
{
public:
	PBD_ClothCuda* _cloths;
public:
	AABB			_boundary;
public:
	REAL3			_gravity;
	REAL			_dt;
	REAL			_invdt;
	uint			_frame;
public:
	Engine() {}
	Engine(REAL3& gravity, REAL dt, char* filename)
	{
		init(gravity, dt, filename);
	}
	~Engine() {}
public:
	void	init(REAL3& gravity, REAL dt, char* filename);
public:
	void	simulation(void);
	void	reset(void);
public:
	void	draw(void);
};

#endif