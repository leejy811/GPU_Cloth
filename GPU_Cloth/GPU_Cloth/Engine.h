#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#pragma once
#include "PBD_ClothCuda.h"
#include <vector>

class Engine
{
public:
	vector<PBD_ClothCuda*> _cloths;
public:
	AABB			_boundary;
public:
	REAL3			_gravity;
	REAL			_dt;
	REAL			_invdt;
	uint			_frame;
public:
	Engine() {}
	Engine(REAL3& gravity, REAL dt, char* filename, uint num)
	{
		init(gravity, dt, filename, num);
	}
	~Engine() {}
public:
	void	init(REAL3& gravity, REAL dt, char* filename, uint num);
public:
	void	simulation(void);
	void	reset(void);
	void	ApplyWind(REAL3 wind);
public:
	void	draw(void);
	void	drawWire(void);
};

#endif