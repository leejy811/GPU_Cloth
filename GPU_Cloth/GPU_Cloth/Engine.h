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
	REAL			_gravity;
	REAL			_dt;
	REAL			_invdt;
	uint			_frame;
public:
	Engine() {}
	Engine(REAL gravity, REAL dt, char* filename, uint num)
	{
		init(gravity, dt, filename, num);
	}
	~Engine() {}
public:
	void	init(REAL gravity, REAL dt, char* filename, uint num);
public:
	void	simulation(void);
	void	reset(void);
	void	ApplyWind(REAL3 wind);
public:
	void	draw(void);
	void	drawBO(const Camera& camera);
	void	drawWire(void);
private:
	uint shadowWidth = 1024;
	uint shadowHeight = 1024;
	uint depthMapFbo;
	uint depthMapTexture;
	void createFBO()
	{
		glGenFramebuffers(1, &depthMapFbo);

		glGenTextures(1, &depthMapTexture);
		glBindTexture(GL_TEXTURE_2D, depthMapTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadowWidth, shadowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		glBindFramebuffer(GL_FRAMEBUFFER, depthMapFbo);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMapTexture, 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
};

#endif