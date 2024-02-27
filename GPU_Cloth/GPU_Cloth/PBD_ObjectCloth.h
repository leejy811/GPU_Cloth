#ifndef __PBD_OBJECT_CLOTH_H__
#define __PBD_OBJECT_CLOTH_H__

#pragma once
#include "Vec3.h"
#include "Vertex.h"
#include "Face.h"
#include <vector>

using namespace std;

class PBD_ObjectCloth
{
public:
	Vec3<double>	_minBoundary;
	Vec3<double>	_maxBoundary;
	vector<Vertex*> _vertices;
	vector<Face*> _faces;
public:
	double			_iteration = 20.0;
	double			_springK = 0.99;
public:
	PBD_ObjectCloth();
	PBD_ObjectCloth(char* filename)
	{
		loadObj(filename);
	}
	~PBD_ObjectCloth();
public:
	void reset(void);
	void	loadObj(char* filename);
	void moveToCenter(double scale);
	void	buildAdjacency(void);
	void	integrate(double dt);
	void	simulation(double dt);
	void	computeRestLength(void);
	void	computeNormal(void);
	void	updateBendSprings(void);
	void	updateStructuralSprings(void);
	void updateNormal(void);
	void	applyWind(vec3 wind);
	void	computeWindForTriangle(vec3 wind, Face* f);
	void	applyExtForces(double dt);
	void	solveDistanceConstraint(int index0, int index1, double restLength);
public:
	void	drawWire(void);
	void	drawSolid(void);
	void	drawPoint(void);
};

#endif
