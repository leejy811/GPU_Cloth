#ifndef __SHADER_H__
#define __SHADER_H__

#pragma once

#include <iostream>
#include <string>
#include <GL/glew.h>
#include <glm/glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL 
#include <glm/glm/gtx/transform.hpp>
#include "Camera.h"

using namespace std;

class Shader
{
	enum
	{
		ModelMatrix,
		ViewMatrix,
		ProjectionMatrix,
		CameraPos,
		LightPosition,
		LightMatrix,
		NUM_UNIFORMS
	};
public:
	static const unsigned int NUM_SHADER = 2;
	GLuint _program;
	GLuint _shaders[NUM_SHADER]; // vertex and fragment shaders
	GLuint _uniforms[NUM_UNIFORMS];
public:
	Shader(const string& fileName);
	virtual ~Shader();
public:
	void		bind(void);
	void		update(const Camera& camera);
	glm::mat3	getNormalMat(const glm::mat4& modelViewMatrix);
};

#endif