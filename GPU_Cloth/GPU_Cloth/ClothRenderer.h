#ifndef __CLOTH_RENDERER_H__
#define __CLOTH_RENDERER_H__

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDA_Common/helper_cuda.h"
#include "CUDA_Common/helper_functions.h"
#include "CUDA_Common/helper_gl.h"
#include "CUDA_Custom/DeviceManager.h"
#include <cuda_gl_interop.h>
#include "Camera.h"
#include "Shader.h"

using namespace std;

class ClothRenderer
{
public:		//Buffer Object
	uint posVbo;
	uint normVbo;
	uint faceIbo;
	uint depthMapFbo;
	uint depthMapTexture;
	cudaGraphicsResource* cuda_posVbo;
	cudaGraphicsResource* cuda_normVbo;
	cudaGraphicsResource* cuda_faceIbo;
public:		//Shader
	Shader* _shader;
	Shader* _depthShader;
private:
	uint shadowWidth = 1024;
	uint shadowHeight = 1024;
public:
	ClothRenderer();
	ClothRenderer(uint numVertex, uint numFace);
	~ClothRenderer();
public:
	void InitBO(uint numVertex, uint numFace);
	void DrawBO(uint numFace, const Camera& camera);
	void MappingBO(REAL3* pos, REAL3* norm, uint3* fIdx, uint numVertex, uint numFace);
	void UnMappingBO();
	void UpdateShader(Shader& shader, const Camera& camera);
	void RenderScene(uint numFace);
private:
	void createBO(uint* vbo, struct cudaGraphicsResource** cuda_resource, uint size, GLenum type);
	void createFBO(uint* fbo, uint* texture);
	void deleteBO(uint* vbo, struct cudaGraphicsResource** cuda_resource);
};

#endif