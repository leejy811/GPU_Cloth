#include "Face.h"

Face::Face()
{

}

Face::~Face()
{

}

void Face::InitDeviceMem(uint numFace)
{
	cudaMalloc(&d_faceIdx, sizeof(uint3) * numFace);	cudaMemset(d_faceIdx, 0, sizeof(uint3) * numFace);
	cudaMalloc(&d_faceAABB, sizeof(AABB) * numFace);	cudaMemset(d_faceAABB, 0, sizeof(AABB) * numFace);
	cudaMalloc(&d_fNormal, sizeof(REAL3) * numFace);	cudaMemset(d_fNormal, 0, sizeof(REAL3) * numFace);
}

void Face::copyToDevice(const Mesh& mesh)
{
	cudaMemcpy(d_faceIdx, &mesh.h_faceIdx[0], sizeof(uint3) * mesh.h_faceIdx.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fNormal, &mesh.h_fNormal[0], sizeof(REAL3) * mesh.h_fNormal.size(), cudaMemcpyHostToDevice);
}

void Face::FreeDeviceMem(void)
{
	cudaFree(d_faceIdx);
	cudaFree(d_faceAABB);
	cudaFree(d_fNormal);
}