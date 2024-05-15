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
	cudaMalloc(&d_fSaturation, sizeof(REAL) * numFace);	cudaMemset(d_fSaturation, 0, sizeof(REAL) * numFace);
	cudaMalloc(&d_fDripbuf, sizeof(REAL) * numFace);	cudaMemset(d_fDripbuf, 0, sizeof(REAL) * numFace);
	cudaMalloc(&d_fDripThres, sizeof(REAL) * numFace);	cudaMemset(d_fDripThres, 0, sizeof(REAL) * numFace);
}

void Face::copyToDevice(const Mesh& mesh)
{
	cudaMemcpy(d_faceIdx, &mesh.h_faceIdx[0], sizeof(uint3) * mesh.h_faceIdx.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_fNormal, &mesh.h_fNormal[0], sizeof(REAL3) * mesh.h_fNormal.size(), cudaMemcpyHostToDevice);

	d_nbFace.copyFromHost(mesh.h_nbFFaces);
}

void Face::copyToHost(Mesh& mesh)
{
	cudaMemcpy(&mesh.h_fSaturation[0], d_fSaturation, sizeof(REAL) * mesh.h_fSaturation.size(), cudaMemcpyDeviceToHost);
}

void Face::FreeDeviceMem(void)
{
	cudaFree(d_faceIdx);
	cudaFree(d_faceAABB);
	cudaFree(d_fNormal);
	cudaFree(d_fSaturation);
	cudaFree(d_fDripbuf);
	cudaFree(d_fDripThres);
}