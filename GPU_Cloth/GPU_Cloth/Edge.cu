#include "Edge.h"

Edge::Edge()
{

}

Edge::~Edge()
{

}

void Edge::InitDeviceMem(uint numEdge)
{
	cudaMalloc(&d_edgeIdx, sizeof(uint2) * numEdge);	cudaMemset(d_edgeIdx, 0, sizeof(uint2) * numEdge);
	cudaMalloc(&d_restAngle, sizeof(REAL) * numEdge);	cudaMemset(d_restAngle, 0, sizeof(REAL) * numEdge);
	cudaMalloc(&d_cotWeight, sizeof(REAL) * numEdge);	cudaMemset(d_cotWeight, 0, sizeof(REAL) * numEdge);
}

void Edge::copyToDevice(const Mesh& mesh)
{
	cudaMemcpy(d_edgeIdx, &mesh.h_edgeIdx[0], sizeof(uint2) * mesh.h_edgeIdx.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_restAngle, &mesh.h_restAngle[0], sizeof(REAL) * mesh.h_restAngle.size(), cudaMemcpyHostToDevice);
	d_nbEFaces.copyFromHost(mesh.h_nbEFaces);
}

void Edge::copyToHost(Mesh& mesh)
{
	cudaMemcpy(&mesh.h_cotWeight[0], d_cotWeight, sizeof(REAL) * mesh.h_cotWeight.size(), cudaMemcpyDeviceToHost);
}

void Edge::FreeDeviceMem(void)
{
	cudaFree(d_edgeIdx);
	cudaFree(d_restAngle);
	d_nbEFaces.clear();
}