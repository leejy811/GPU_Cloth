#include "Vertex.h"

Vertex::Vertex()
{

}

Vertex::~Vertex()
{

}

void Vertex::InitDeviceMem(uint numVertex)
{
	cudaMalloc(&d_restPos, sizeof(REAL3) * numVertex);	cudaMemset(d_restPos, 0, sizeof(REAL3) * numVertex);
	cudaMalloc(&d_Pos, sizeof(REAL3) * numVertex);		cudaMemset(d_Pos, 0, sizeof(REAL3) * numVertex);
	cudaMalloc(&d_Pos1, sizeof(REAL3) * numVertex);		cudaMemset(d_Pos1, 0, sizeof(REAL3) * numVertex);
	cudaMalloc(&d_Vel, sizeof(REAL3) * numVertex);		cudaMemset(d_Vel, 0, sizeof(REAL3) * numVertex);
	cudaMalloc(&d_vNormal, sizeof(REAL3) * numVertex);	cudaMemset(d_vNormal, 0, sizeof(REAL3) * numVertex);
	cudaMalloc(&d_InvMass, sizeof(REAL) * numVertex);		cudaMemset(d_InvMass, 0, sizeof(REAL) * numVertex);
	cudaMalloc(&d_SatMass, sizeof(REAL) * numVertex);		cudaMemset(d_SatMass, 0, sizeof(REAL) * numVertex);
	cudaMalloc(&d_vSaturation, sizeof(REAL) * numVertex);		cudaMemset(d_vSaturation, 0, sizeof(REAL) * numVertex);
	cudaMalloc(&d_Adhesion, sizeof(REAL3) * numVertex);		cudaMemset(d_Adhesion, 0, sizeof(REAL3) * numVertex);
	cudaMalloc(&d_vAngle, sizeof(REAL) * numVertex);		cudaMemset(d_vAngle, 0, sizeof(REAL) * numVertex);
}

void Vertex::copyToDevice(const Mesh& mesh)
{
	cudaMemcpy(d_restPos, &mesh.h_pos[0], sizeof(REAL3) * mesh.h_pos.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pos, &mesh.h_pos[0], sizeof(REAL3) * mesh.h_pos.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Pos1, &mesh.h_pos1[0], sizeof(REAL3) * mesh.h_pos1.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Vel, &mesh.h_vel[0], sizeof(REAL3) * mesh.h_vel.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vNormal, &mesh.h_vNormal[0], sizeof(REAL3) * mesh.h_vNormal.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_InvMass, &mesh.h_invMass[0], sizeof(REAL) * mesh.h_invMass.size(), cudaMemcpyHostToDevice);

	d_nbFaces.copyFromHost(mesh.h_nbVFaces);
	d_nbVertices.copyFromHost(mesh.h_nbVertices);
}

void Vertex::copyToHost(Mesh& mesh)
{
	cudaMemcpy(&mesh.h_pos[0], d_Pos, sizeof(REAL3) * mesh.h_pos.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(&mesh.h_vNormal[0], d_vNormal, sizeof(REAL3) * mesh.h_vNormal.size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(&mesh.h_vAngle[0], d_vAngle, sizeof(REAL) * mesh.h_vAngle.size(), cudaMemcpyDeviceToHost);
}

void Vertex::FreeDeviceMem(void)
{
	cudaFree(d_restPos);
	cudaFree(d_Pos);
	cudaFree(d_Pos1);
	cudaFree(d_Vel);
	cudaFree(d_vNormal);
	cudaFree(d_InvMass);
	cudaFree(d_SatMass);
	cudaFree(d_vSaturation);
	cudaFree(d_Adhesion);
	cudaFree(d_vAngle);

	d_nbFaces.clear();
	d_nbVertices.clear();
}