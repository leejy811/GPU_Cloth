#include "Vertex.h"

Vertex::Vertex()
{

}

Vertex::~Vertex()
{

}

void Vertex::InitDeviceMem(uint numVertex)
{
	d_restPos.resize(numVertex);		d_restPos.memset(0);
	d_Pos.resize(numVertex);			d_Pos.memset(0);
	d_Pos1.resize(numVertex);			d_Pos1.memset(0);
	d_Vel.resize(numVertex);			d_Vel.memset(0);
	d_vNormal.resize(numVertex);		d_vNormal.memset(0);
	d_InvMass.resize(numVertex);		d_InvMass.memset(0);
}

void Vertex::copyToDevice(void)
{
	d_restPos.copyFromHost(h_pos);
	d_Pos.copyFromHost(h_pos);
	d_Pos1.copyFromHost(h_pos1);
	d_Vel.copyFromHost(h_vel);
	d_vNormal.copyFromHost(h_vNormal);
	d_InvMass.copyFromHost(h_invMass);

	d_nbFaces.copyFromHost(h_nbVFaces);
	d_nbVertices.copyFromHost(h_nbVertices);
}

void Vertex::copyToHost(void)
{
	d_Pos.copyToHost(h_pos);
	d_vNormal.copyToHost(h_vNormal);
}

void Vertex::copyHostValue(const Mesh& mesh)
{
	h_pos = mesh.h_pos;
	h_pos1 = mesh.h_pos1;
	h_vel = mesh.h_vel;
	h_vNormal = mesh.h_vNormal;
	h_invMass = mesh.h_invMass;
	h_nbVFaces = mesh.h_nbVFaces;
	h_nbVertices = mesh.h_nbVertices;
}

void Vertex::FreeDeviceMem(void)
{
	d_restPos.free();
	d_Pos.free();
	d_Pos1.free();
	d_Vel.free();
	d_vNormal.free();
	d_InvMass.free();

	d_nbFaces.clear();
	d_nbVertices.clear();
}