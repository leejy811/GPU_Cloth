#include "Face.h"

Face::Face()
{

}

Face::~Face()
{

}

void Face::InitDeviceMem(uint numFace)
{
	d_faceIdx.resize(numFace);		d_faceIdx.memset(0);
	d_faceAABB.resize(numFace);			d_faceAABB.memset(0);
	d_fNormal.resize(numFace);			d_fNormal.memset(0);
}

void Face::copyToDevice(void)
{
	d_faceIdx.copyFromHost(h_faceIdx);
	d_fNormal.copyFromHost(h_fNormal);
}

void Face::copyHostValue(const Mesh& mesh)
{
	h_faceIdx = mesh.h_faceIdx;
	h_fNormal = mesh.h_fNormal;
}

void Face::FreeDeviceMem(void)
{
	d_faceIdx.free();
	d_faceAABB.free();
	d_fNormal.free();
}