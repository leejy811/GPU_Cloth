#include "DeviceVertex.cuh"

DeviceVertex::DeviceVertex()
{

}

DeviceVertex::~DeviceVertex()
{

}

void DeviceVertex::InitMem()
{
	cudaMalloc(&_dPos, sizeof(float3) * _numVertices);			cudaMemset(_dPos, 0, sizeof(float3) * _numVertices);
	cudaMalloc(&_dPos1, sizeof(float3) * _numVertices);		cudaMemset(_dPos1, 0, sizeof(float3) * _numVertices);
	cudaMalloc(&_dVel, sizeof(float3) * _numVertices);			cudaMemset(_dVel, 0, sizeof(float3) * _numVertices);
	cudaMalloc(&_dNormal, sizeof(float3) * _numVertices);		cudaMemset(_dNormal, 0, sizeof(float3) * _numVertices);
	cudaMalloc(&_dInvMass, sizeof(float) * _numVertices);		cudaMemset(_dInvMass, 0, sizeof(float) * _numVertices);
}

void DeviceVertex::FreeMem()
{
	cudaFree(_dPos);
	cudaFree(_dPos1);
	cudaFree(_dVel);
	cudaFree(_dNormal);
	cudaFree(_dInvMass);
}

void DeviceVertex::CopyToDevice()
{
	cudaMemcpy(_dPos, &_pos[0], sizeof(float3) * _numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(_dPos1, &_pos1[0], sizeof(float3) * _numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(_dVel, &_vel[0], sizeof(float3) * _numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(_dNormal, &_normal[0], sizeof(float3) * _numVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(_dInvMass, &_invMass[0], sizeof(float) * _numVertices, cudaMemcpyHostToDevice);
}

void DeviceVertex::CopyToHost()
{
	cudaMemcpy(&_pos[0], _dPos, sizeof(float3) * _numVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_pos1[0], _dPos1, sizeof(float3) * _numVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_vel[0], _dVel, sizeof(float3) * _numVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_normal[0], _dNormal, sizeof(float3) * _numVertices, cudaMemcpyDeviceToHost);
	cudaMemcpy(&_invMass[0], _dInvMass, sizeof(float) * _numVertices, cudaMemcpyDeviceToHost);
}