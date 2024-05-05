#include "Constraint.h"
#include <stdio.h>

__constant__ ConstParam constParam;

__global__ void SolveDistConstraint_kernel(Vertex ver, uint2* eIdx, REAL* rest, uint* colorArray, uint* colorIdx, uint cid, uint numConst)
{
	//uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	//if (idx >= numConst)
	//	return;

	//uint eid = colorArray[colorIdx[cid] + idx];
	//uint id0 = eIdx[eid].x;
	//uint id1 = eIdx[eid].y;

	//REAL c_p1p2 = Length(ver.d_Pos1[id0] - ver.d_Pos1[id1]) - rest[eid];
	//REAL3 dp1 = ver.d_Pos1[id0] - ver.d_Pos1[id1];
	//REAL3 dp2 = ver.d_Pos1[id0] - ver.d_Pos1[id1];
	//Normalize(dp1);
	//Normalize(dp2);
	//dp1 *= -ver.d_InvMass[id0] / (ver.d_InvMass[id0] + ver.d_InvMass[id1]) * c_p1p2;
	//dp2 *= ver.d_InvMass[id1] / (ver.d_InvMass[id0] + ver.d_InvMass[id1]) * c_p1p2;
	//dp1 *= 1.0 - powf((1.0 - constParam._springK), 1.0 / (REAL)constParam._iteration);
	//dp2 *= 1.0 - powf((1.0 - constParam._springK), 1.0 / (REAL)constParam._iteration);
	//ver.d_Pos1[id0] += dp1;
	//ver.d_Pos1[id1] += dp2;
}