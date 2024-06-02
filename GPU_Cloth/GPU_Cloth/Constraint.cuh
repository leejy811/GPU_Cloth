#include "Constraint.h"

__constant__ ConstParam constParam;

__global__ void SolveDistConstraint_kernel(REAL3* pos, REAL* invm, REAL* satm, uint2* eIdx, REAL* rest, uint* colorArray, uint* colorIdx, uint cid, uint numConst)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numConst)
		return;

	uint eid = colorArray[colorIdx[cid] + idx];
	uint id0 = eIdx[eid].x;
	uint id1 = eIdx[eid].y;

	REAL invmass0 = 1.0 / ((1.0 / invm[id0]) + satm[id0]);
	REAL invmass1 = 1.0 / ((1.0 / invm[id1]) + satm[id1]);

	REAL c_p1p2 = Length(pos[id0] - pos[id1]) - rest[eid];
	REAL3 dp1 = pos[id0] - pos[id1];
	REAL3 dp2 = pos[id0] - pos[id1];
	Normalize(dp1);
	Normalize(dp2);
	dp1 *= -invmass0 / (invmass0 + invmass1) * c_p1p2;
	dp2 *= invmass1 / (invmass0 + invmass1) * c_p1p2;
	dp1 *= 1.0 - powf((1.0 - constParam._springK), 1.0 / (REAL)constParam._iteration);
	dp2 *= 1.0 - powf((1.0 - constParam._springK), 1.0 / (REAL)constParam._iteration);
	pos[id0] += dp1;
	pos[id1] += dp2;
}