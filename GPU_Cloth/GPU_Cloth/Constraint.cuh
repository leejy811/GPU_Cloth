#include "Constraint.h"
#include <stdio.h>

__constant__ ConstParam constParam;

__global__ void SolveDistConstraint_kernel(REAL3* pos, REAL* invm, uint2* eIdx, REAL* rest, uint* colorArray, uint* colorIdx, uint cid, uint numConst)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numConst)
		return;

	uint eid = colorArray[colorIdx[cid] + idx];
	uint id0 = eIdx[eid].x;
	uint id1 = eIdx[eid].y;

	REAL c_p1p2 = Length(pos[id0] - pos[id1]) - rest[eid];
	REAL3 dp1 = pos[id0] - pos[id1];
	REAL3 dp2 = pos[id0] - pos[id1];
	Normalize(dp1);
	Normalize(dp2);
	dp1 *= -invm[id0] / (invm[id0] + invm[id1]) * c_p1p2;
	dp2 *= invm[id1] / (invm[id0] + invm[id1]) * c_p1p2;
	dp1 *= 1.0 - powf((1.0 - constParam._springK), 1.0 / (REAL)constParam._iteration);
	dp2 *= 1.0 - powf((1.0 - constParam._springK), 1.0 / (REAL)constParam._iteration);
	pos[id0] += dp1;
	pos[id1] += dp2;
}