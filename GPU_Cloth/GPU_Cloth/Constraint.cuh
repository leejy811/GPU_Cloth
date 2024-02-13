#include "Constraint.h"

__global__ void SolveDistConstraint_kernel(REAL3* pos1, REAL* invm, uint2* eIdx, REAL* rest, uint* ColorArray, REAL k, uint iter, uint num)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= num)
		return;

	uint eid = ColorArray[idx];
	uint id0 = eIdx[eid].x;
	uint id1 = eIdx[eid].y;

	REAL c_p1p2 = Length((pos1[id0] - pos1[id1])) - rest[eid];
	REAL3 dp1 = pos1[id0] - pos1[id1];
	REAL3 dp2 = pos1[id0] - pos1[id1];
	Normalize(dp1);
	Normalize(dp2);
	dp1 *= invm[id0] / (invm[id0] + invm[id1]) * c_p1p2;
	dp2 *= invm[id1] / (invm[id0] + invm[id1]) * c_p1p2;
	dp1 *= 1.0 - pow((1.0 - k), 1.0 / iter);
	dp2 *= 1.0 - pow((1.0 - k), 1.0 / iter);
	pos1[id0] += dp1;
	pos1[id1] += dp2;
}