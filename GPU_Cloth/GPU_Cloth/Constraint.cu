#include "Constraint.cuh"

Constraint::Constraint()
{

}

Constraint::~Constraint()
{

}

void Constraint::InitConstraintColor(void)
{

}

void Constraint::IterateConstraint(Dvector<REAL3>& pos1, Dvector<REAL>& invm)
{
	for (int i = 0; i < _numColor; i++)
	{
		uint numConst = h_ColorIdx._index[i + 1] - h_ColorIdx._index[i];
		SolveDistanceConstraint_kernel(numConst, i, pos1, invm);
	}
}

void Constraint::SolveDistanceConstraint_kernel(uint numConst, uint idx, Dvector<REAL3>& pos1, Dvector<REAL>& invm)
{
	SolveDistConstraint_kernel << <divup(numConst, CONST_BLOCK_SIZE), CONST_BLOCK_SIZE >> >
		(pos1(), invm(), d_EdgeIdx(), d_RestLength(), &d_ColorIdx._array()[idx], _springK, _iteration, numConst);
}

void Constraint::InitDeviceMem(void)
{
	d_EdgeIdx.resize(_numConstraint);				d_EdgeIdx.memset(0);
	d_RestLength.resize(_numConstraint);		d_RestLength.memset(0);
}

void	Constraint::copyToDevice(void)
{
	d_EdgeIdx.copyFromHost(h_EdgeIdx);
	d_RestLength.copyFromHost(h_RestLength);
}

void	Constraint::copyToHost(void)
{
	d_EdgeIdx.copyToHost(h_EdgeIdx);
	d_RestLength.copyToHost(h_RestLength);
}

void Constraint::FreeDeviceMem(void)
{
	d_EdgeIdx.free();
	d_RestLength.free();
}

