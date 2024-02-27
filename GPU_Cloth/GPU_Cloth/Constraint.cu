#include "Constraint.cuh"
#include <algorithm>

Constraint::Constraint()
{
	
}

Constraint::~Constraint()
{
	FreeDeviceMem();
}

void Constraint::Init(void)
{
	InitGraphEdge();
	InitConstraintColor();
}

void Constraint::InitGraphEdge(void)
{
	for (int i = 0; i < _numConstraint; i++)
	{
		for (int j = i + 1; j < _numConstraint; j++)
		{
			if (h_EdgeIdx[i].x == h_EdgeIdx[j].x || h_EdgeIdx[i].x == h_EdgeIdx[j].y || h_EdgeIdx[i].y == h_EdgeIdx[j].x || h_EdgeIdx[i].y == h_EdgeIdx[j].y)
			{
				h_GraphIdx.push_back(make_uint2(i, j));
			}
		}
	}
}

void Constraint::InitConstraintColor(void)
{
	vector<uint> v(_numConstraint);
	for (int i = 1; i < _numConstraint; ++i)
	{
		vector<uint2> neighbour_edge;

		for (auto e : h_GraphIdx)
		{
			if (e.x == i || e.y == i)
				neighbour_edge.push_back(e);
		}

		set<uint> neighbours;

		for (const auto& e : neighbour_edge)
		{
			if (e.x == i)
				neighbours.insert(v[e.y]);
			else if (e.y == i)
				neighbours.insert(v[e.x]);
		}

		uint smallest = 1;

		for (;; ++smallest)
		{
			if (neighbours.find(smallest) == neighbours.end())
				break;
		}

		v[i] = smallest;
	}

	_numColor = *max_element(v.begin(), v.end()) + 1;
	vector<set<uint>> graph(_numColor);
	for (int i = 0; i < _numConstraint; i++)
	{
		graph[v[i]].insert(i);
	}

	h_ColorIdx.init(graph);
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
		(pos1(), invm(), d_EdgeIdx(), d_RestLength(), d_ColorIdx._array(), d_ColorIdx._index(), _springK, _iteration, idx, numConst);
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
	d_ColorIdx.copyFromHost(h_ColorIdx);
}

void	Constraint::copyToHost(void)
{
	d_EdgeIdx.copyToHost(h_EdgeIdx);
	d_RestLength.copyToHost(h_RestLength);
	d_ColorIdx.copyToHost(h_ColorIdx);
}

void Constraint::FreeDeviceMem(void)
{
	d_EdgeIdx.free();
	d_RestLength.free();
	d_ColorIdx.clear();
}

