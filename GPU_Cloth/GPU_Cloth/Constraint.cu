#include "Constraint.cuh"
#include <algorithm>
#include <GL/glut.h>

Constraint::Constraint()
{
	
}

Constraint::~Constraint()
{
	FreeDeviceMem();
}

void Constraint::Init(int numVertices)
{
	InitGraphEdge(numVertices);
	InitGraphAdjacency();
	InitConstraintColor();

	cudaMemcpyToSymbol(&constParam, &_param, sizeof(ConstParam));
}

void Constraint::InitGraphEdge(int numVertices)
{
	for (int i = 0; i < numVertices; i++)
	{
		for (int j = h_nbCEdges._index[i]; j < h_nbCEdges._index[i + 1]; j++)
		{
			for (int k = j + 1; k < h_nbCEdges._index[i + 1]; k++)
			{
				h_GraphIdx.push_back(make_uint2(h_nbCEdges._array[j], h_nbCEdges._array[k]));
			}
		}
	}
}

void Constraint::InitGraphAdjacency()
{
	vector<set<uint>> nbVs(_param._numConstraint);

	for (int i = 0; i < h_GraphIdx.size(); i++)
	{
		nbVs[h_GraphIdx[i].x].insert(h_GraphIdx[i].y);
		nbVs[h_GraphIdx[i].y].insert(h_GraphIdx[i].x);
	}

	h_nbGVertices.init(nbVs);
}

void Constraint::InitConstraintColor(void)
{
	vector<uint> v(_param._numConstraint);

	for (int i = 1; i < _param._numConstraint; i++)
	{
		set<uint> neighbours;

		for (int j = h_nbGVertices._index[i]; j < h_nbGVertices._index[i + 1]; j++)
		{
			uint nbVId = h_nbGVertices._array[j];
			neighbours.insert(v[nbVId]);
		}

		uint smallest = 1;

		for (;; ++smallest)
		{
			if (neighbours.find(smallest) == neighbours.end())
				break;
		}

		v[i] = smallest;
	}

	_param._numColor = *max_element(v.begin(), v.end()) + 1;
	vector<set<uint>> graph(_param._numColor);
	for (int i = 0; i < _param._numConstraint; i++)
	{
		graph[v[i]].insert(i);
	}

	h_ColorIdx.init(graph);
}

void Constraint::IterateConstraint(Vertex vertex)
{
	for (int i = 0; i < _param._numColor; i++)
	{
		uint numConst = h_ColorIdx._index[i + 1] - h_ColorIdx._index[i];
		SolveDistanceConstraint_kernel(numConst, i, vertex);
	}
}

void Constraint::SolveDistanceConstraint_kernel(uint numConst, uint idx, Vertex vertex)
{
	SolveDistConstraint_kernel << <divup(numConst, CONST_BLOCK_SIZE), CONST_BLOCK_SIZE >> >
		(vertex, d_EdgeIdx(), d_RestLength(), d_ColorIdx._array(), d_ColorIdx._index(), idx, numConst);
}

void Constraint::InitDeviceMem(void)
{
	d_EdgeIdx.resize(_param._numConstraint);		d_EdgeIdx.memset(0);
	d_RestLength.resize(_param._numConstraint);		d_RestLength.memset(0);
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