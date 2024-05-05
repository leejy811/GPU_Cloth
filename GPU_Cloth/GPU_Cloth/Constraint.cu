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
	vector<set<uint>> nbVs(_numConstraint);

	for (int i = 0; i < h_GraphIdx.size(); i++)
	{
		nbVs[h_GraphIdx[i].x].insert(h_GraphIdx[i].y);
		nbVs[h_GraphIdx[i].y].insert(h_GraphIdx[i].x);
	}

	h_nbGVertices.init(nbVs);
}

void Constraint::InitConstraintColor(void)
{
	vector<uint> v(_numConstraint);

	for (int i = 1; i < _numConstraint; i++)
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

	_numColor = *max_element(v.begin(), v.end()) + 1;
	vector<set<uint>> graph(_numColor);
	for (int i = 0; i < _numConstraint; i++)
	{
		graph[v[i]].insert(i);
	}

	h_ColorIdx.init(graph);

	int count = 0;

	colorEdges.resize(_numConstraint, false);
	for (int i = 0; i < _numColor; i++)
	{
		uint numConst = h_ColorIdx._index[i + 1] - h_ColorIdx._index[i];
		for (int j = 0; j < numConst; j++)
		{
			for (int k = j + 1; k < numConst; k++)
			{
				uint eid0 = h_ColorIdx._array[h_ColorIdx._index[i] + j];
				uint eid1 = h_ColorIdx._array[h_ColorIdx._index[i] + k];
				uint2 e0 = h_EdgeIdx[eid0];
				uint2 e1 = h_EdgeIdx[eid1];

				if (e0.x == e1.x || e0.x == e1.y || e0.y == e1.x || e0.y == e1.y)
				{
					colorEdges[eid0] = true;
					colorEdges[eid1] = true;
					count += 2;
				}
			}
		}
	}

	printf("%d / %d\n", count, _numConstraint);
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

void Constraint::Draw(vector<REAL3>& pos, bool isBend)
{
	/*for (int i = 0; i < _numColor; i++)
	{
		uint numConst = h_ColorIdx._index[i + 1] - h_ColorIdx._index[i];
		for (int j = 0; j < numConst; j++)
		{
			uint eid = h_ColorIdx._array[h_ColorIdx._index[i] + j];
			uint ino0 = h_EdgeIdx[eid].x;
			uint ino1 = h_EdgeIdx[eid].y;
			REAL3 a = pos[ino0];
			REAL3 b = pos[ino1];

			if (i == 2)
				glColor3f(1, 0, 0);
			else
				glColor3f(0, 1, 0);

			glVertex3f(a.x, a.y, a.z);
			glVertex3f(b.x, b.y, b.z);
		}
	}*/

	for (int i = 0; i < _numConstraint; i++)
	{
		uint ino0 = h_EdgeIdx[i].x;
		uint ino1 = h_EdgeIdx[i].y;
		REAL3 a = pos[ino0];
		REAL3 b = pos[ino1];

		if (isBend)
		{
			if (colorEdges[i] && Length(a - b) > 0.5)
				glColor3f(1, 0, 0);
			else
				glColor3f(0, 1, 0);

			glVertex3f(a.x, a.y, a.z);
			glVertex3f(b.x, b.y, b.z);
		}
		else
		{
			glColor3f(0, 1, 0);
			glVertex3f(a.x, a.y, a.z);
			glVertex3f(b.x, b.y, b.z);
		}
	}
}