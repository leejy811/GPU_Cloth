#include "Vertex.h"

Vertex::Vertex()
{
}


Vertex::~Vertex()
{
}

bool Vertex::hasNbVertex(Vertex* v)
{
	for (auto nv : _nbVertices)
	{
		if (nv == v)
		{
			return true;
		}
	}
	return false;
}