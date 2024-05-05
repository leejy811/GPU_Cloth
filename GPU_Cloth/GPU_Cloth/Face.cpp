#include "Face.h"

Face::Face()
{
}


Face::~Face()
{
}

int	Face::getIndex(Vertex* v)
{
	for (int i = 0; i < 3; i++)
	{
		if (_vertices[i] == v)
		{
			return i;
		}
	}
	return -1;
}