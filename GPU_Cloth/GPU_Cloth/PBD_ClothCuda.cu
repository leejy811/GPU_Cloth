#include "PBD_ClothCuda.cuh"

PBD_ClothCuda::PBD_ClothCuda()
{
	
}

PBD_ClothCuda::~PBD_ClothCuda()
{
	FreeDeviceMem();
}

void PBD_ClothCuda::Init(uint iter, REAL damp, REAL stiff)
{
	_iteration = iter;
	_linearDamping = damp;
	_springK = stiff;

	h_pos1.resize(_numVertices, make_REAL3(0.0, 0.0, 0.0));
	h_vel.resize(_numVertices, make_REAL3(0.0, 0.0, 0.0));
	h_invMass.resize(_numVertices, 0.0);

	_strechSpring->InitDeviceMem();
	_strechSpring->copyToDevice();

	_bendSpring->InitDeviceMem();
	_bendSpring->copyToDevice();

	InitDeviceMem();
	copyToDevice();
	copyNbToDevice();
}

void	PBD_ClothCuda::LoadObj(char* filename)
{
	h_faceIdx.clear();
	h_pos.clear();

	bool flag = true;
	ifstream fin;
	fin.open(filename);
	if (fin.is_open())
	{
		while (!fin.eof())
		{
			string head;
			fin >> head;
			if (head.length() > 1)
				continue;
			if (head[0] == 'v')
			{
				REAL3 x;
				fin >> x.x >> x.y >> x.z;
				h_pos.push_back(make_REAL3(x.x, x.y, x.z));
				if (flag)
				{
					_boundary._min = _boundary._max = x;
					flag = false;
				}
				else addAABB(_boundary, x);
			}
			else if (head[0] == 'f')
			{
				uint3 x;
				fin >> x.x >> x.y >> x.z;
				h_faceIdx.push_back(make_uint3(x.x - 1u, x.y - 1u, x.z - 1u));
			}
		}
		fin.close();
	}
	if (h_pos.empty() || h_faceIdx.empty())
	{
		printf("Error : Mesh_init : Object Load Error\n");
		exit(1);
		return;
	}
	_numFaces = h_faceIdx.size();
	_numVertices = h_pos.size();
	moveCenter(1.0);
	SetMass();
	buildAdjacency();
	computeNormal();

	printf("Num of Faces: %d, Num of Vertices: %d\n", _numFaces, _numVertices);
	printf("Num of Strech: %d, Num of Bend: %d\n", _strechSpring->_numConstraint, _strechSpring->_numConstraint);
	printf("Num of Color Strech: %d, Num of Color Bend: %d\n", _strechSpring->_numColor, _strechSpring->_numColor);
}

void PBD_ClothCuda::moveCenter(REAL scale)
{
	REAL3 size = _boundary._max - _boundary._min;
	REAL max_length = size.x;
	if (max_length < size.y)
		max_length = size.y;
	if (max_length < size.z)
		max_length = size.z;
	max_length = 2.0 * scale / max_length;

	REAL3 prevCenter = (_boundary._min + _boundary._max) * (REAL)0.5;

	bool flag = false;
	uint vlen = h_pos.size();
	for (uint i = 0u; i < vlen; i++)
	{
		REAL3 pos = h_pos[i];
		REAL3 grad = pos - prevCenter;
		grad *= max_length;
		pos = grad;
		h_pos[i] = pos;
		if (flag) addAABB(_boundary, pos);
		else
		{
			_boundary._min = _boundary._max = pos;
			flag = true;
		}
	}
}

void PBD_ClothCuda::SetMass(void)
{
	for (int i = 0; i < _numVertices; i++)
	{
		h_invMass.push_back(1.0);
	}
}

void PBD_ClothCuda::buildAdjacency(void)
{
	//Neighbor
	vector<set<uint>> nbFs(_numVertices);
	vector<set<uint>> nbVs(_numVertices);

	for (uint i = 0u; i < _numFaces; i++)
	{
		uint ino0 = h_faceIdx[i].x;
		uint ino1 = h_faceIdx[i].y;
		uint ino2 = h_faceIdx[i].z;
		nbFs[ino0].insert(i);
		nbFs[ino1].insert(i);
		nbFs[ino2].insert(i);
		nbVs[ino0].insert(ino1);
		nbVs[ino0].insert(ino2);
		nbVs[ino1].insert(ino2);
		nbVs[ino1].insert(ino0);
		nbVs[ino2].insert(ino0);
		nbVs[ino2].insert(ino1);
	}

	h_nbFaces.init(nbFs);
	h_nbVertices.init(nbVs);

	//Constraint
	_strechSpring = new Constraint(5, 0.9);
	_bendSpring = new Constraint(5, 0.9);

	for (uint i = 0u; i < _numFaces; i++)
	{
		uint ino[3] = { h_faceIdx[i].x, h_faceIdx[i].y, h_faceIdx[i].z };

		for (int j = 0; j < 3; j++)
		{
			uint id0 = ino[j];
			uint id1 = ino[(j + 1) % 3];
			_strechSpring->h_EdgeIdx.push_back(make_uint2(id0, id1));
			_strechSpring->h_RestLength.push_back(Length(h_pos[id0] - h_pos[id1]));
			_strechSpring->_numConstraint++;
		}

		for (int j = 0; j < 3; j++)
		{
			uint id0 = ino[j];
			uint id1 = ino[(j + 1) % 3];
			uint id2 = ino[(j + 2) % 3];
			for (int k = 0; k < h_nbVertices._index[id0 + 1] - h_nbVertices._index[id0]; k++)
			{
				bool fiag = false;
				for (int u = 0; u < h_nbVertices._index[id1 + 1] - h_nbVertices._index[id1]; u++)
				{
					if (h_nbVertices._array[id0 + k] == h_nbVertices._array[id1 + u] && h_nbVertices._array[id0 + k] != id2)
					{
						_bendSpring->h_EdgeIdx.push_back(make_uint2(id2, h_nbVertices._array[id0 + k]));
						_bendSpring->h_RestLength.push_back(Length(h_pos[id2] - h_pos[h_nbVertices._array[id0 + k]]));
						_bendSpring->_numConstraint++;
						fiag = true;
						break;
					}
				}
				if (fiag) break;
			}
		}
	}

	_strechSpring->Init();
	_bendSpring->Init();
}

void PBD_ClothCuda::computeNormal(void)
{
	h_fNormal.resize(_numFaces);
	h_vNormal.clear();
	h_vNormal.resize(_numVertices, make_REAL3(0.0, 0.0, 0.0));

	for (uint i = 0u; i < _numFaces; i++)
	{
		uint ino0 = h_faceIdx[i].x;
		uint ino1 = h_faceIdx[i].y;
		uint ino2 = h_faceIdx[i].z;

		REAL3 a = h_pos[ino0];
		REAL3 b = h_pos[ino1];
		REAL3 c = h_pos[ino2];

		REAL3 norm = Cross(a - b, a - c);
		Normalize(norm);
		h_fNormal[i] = norm;

		REAL radian = AngleBetweenVectors(a - b, a - c);
		h_vNormal[ino0] += norm * radian;
		radian = AngleBetweenVectors(b - a, b - c);
		h_vNormal[ino1] += norm * radian;
		radian = AngleBetweenVectors(c - a, c - b);
		h_vNormal[ino2] += norm * radian;
	}

	for (uint i = 0u; i < _numVertices; i++)
	{
		Normalize(h_vNormal[i]);
	}
}

void PBD_ClothCuda::ComputeExternalForce_kernel(REAL3& gravity, REAL dt)
{
	CompExternlaForce_kernel << <divup(_numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_Pos1(), d_Vel(), d_InvMass(), gravity, _externalForce, _linearDamping, _numVertices, dt);
}

void PBD_ClothCuda::Intergrate_kernel(REAL invdt)
{
	CompIntergrate_kernel << <divup(_numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_Pos1(), d_Vel(), _numVertices, invdt);
}

void PBD_ClothCuda::ComputeFaceNormal_kernel(void)
{
	CompFaceNorm_kernel << <divup(_numFaces, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_faceIdx(), d_Pos(), d_fNormal(), _numFaces);
}

void PBD_ClothCuda::ComputeVertexNormal_kernel(void)
{
	CompVertexNorm_kernel << <divup(_numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_nbFaces._index(), d_nbFaces._array(), d_fNormal(), d_vNormal(), _numVertices);
}

void PBD_ClothCuda::ComputeWind_kernel(REAL3 wind)
{
	CompWind_kernel << <divup(_numFaces, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_faceIdx(), d_Pos1(), d_Vel(), wind, _numFaces);
}

void PBD_ClothCuda::ProjectConstraint_kernel(void)
{
	for (int i = 0; i < _iteration; i++)
	{
		_strechSpring->IterateConstraint(d_Pos1, d_InvMass);
		_bendSpring->IterateConstraint(d_Pos1, d_InvMass);
	}
}

void PBD_ClothCuda::draw(void)
{
	glEnable(GL_LIGHTING);
	for (uint i = 0u; i < _numFaces; i++)
	{
		uint ino0 = h_faceIdx[i].x;
		uint ino1 = h_faceIdx[i].y;
		uint ino2 = h_faceIdx[i].z;
		REAL3 a = h_pos[ino0];
		REAL3 b = h_pos[ino1];
		REAL3 c = h_pos[ino2];

		glBegin(GL_POLYGON);

		glNormal3f(h_vNormal[ino0].x, h_vNormal[ino0].y, h_vNormal[ino0].z);
		glVertex3f(a.x, a.y, a.z);
		glNormal3f(h_vNormal[ino1].x, h_vNormal[ino1].y, h_vNormal[ino1].z);
		glVertex3f(b.x, b.y, b.z);
		glNormal3f(h_vNormal[ino2].x, h_vNormal[ino2].y, h_vNormal[ino2].z);
		glVertex3f(c.x, c.y, c.z);

		glEnd();
	}
	glEnable(GL_LIGHTING);
}

void PBD_ClothCuda::InitDeviceMem(void)
{
	d_faceIdx.resize(_numFaces);			d_faceIdx.memset(0);
	d_Pos.resize(_numVertices);				d_Pos.memset(0);
	d_Pos1.resize(_numVertices);			d_Pos1.memset(0);
	d_Vel.resize(_numVertices);				d_Vel.memset(0);
	d_fNormal.resize(_numFaces);			d_fNormal.memset(0);
	d_vNormal.resize(_numVertices);		d_vNormal.memset(0);
	d_InvMass.resize(_numVertices);		d_InvMass.memset(0);
}

void PBD_ClothCuda::FreeDeviceMem(void)
{
	d_faceIdx.free();
	d_Pos.free();
	d_Pos1.free();
	d_Vel.free();
	d_fNormal.free();
	d_vNormal.free();
	d_InvMass.free();
}

void	PBD_ClothCuda::copyToDevice(void)
{
	d_faceIdx.copyFromHost(h_faceIdx);
	d_Pos.copyFromHost(h_pos);
	d_Pos1.copyFromHost(h_pos1);
	d_Vel.copyFromHost(h_vel);
	d_fNormal.copyFromHost(h_fNormal);
	d_vNormal.copyFromHost(h_vNormal);
	d_InvMass.copyFromHost(h_invMass);
}

void	PBD_ClothCuda::copyToHost(void)
{
	d_faceIdx.copyToHost(h_faceIdx);
	d_Pos.copyToHost(h_pos);
	d_Pos1.copyToHost(h_pos1);
	d_Vel.copyToHost(h_vel);
	d_fNormal.copyToHost(h_fNormal);
	d_vNormal.copyToHost(h_vNormal);
	d_InvMass.copyToHost(h_invMass);
}

void	PBD_ClothCuda::copyNbToDevice(void)
{
	d_nbFaces.copyFromHost(h_nbFaces);
	d_nbVertices.copyFromHost(h_nbVertices);
}

void	PBD_ClothCuda::copyNbToHost(void)
{
	d_nbFaces.copyToHost(h_nbFaces);
	d_nbVertices.copyToHost(h_nbVertices);
}