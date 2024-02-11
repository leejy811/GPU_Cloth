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

	InitDeviceMem();
	copyToDevice();
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
	//buildAdjacency();
	computeNormal();

	printf("Num of Faces: %d, Num of Vertices: %d\n", _numFaces, _numVertices);
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

void PBD_ClothCuda::ComputeGravityForce_kernel(REAL3& gravity, REAL dt)
{
	ComputeGravity_kernel << <divup(_numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_Pos1(), d_Vel(), gravity, _linearDamping, _numVertices, dt);
}

void PBD_ClothCuda::Intergrate_kernel(REAL invdt)
{
	ComputeIntergrate_kernel << <divup(_numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_Pos1(), d_Vel(), _numVertices, invdt);
}

void PBD_ClothCuda::ComputeNormal_kernel(void)
{
	ComputeNorm_kernel << <divup(_numFaces, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_faceIdx(), d_Pos(), d_fNormal(), _numFaces);
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

		glNormal3f(h_fNormal[i].x, h_fNormal[i].y, h_fNormal[i].z);
		glVertex3f(a.x, a.y, a.z);
		glVertex3f(b.x, b.y, b.z);
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