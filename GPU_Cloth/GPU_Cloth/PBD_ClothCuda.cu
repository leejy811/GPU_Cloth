#include "PBD_ClothCuda.cuh"

PBD_ClothCuda::PBD_ClothCuda()
{
	
}

PBD_ClothCuda::~PBD_ClothCuda()
{
	FreeDeviceMem();
}

void PBD_ClothCuda::Init()
{
	_thickness = 4.0 / _gridRes;
	h_flag.resize(_numVertices, false);
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

	d_restPos.resize(_numVertices);
	d_restPos.memset(0);
	d_restPos.copyFromHost(h_pos);
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
	moveCenter(0.5);
	SetMass();
	buildAdjacency();
	computeNormal();

	REAL3 maxPos = make_REAL3(-100.0, -100.0, -100.0);
	REAL3 minPos = make_REAL3(100.0, 100.0, 100.0);

	for (int i = 0; i < _numVertices; i++)
	{
		if (maxPos.x < h_pos[i].x)
			maxPos.x = h_pos[i].x;
		if (maxPos.y < h_pos[i].y)
			maxPos.y = h_pos[i].y;
		if (maxPos.z < h_pos[i].z)
			maxPos.z = h_pos[i].z;

		if (minPos.x > h_pos[i].x)
			minPos.x = h_pos[i].x;
		if (minPos.y > h_pos[i].y)
			minPos.y = h_pos[i].y;
		if (minPos.z > h_pos[i].z)
			minPos.z = h_pos[i].z;
	}

	printf("mas Pos : %f / %f / %f\n", maxPos.x, maxPos.y, maxPos.z);
	printf("min Pos : %f / %f / %f\n", minPos.x, minPos.y, minPos.z);

	printf("Num of Faces: %d, Num of Vertices: %d\n", _numFaces, _numVertices);
	printf("Num of Strech: %d, Num of Bend: %d\n", _strechSpring->_numConstraint, _bendSpring->_numConstraint);
	printf("Num of Color Strech: %d, Num of Color Bend: %d\n", _strechSpring->_numColor, _bendSpring->_numColor);
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

	for (int i = 0; i < _numVertices; i++)
	{
		h_pos[i] += make_REAL3(1.0, 1.0, 1.0);
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
	buildAdjacency_VF();
	buildEdges();
	buildAdjacency_EF();
	buildConstraint();
}

void PBD_ClothCuda::buildAdjacency_VF(void)
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

	h_nbVFaces.init(nbFs);
	h_nbVertices.init(nbVs);
}

void PBD_ClothCuda::buildEdges(void)
{
	vector<bool> flags(_numVertices, false);

	for (int i = 0; i < _numVertices; i++)
	{
		for (int j = h_nbVertices._index[i]; j < h_nbVertices._index[i + 1]; j++)
		{
			if (!flags[h_nbVertices._array[j]])
				h_edgeIdx.push_back(make_uint2(i, h_nbVertices._array[j]));
		}
		flags[i] = true;
	}

	_numEdges = h_edgeIdx.size();
}

void PBD_ClothCuda::buildAdjacency_EF(void)
{
	vector<set<uint>> nbEF(_numEdges);

	for (int i = 0; i < _numEdges; i++)
	{
		uint eid0 = h_edgeIdx[i].x;
		uint eid1 = h_edgeIdx[i].y;

		for (int j = h_nbVFaces._index[eid0]; j < h_nbVFaces._index[eid0 + 1]; j++)
		{
			uint fIdx = h_nbVFaces._array[j];
			uint fid0 = h_faceIdx[fIdx].x;
			uint fid1 = h_faceIdx[fIdx].y;
			uint fid2 = h_faceIdx[fIdx].z;
			if ((fid0 == eid0 || fid1 == eid0 || fid2 == eid0) && (fid0 == eid1 || fid1 == eid1 || fid2 == eid1))
			{
				nbEF[i].insert(fIdx);
			}
		}
	}

	h_nbEFaces.init(nbEF);
}

void PBD_ClothCuda::buildConstraint(void)
{
	//Constraint
	_strechSpring = new Constraint(_iteration, _springK);
	_bendSpring = new Constraint(_iteration, _springK);

	vector<set<uint>> stretchEdges(_numVertices);
	vector<set<uint>> bendEdges(_numVertices);

	for (int i = 0; i < _numEdges; i++)
	{
		//strech spring
		_strechSpring->h_EdgeIdx.push_back(h_edgeIdx[i]);
		_strechSpring->h_RestLength.push_back(Length(h_pos[h_edgeIdx[i].x] - h_pos[h_edgeIdx[i].y]));
		_strechSpring->_numConstraint++;
		stretchEdges[h_edgeIdx[i].x].insert(i);
		stretchEdges[h_edgeIdx[i].y].insert(i);

		//bend spring
		if (h_nbEFaces._index[i + 1] - h_nbEFaces._index[i] != 2)
			continue;

		uint edgeSum = h_edgeIdx[i].x + h_edgeIdx[i].y;
		uint fId0 = h_nbEFaces._array[h_nbEFaces._index[i]];
		uint fId1 = h_nbEFaces._array[h_nbEFaces._index[i] + 1];

		uint vId0 = h_faceIdx[fId0].x + h_faceIdx[fId0].y + h_faceIdx[fId0].z - edgeSum;
		uint vId1 = h_faceIdx[fId1].x + h_faceIdx[fId1].y + h_faceIdx[fId1].z - edgeSum;

		_bendSpring->h_EdgeIdx.push_back(make_uint2(vId0, vId1));
		_bendSpring->h_RestLength.push_back(Length(h_pos[vId0] - h_pos[vId1]));

		bendEdges[vId0].insert(_bendSpring->_numConstraint);
		bendEdges[vId1].insert(_bendSpring->_numConstraint);
		
		stretchEdges[h_edgeIdx[i].x].insert(i);
		stretchEdges[h_edgeIdx[i].y].insert(i);

		_bendSpring->_numConstraint++;
	}

	_strechSpring->h_nbCEdges.init(stretchEdges);
	_bendSpring->h_nbCEdges.init(bendEdges);

	_strechSpring->Init(_numVertices);
	_bendSpring->Init(_numVertices);
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
		(d_Pos(), d_Pos1(), d_Vel(), d_InvMass(), gravity, _externalForce, _linearDamping, _numVertices, dt, _thickness);
}

void PBD_ClothCuda::Intergrate_kernel(REAL invdt)
{
	CompIntergrate_kernel << <divup(_numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_Pos1(), d_Vel(), _numVertices, _thickness, invdt);
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
	_strechSpring->IterateConstraint(d_Pos1, d_InvMass);
	_bendSpring->IterateConstraint(d_Pos1, d_InvMass);
}

void PBD_ClothCuda::SetHashTable_kernel(void)
{
	CalculateHash_Kernel();
	SortParticle_Kernel();
	FindCellStart_Kernel();
}

void PBD_ClothCuda::CalculateHash_Kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_numVertices, 256, numBlocks, numThreads);

	CalculateHash_D << <numBlocks, numThreads >> >
		(d_gridHash(), d_gridIdx(), d_Pos1(), _gridRes, _numVertices);
}

void PBD_ClothCuda::SortParticle_Kernel(void)
{
	thrust::sort_by_key(thrust::device_ptr<uint>(d_gridHash()),
		thrust::device_ptr<uint>(d_gridHash() + _numVertices),
		thrust::device_ptr<uint>(d_gridIdx()));
}

void PBD_ClothCuda::FindCellStart_Kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_numVertices, 256, numBlocks, numThreads);

	uint smemSize = sizeof(uint) * (numThreads + 1);
	FindCellStart_D << <numBlocks, numThreads, smemSize >> >
		(d_gridHash(), d_gridIdx(), d_cellStart(), d_cellEnd(), d_Pos1(), d_Vel(), d_sortedPos(), d_sortedVel(), _numVertices);
}

void PBD_ClothCuda::UpdateFaceAABB_Kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_numFaces, 256, numBlocks, numThreads);

	UpdateFaceAABB << <numBlocks, numThreads >> >
		(d_faceIdx(), d_Pos1(), d_faceAABB(), _numFaces);
}

void PBD_ClothCuda::Colide_kernel()
{
	uint numThreads, numBlocks;
	Dvector<REAL> impulse(_numVertices * 3u);
	impulse.memset(0);
	ComputeGridSize(_numVertices, 64, numBlocks, numThreads);
	Colide_PP << <numBlocks, numThreads >> >
		(d_restPos(), d_Pos1(), d_Pos(), d_sortedVel(), d_Vel(), d_gridHash(), d_gridIdx(), d_cellStart(), d_cellEnd(), impulse(), _thickness, _gridRes, _numVertices, d_flag());
	
	ComputeGridSize(_numFaces, 64, numBlocks, numThreads);
	//Colide_PT << <numBlocks, numThreads >> >
	//	(d_faceIdx(), d_Pos1(), d_faceAABB(), d_gridHash(), d_gridIdx(), d_cellStart(), d_cellEnd(), impulse(), _thickness, _gridRes, _numFaces, d_flag());

	ComputeGridSize(_numVertices, 64, numBlocks, numThreads);
	ApplyImpulse_kernel << <numBlocks, numThreads >> >
		(d_Pos1(), impulse(), _numVertices, _selfColliDamping);
}

void PBD_ClothCuda::LevelSetCollision_kernel(void)
{
	LevelSetCollision_D << <divup(_numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Pos(), d_Vel(), _numVertices);
}

void PBD_ClothCuda::draw(void)
{
	//glDisable(GL_LIGHTING);
	
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

	//glBegin(GL_POINTS);
	//glPointSize(5.0f);
	//for (uint i = 0u; i < _numVertices; i++)
	//{
	//	REAL3 a = h_pos[i];

	//	if (h_flag[i])
	//	{
	//		glPointSize(100.0f);
	//		glColor3f(1, 1, 1);
	//		glVertex3f(a.x, a.y, a.z);
	//	}
	//	else
	//		glColor3f(1, 0, 0);
	//}
	//glEnd();

	//glBegin(GL_LINES);
	//glLineWidth(5.0f);

	//_bendSpring->Draw(h_pos, true);
	//glEnd();
	glEnable(GL_LIGHTING);
}

void PBD_ClothCuda::drawWire(void)
{
	glEnable(GL_LIGHTING);

	glBegin(GL_LINES);
	for (uint i = 0u; i < _numEdges; i++)
	{
		uint ino0 = h_edgeIdx[i].x;
		uint ino1 = h_edgeIdx[i].y;
		REAL3 a = h_pos[ino0];
		REAL3 b = h_pos[ino1];

		glNormal3f(h_vNormal[ino0].x, h_vNormal[ino0].y, h_vNormal[ino0].z);
		glVertex3f(a.x, a.y, a.z);
		glNormal3f(h_vNormal[ino1].x, h_vNormal[ino1].y, h_vNormal[ino1].z);
		glVertex3f(b.x, b.y, b.z);
	}
	glEnd();

	glEnable(GL_LIGHTING);
}

void PBD_ClothCuda::InitDeviceMem(void)
{
	d_flag.resize(_numVertices);			d_flag.memset(0);
	d_gridHash.resize(_numVertices);			d_gridHash.memset(0);
	d_gridIdx.resize(_numVertices);			d_gridIdx.memset(0);
	d_cellStart.resize(_gridRes * _gridRes * _gridRes);			d_cellStart.memset(0);
	d_cellEnd.resize(_gridRes * _gridRes * _gridRes);			d_cellEnd.memset(0);
	d_faceAABB.resize(_numFaces);			d_faceAABB.memset(0);
	d_faceIdx.resize(_numFaces);			d_faceIdx.memset(0);
	d_sortedPos.resize(_numVertices);		d_sortedPos.memset(0);
	d_Pos.resize(_numVertices);				d_Pos.memset(0);
	d_Pos1.resize(_numVertices);			d_Pos1.memset(0);
	d_sortedVel.resize(_numVertices);		d_sortedVel.memset(0);
	d_Vel.resize(_numVertices);				d_Vel.memset(0);
	d_fNormal.resize(_numFaces);			d_fNormal.memset(0);
	d_vNormal.resize(_numVertices);		d_vNormal.memset(0);
	d_InvMass.resize(_numVertices);		d_InvMass.memset(0);
}

void PBD_ClothCuda::FreeDeviceMem(void)
{
	d_flag.free();
	d_gridHash.free();
	d_gridIdx.free();
	d_cellStart.free();
	d_cellEnd.free();
	d_faceAABB.free();
	d_faceIdx.free();
	d_sortedPos.free();
	d_Pos.free();
	d_Pos1.free();
	d_sortedVel.free();
	d_Vel.free();
	d_fNormal.free();
	d_vNormal.free();
	d_InvMass.free();
}

void	PBD_ClothCuda::copyToDevice(void)
{
	d_flag.copyFromHost(h_flag);
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
	d_flag.copyToHost(h_flag);
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
	d_nbFaces.copyFromHost(h_nbVFaces);
	d_nbVertices.copyFromHost(h_nbVertices);
}

void	PBD_ClothCuda::copyNbToHost(void)
{
	d_nbFaces.copyToHost(h_nbVFaces);
	d_nbVertices.copyToHost(h_nbVertices);
}