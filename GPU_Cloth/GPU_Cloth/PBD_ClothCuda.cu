#include "PBD_ClothCuda.cuh"

PBD_ClothCuda::PBD_ClothCuda()
{

}

PBD_ClothCuda::~PBD_ClothCuda()
{
	FreeDeviceMem();
}

void PBD_ClothCuda::Init(char* filename)
{
	_strechSpring = new Constraint(_param._iteration, _param._springK);
	_bendSpring = new Constraint(_param._iteration, _param._springK);
	h_Mesh = new Mesh(filename, _boundary);

	_param._numFaces = h_Mesh->h_faceIdx.size();
	_param._numVertices = h_Mesh->h_pos.size();
	_param._numEdges = h_Mesh->h_edgeIdx.size();
	buildConstraint();
	cudaMemcpyToSymbol(clothParam, &_param, sizeof(ClothParam));
	d_Hash.InitParam(_param);

	_strechSpring->InitDeviceMem();
	_strechSpring->copyToDevice();

	_bendSpring->InitDeviceMem();
	_bendSpring->copyToDevice();

	InitDeviceMem();
	copyToDevice();

	//InitSaturation << <divup(_param._numFaces, BLOCK_SIZE), BLOCK_SIZE >> >
	//	(d_Face, d_Vertex);

	_clothRenderer = new ClothRenderer(_param._numVertices, _param._numFaces);

	printf("Num of Faces: %d, Num of Vertices: %d\n", _param._numFaces, _param._numVertices);
	printf("Num of Strech: %d, Num of Bend: %d\n", _strechSpring->_param._numConstraint, _bendSpring->_param._numConstraint);
	printf("Num of Color Strech: %d, Num of Color Bend: %d\n", _strechSpring->_param._numColor, _bendSpring->_param._numColor);
}

void PBD_ClothCuda::InitParam(REAL gravity, REAL dt)
{
	_param._gridRes = 128;
	_param._thickness = 4.0 / _param._gridRes;
	_param._selfColliDamping = 0.05f;

	_param._iteration = 10;
	_param._springK = 0.9f;
	_param._linearDamping = 0.99f;

	_param._gravity = gravity;
	_param._subdt = dt / _param._iteration;
	//_param._subdt = dt;
	_param._subInvdt = 1.0 / _param._subdt;

	_param._maxsaturation = 200.0;
	_param._absorptionK = 0.001;

	_param._diffusK = 0.08;
	_param._gravityDiffusK = 0.08;

	_param._surfTension = 75;
	_param._adhesionThickness = 8.0 / _param._gridRes;
}

void PBD_ClothCuda::buildConstraint(void)
{
	vector<set<uint>> stretchEdges(_param._numVertices);
	vector<set<uint>> bendEdges(_param._numVertices);

	for (int i = 0; i < _param._numEdges; i++)
	{
		//strech spring
		_strechSpring->h_EdgeIdx.push_back(h_Mesh->h_edgeIdx[i]);
		_strechSpring->h_RestLength.push_back(Length(h_Mesh->h_pos[h_Mesh->h_edgeIdx[i].x] - h_Mesh->h_pos[h_Mesh->h_edgeIdx[i].y]));
		_strechSpring->_param._numConstraint++;
		stretchEdges[h_Mesh->h_edgeIdx[i].x].insert(i);
		stretchEdges[h_Mesh->h_edgeIdx[i].y].insert(i);

		//bend spring
		if (h_Mesh->h_nbEFaces._index[i + 1] - h_Mesh->h_nbEFaces._index[i] != 2)
			continue;

		uint edgeSum = h_Mesh->h_edgeIdx[i].x + h_Mesh->h_edgeIdx[i].y;
		uint fId0 = h_Mesh->h_nbEFaces._array[h_Mesh->h_nbEFaces._index[i]];
		uint fId1 = h_Mesh->h_nbEFaces._array[h_Mesh->h_nbEFaces._index[i] + 1];

		uint vId0 = h_Mesh->h_faceIdx[fId0].x + h_Mesh->h_faceIdx[fId0].y + h_Mesh->h_faceIdx[fId0].z - edgeSum;
		uint vId1 = h_Mesh->h_faceIdx[fId1].x + h_Mesh->h_faceIdx[fId1].y + h_Mesh->h_faceIdx[fId1].z - edgeSum;

		_bendSpring->h_EdgeIdx.push_back(make_uint2(vId0, vId1));
		_bendSpring->h_RestLength.push_back(Length(h_Mesh->h_pos[vId0] - h_Mesh->h_pos[vId1]));

		bendEdges[vId0].insert(_bendSpring->_param._numConstraint);
		bendEdges[vId1].insert(_bendSpring->_param._numConstraint);

		stretchEdges[h_Mesh->h_edgeIdx[i].x].insert(i);
		stretchEdges[h_Mesh->h_edgeIdx[i].y].insert(i);

		_bendSpring->_param._numConstraint++;
	}

	_strechSpring->h_nbCEdges.init(stretchEdges);
	_bendSpring->h_nbCEdges.init(bendEdges);

	_strechSpring->Init(_param._numVertices);
	_bendSpring->Init(_param._numVertices);

	h_Mesh->h_restAngle.resize(_param._numEdges);
	ComputeRestAngle();
}

void PBD_ClothCuda::ComputeRestAngle(void)
{
	for (int i = 0; i < _param._numVertices; i++)
	{
		h_Mesh->h_vAngle[i] = 0.0;
	}

	for (int i = 0; i < _param._numEdges; i++)
	{
		if (h_Mesh->h_nbEFaces._index[i + 1] - h_Mesh->h_nbEFaces._index[i] != 2)
			continue;

		uint edgeSum = h_Mesh->h_edgeIdx[i].x + h_Mesh->h_edgeIdx[i].y;
		uint fId0 = h_Mesh->h_nbEFaces._array[h_Mesh->h_nbEFaces._index[i]];
		uint fId1 = h_Mesh->h_nbEFaces._array[h_Mesh->h_nbEFaces._index[i] + 1];

		uint vId0 = h_Mesh->h_faceIdx[fId0].x + h_Mesh->h_faceIdx[fId0].y + h_Mesh->h_faceIdx[fId0].z - edgeSum;
		uint vId1 = h_Mesh->h_faceIdx[fId1].x + h_Mesh->h_faceIdx[fId1].y + h_Mesh->h_faceIdx[fId1].z - edgeSum;
		uint vId2 = h_Mesh->h_edgeIdx[i].x;
		uint vId3 = h_Mesh->h_edgeIdx[i].y;

		REAL3 p0 = h_Mesh->h_pos[vId0];
		REAL3 p1 = h_Mesh->h_pos[vId1];
		REAL3 p2 = h_Mesh->h_pos[vId2];
		REAL3 p3 = h_Mesh->h_pos[vId3];

		REAL3 e = p3 - p2;
		REAL length = Length(e);
		if (length < 1e-6)
		{
			return;
		}

		REAL invlength = 1.0 / length;
		REAL3 n1 = Cross(p2 - p0, p3 - p0);
		REAL3 n2 = Cross(p3 - p1, p2 - p1);
		n1 /= LengthSquared(n1);
		n2 /= LengthSquared(n2);

		REAL3 d0 = n1 * length;
		REAL3 d1 = n2 * length;
		REAL3 d2 = n1 * (Dot(p0 - p3, e) * invlength) + n2 * (Dot(p1 - p3, e) * invlength);
		REAL3 d3 = n1 * (Dot(p2 - p0, e) * invlength) + n2 * (Dot(p2 - p1, e) * invlength);

		Normalize(n1);
		Normalize(n2);
		REAL dot = Dot(n1, n2);

		if (dot < -1.0)
			dot = -1.0;
		if (dot > 1.0)
			dot = 1.0;

		REAL restAngle = acos(dot);
		h_Mesh->h_restAngle[i] = restAngle;

		h_Mesh->h_vAngle[vId2] += restAngle;
		h_Mesh->h_vAngle[vId3] += restAngle;
	}

	for (int i = 0; i < _param._numVertices; i++)
	{
		uint numNb = h_Mesh->h_nbVertices._index[i + 1] - h_Mesh->h_nbVertices._index[i];
		if (numNb == 0)
			continue;
		h_Mesh->h_vAngle[i] /= numNb;
	}
}

void PBD_ClothCuda::ComputeLaplacian(void)
{
	REAL lambda = 2.0;
	vector<REAL3> pos1;
	for (int i = 0; i < _param._numVertices; i++)
	{
		REAL3 pos = h_Mesh->h_pos[i];
		REAL3 sumPos = make_REAL3(0.0, 0.0, 0.0);

		uint fid0 = h_Mesh->h_nbVFaces._index[i];
		uint fid1 = h_Mesh->h_nbVFaces._index[i + 1];
		uint numnbF = fid1 - fid0;
		REAL satSum = 0.0f;
		for (int j = 0; j < fid1 - fid0; j++)
		{
			uint fIdx = h_Mesh->h_nbVFaces._array[h_Mesh->h_nbVFaces._index[i] + j];
			satSum += h_Mesh->h_fSaturation[fIdx];
		}

		if (fid1 - fid0 == 0)
			continue;

		REAL weight = h_Mesh->h_vAngle[i] * (satSum / (_param._maxsaturation * numnbF));
		uint id0 = h_Mesh->h_nbVertices._index[i];
		uint id1 = h_Mesh->h_nbVertices._index[i + 1];
		uint numnbV = id1 - id0;
		for (int j = 0;j < numnbV;j++)
		{
			uint vIdx = h_Mesh->h_nbVertices._array[h_Mesh->h_nbVertices._index[i] + j];
			sumPos += h_Mesh->h_pos[vIdx];
		}
		REAL valence = id1 - id0;
		if (valence == 0)
			continue;
		REAL3 finalPos = (sumPos / valence - pos) * -1 * lambda * weight + pos;

		if (h_Mesh->h_pos[i].x > 0.89)
		{
			pos1.push_back(pos);
			continue;
		}
		pos1.push_back(finalPos);
	}

	h_Mesh->h_pos = pos1;
}

void PBD_ClothCuda::Simulation()
{
	for (int i = 0; i < _param._iteration; i++)
	{
		ComputeFaceNormal_kernel();
		ComputeVertexNormal_kernel();
		ComputeGravity_kernel();
		ProjectConstraint_kernel();
		SetHashTable_kernel();
		UpdateFaceAABB_Kernel();
		Colide_kernel();
		//Intergrate_kernel();
		//LevelSetCollision_kernel();
		WetCloth_Kernel();
		//ComputeWrinkCloth_kernel();
	}

	_clothRenderer->MappingBO(d_Vertex.d_Pos, d_Vertex.d_vNormal, d_Face.d_faceIdx, _param._numVertices, _param._numFaces);

	//copyToHost();

	//ComputeFaceNormal_kernel();
	//ComputeVertexNormal_kernel();
	//ComputeGravity_kernel();
	//for (int i = 0; i < _param._iteration; i++)
	//{
	//	ProjectConstraint_kernel();
	//}
	//Intergrate_kernel();
	//copyToHost();
}

void PBD_ClothCuda::ComputeWrinkCloth_kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_param._numEdges, 64, numBlocks, numThreads);

	cudaMemset(d_Vertex.d_vAngle, 0, sizeof(REAL) * _param._numVertices);
	ComputeAngle_kernel << <numBlocks, numThreads >> >
		(d_Edge, d_Vertex, d_Face);

	ComputeGridSize(_param._numVertices, 64, numBlocks, numThreads);
	ComputeLaplacian_kernel << <numBlocks, numThreads >> >
		(d_Vertex, d_Face);
}

void PBD_ClothCuda::ComputeGravity_kernel()
{
	CompGravity_kernel << <divup(_param._numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Vertex);
}

void PBD_ClothCuda::Intergrate_kernel()
{
	CompIntergrate_kernel << <divup(_param._numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Vertex);
}

void PBD_ClothCuda::ComputeFaceNormal_kernel(void)
{
	CompFaceNorm_kernel << <divup(_param._numFaces, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Face, d_Vertex);
}

void PBD_ClothCuda::ComputeVertexNormal_kernel(void)
{
	CompVertexNorm_kernel << <divup(_param._numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Face, d_Vertex);
}

void PBD_ClothCuda::ComputeWind_kernel(REAL3 wind)
{
	CompWind_kernel << <divup(_param._numFaces, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Face, d_Vertex, wind);
}

void PBD_ClothCuda::ProjectConstraint_kernel(void)
{
	_strechSpring->IterateConstraint(d_Vertex.d_Pos1, d_Vertex.d_InvMass, d_Vertex.d_SatMass);
	_bendSpring->IterateConstraint(d_Vertex.d_Pos1, d_Vertex.d_InvMass, d_Vertex.d_SatMass);
	//AngleConstraint_kernel();
}

void PBD_ClothCuda::AngleConstraint_kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_param._numEdges, 64, numBlocks, numThreads);

	Dvector<REAL> deltaPos(_param._numVertices * 3u);	deltaPos.memset(0);
	SolveAngleConstraint_kernel << <numBlocks, numThreads >> >
		(d_Edge, d_Vertex, d_Face, deltaPos());

	ComputeGridSize(_param._numVertices, 64, numBlocks, numThreads);
	ApplyConstraintDeltaPos_kernel << <numBlocks, numThreads >> >
		(d_Vertex, deltaPos());
}

void PBD_ClothCuda::SetHashTable_kernel(void)
{
	d_Hash.SetHashTable_kernel(d_Vertex.d_Pos1);
}

void PBD_ClothCuda::UpdateFaceAABB_Kernel(void)
{
	uint numThreads, numBlocks;
	ComputeGridSize(_param._numFaces, 256, numBlocks, numThreads);

	UpdateFaceAABB << <numBlocks, numThreads >> >
		(d_Face, d_Vertex);
}

void PBD_ClothCuda::Colide_kernel(void)
{
	//SelfCollision_kernel();
	//AdhesionForce_kernel();
}

void PBD_ClothCuda::SelfCollision_kernel(void)
{
	uint numThreads, numBlocks;

	Dvector<REAL> impulse(_param._numVertices * 3u);
	impulse.memset(0);

	ComputeGridSize(_param._numVertices, 64, numBlocks, numThreads);
	Colide_PP << <numBlocks, numThreads >> >
		(d_Vertex, d_Hash, impulse());

	ComputeGridSize(_param._numFaces, 64, numBlocks, numThreads);
	Colide_PT << <numBlocks, numThreads >> >
		(d_Face, d_Vertex, d_Hash, impulse());

	ComputeGridSize(_param._numVertices, 64, numBlocks, numThreads);
	ApplyImpulse_kernel << <numBlocks, numThreads >> >
		(d_Vertex, impulse());
}

void PBD_ClothCuda::AdhesionForce_kernel(void)
{
	uint numThreads, numBlocks;

	Dvector<REAL> adhesion(_param._numVertices * 3u);
	adhesion.memset(0);

	ComputeGridSize(_param._numVertices, 64, numBlocks, numThreads);
	UpdateAdhesionForce_kernel << <numBlocks, numThreads >> >
		(d_Vertex, d_Hash, adhesion());

	ComputeGridSize(_param._numVertices, 64, numBlocks, numThreads);
	ApplyAdhesion_kernel << <numBlocks, numThreads >> >
		(d_Vertex, adhesion());
}

void PBD_ClothCuda::WetCloth_Kernel(void)
{
	//Absorption_Kernel();
	Diffusion_Kernel();
	Dripping_Kernel();
	UpdateMass_Kernel();
}

void PBD_ClothCuda::Absorption_Kernel(void)
{
	WaterAbsorption_Kernel << <divup(_param._numFaces, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Face, d_Vertex);
}

void PBD_ClothCuda::Diffusion_Kernel(void)
{
	Dvector<REAL> deltaS(_param._numFaces);		deltaS.memset(0);
	Dvector<REAL> deltaD(_param._numFaces);		deltaD.memset(0);

	uint numThreads, numBlocks;
	ComputeGridSize(_param._numFaces, 64, numBlocks, numThreads);
	WaterDiffusion_Kernel << <numBlocks, numThreads >> >
		(d_Face, d_Vertex, deltaS(), deltaD());

	ApplyDeltaSaturation_Kernel << <numBlocks, numThreads >> >
		(d_Face, deltaS(), deltaD());
}

void PBD_ClothCuda::Dripping_Kernel(void)
{
	WaterDripping_Kernel << <divup(_param._numFaces, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Face, d_Vertex);
}

void PBD_ClothCuda::UpdateMass_Kernel(void)
{
	UpdateVertexMass_Kernel << <divup(_param._numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Face, d_Vertex);
}

void PBD_ClothCuda::LevelSetCollision_kernel(void)
{
	LevelSetCollision_D << <divup(_param._numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Vertex, d_Face, true);

	LevelSetCollision_D << <divup(_param._numVertices, BLOCK_SIZE), BLOCK_SIZE >> >
		(d_Vertex, d_Face, false);
}

void PBD_ClothCuda::draw(void)
{
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	for (uint i = 0u; i < _param._numFaces; i++)
	{
		uint ino0 = h_Mesh->h_faceIdx[i].x;
		uint ino1 = h_Mesh->h_faceIdx[i].y;
		uint ino2 = h_Mesh->h_faceIdx[i].z;
		REAL3 a = h_Mesh->h_pos[ino0];
		REAL3 b = h_Mesh->h_pos[ino1];
		REAL3 c = h_Mesh->h_pos[ino2];

		glBegin(GL_POLYGON);

		//REAL ratio = h_Mesh->h_fSaturation[i] / _param._maxsaturation;
		//REAL3 color = ScalarToColor(ratio);
		//glColor3f(color.x, color.y, color.z);

		//REAL ratio = h_Mesh->h_fSaturation[i] / _param._maxsaturation;
		//REAL color = 1.0 - ratio;
		//glColor3f(color, color, color);

		glColor3f(1.0f, 1.0f, 1.0f);

		//REAL color = (h_Mesh->h_vAngle[ino0] + h_Mesh->h_vAngle[ino1] + h_Mesh->h_vAngle[ino2]) / 3;
		//glColor3f(color, color, color);

		glNormal3f(h_Mesh->h_vNormal[ino0].x, h_Mesh->h_vNormal[ino0].y, h_Mesh->h_vNormal[ino0].z);
		glVertex3f(a.x, a.y, a.z);
		glNormal3f(h_Mesh->h_vNormal[ino1].x, h_Mesh->h_vNormal[ino1].y, h_Mesh->h_vNormal[ino1].z);
		glVertex3f(b.x, b.y, b.z);
		glNormal3f(h_Mesh->h_vNormal[ino2].x, h_Mesh->h_vNormal[ino2].y, h_Mesh->h_vNormal[ino2].z);
		glVertex3f(c.x, c.y, c.z);

		glEnd();
	}

	glCullFace(GL_FRONT);
	for (uint i = 0u; i < _param._numFaces; i++)
	{
		uint ino0 = h_Mesh->h_faceIdx[i].x;
		uint ino1 = h_Mesh->h_faceIdx[i].y;
		uint ino2 = h_Mesh->h_faceIdx[i].z;
		REAL3 a = h_Mesh->h_pos[ino0];
		REAL3 b = h_Mesh->h_pos[ino1];
		REAL3 c = h_Mesh->h_pos[ino2];

		glBegin(GL_POLYGON);

		//REAL ratio = h_Mesh->h_fSaturation[i] / _param._maxsaturation;
		//REAL3 color = ScalarToColor(ratio);
		//glColor3f(color.x, color.y, color.z);

		//REAL ratio = h_Mesh->h_fSaturation[i] / _param._maxsaturation;
		//REAL color = 1.0 - ratio;
		//glColor3f(color, color, color);

		glColor3f(1.0f, 1.0f, 1.0f);

		//REAL color = (h_Mesh->h_vAngle[ino0] + h_Mesh->h_vAngle[ino1] + h_Mesh->h_vAngle[ino2]) / 3;
		//glColor3f(color, color, color);

		glNormal3f(h_Mesh->h_vNormal[ino0].x * -1, h_Mesh->h_vNormal[ino0].y * -1, h_Mesh->h_vNormal[ino0].z * -1);
		glVertex3f(a.x, a.y, a.z);
		glNormal3f(h_Mesh->h_vNormal[ino1].x * -1, h_Mesh->h_vNormal[ino1].y * -1, h_Mesh->h_vNormal[ino1].z * -1);
		glVertex3f(b.x, b.y, b.z);
		glNormal3f(h_Mesh->h_vNormal[ino2].x * -1, h_Mesh->h_vNormal[ino2].y * -1, h_Mesh->h_vNormal[ino2].z * -1);
		glVertex3f(c.x, c.y, c.z);

		glEnd();
	}

	//glTranslatef(0.5f, 0.5f, 0.5f);
	//glColor3f(1, 1, 1);
	//glutSolidSphere(0.08f, 50, 50);

	glEnable(GL_LIGHTING);
}

void PBD_ClothCuda::drawBO(const Camera& camera)
{
	_clothRenderer->DrawBO(_param._numFaces, camera);
}

void PBD_ClothCuda::drawWire(void)
{
	glEnable(GL_LIGHTING);
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);

	for (uint i = 0u; i < _param._numEdges; i++)
	{
		uint ino0 = h_Mesh->h_edgeIdx[i].x;
		uint ino1 = h_Mesh->h_edgeIdx[i].y;
		REAL3 a = h_Mesh->h_pos[ino0];
		REAL3 b = h_Mesh->h_pos[ino1];

		uint fId0 = h_Mesh->h_nbEFaces._array[h_Mesh->h_nbEFaces._index[i]];
		uint fId1 = h_Mesh->h_nbEFaces._array[h_Mesh->h_nbEFaces._index[i] + 1];

		REAL satSum = h_Mesh->h_fSaturation[fId0] + h_Mesh->h_fSaturation[fId1];
		REAL color = ((h_Mesh->h_restAngle[i] * 0.5) + 0.5) * (satSum / (_param._maxsaturation * 2));
		glColor3f(color, color, color);

		glNormal3f(h_Mesh->h_vNormal[ino0].x, h_Mesh->h_vNormal[ino0].y, h_Mesh->h_vNormal[ino0].z);
		glVertex3f(a.x, a.y, a.z);
		glNormal3f(h_Mesh->h_vNormal[ino1].x, h_Mesh->h_vNormal[ino1].y, h_Mesh->h_vNormal[ino1].z);
		glVertex3f(b.x, b.y, b.z);
	}
	glEnd();

	glEnable(GL_LIGHTING);
}

REAL3 PBD_ClothCuda::ScalarToColor(REAL val)
{
	REAL fColorMap[5][3] = { { 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };   //Red->Blue
	REAL v = val;
	if (val > 1.0) v = 1.0; if (val < 0.0) v = 0.0; v *= 4.0;
	int low = (int)floor(v), high = (int)ceil(v);
	REAL t = v - low;
	REAL x = (fColorMap[low][0]) * (1 - t) + (fColorMap[high][0]) * t;
	REAL y = (fColorMap[low][1]) * (1 - t) + (fColorMap[high][1]) * t;
	REAL z = (fColorMap[low][2]) * (1 - t) + (fColorMap[high][2]) * t;
	REAL3 color = make_REAL3(x, y, z);
	return color;
}


void PBD_ClothCuda::InitDeviceMem(void)
{
	d_Vertex.InitDeviceMem(_param._numVertices);
	d_Face.InitDeviceMem(_param._numFaces);
	d_Edge.InitDeviceMem(_param._numEdges);
	d_Hash.InitDeviceMem(_param._numVertices);
}

void PBD_ClothCuda::FreeDeviceMem(void)
{
	d_Vertex.FreeDeviceMem();
	d_Face.FreeDeviceMem();
	d_Edge.FreeDeviceMem();
	d_Hash.FreeDeviceMem();
}

void PBD_ClothCuda::copyToDevice(void)
{
	d_Vertex.copyToDevice(*h_Mesh);
	d_Face.copyToDevice(*h_Mesh);
	d_Edge.copyToDevice(*h_Mesh);
}

void PBD_ClothCuda::copyToHost(void)
{
	d_Vertex.copyToHost(*h_Mesh);
	d_Face.copyToHost(*h_Mesh);
}