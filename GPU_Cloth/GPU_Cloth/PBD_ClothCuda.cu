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

	printf("Num of Faces: %d, Num of Vertices: %d\n", _param._numFaces, _param._numVertices);
	printf("Num of Strech: %d, Num of Bend: %d\n", _strechSpring->_param._numConstraint, _bendSpring->_param._numConstraint);
	printf("Num of Color Strech: %d, Num of Color Bend: %d\n", _strechSpring->_param._numColor, _bendSpring->_param._numColor);
}

void PBD_ClothCuda::InitParam(REAL gravity, REAL dt)
{
	_param._gridRes = 128;
	_param._thickness = 4.0 / _param._gridRes;
	_param._selfColliDamping = 0.1f;

	_param._iteration = 10;
	_param._springK = 0.9f;
	_param._linearDamping = 0.99f;

	_param._gravity = gravity;
	_param._subdt = dt / _param._iteration;
	_param._subInvdt = 1.0 / _param._subdt;

	_param._maxsaturation = 20.0;
	_param._absorptionK = 0.05;

	_param._diffusK = 0.05;
	_param._gravityDiffusK = 0.01;
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

void PBD_ClothCuda::WetCloth_Kernel(void)
{
	Absorption_Kernel();
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
	Dvector<REAL> deltaS(_param._numFaces);
	deltaS.memset(0);

	uint numThreads, numBlocks;
	ComputeGridSize(_param._numFaces, 64, numBlocks, numThreads);
	WaterDiffusion_Kernel << <numBlocks, numThreads >> >
		(d_Face, d_Vertex, deltaS());

	ApplyDeltaSaturation_Kernel << <numBlocks, numThreads >> >
		(d_Face, deltaS());
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
		(d_Vertex);
}

void PBD_ClothCuda::draw(void)
{
	glEnable(GL_COLOR_MATERIAL);
	for (uint i = 0u; i < _param._numFaces; i++)
	{
		uint ino0 = h_Mesh->h_faceIdx[i].x;
		uint ino1 = h_Mesh->h_faceIdx[i].y;
		uint ino2 = h_Mesh->h_faceIdx[i].z;
		REAL3 a = h_Mesh->h_pos[ino0];
		REAL3 b = h_Mesh->h_pos[ino1];
		REAL3 c = h_Mesh->h_pos[ino2];

		glBegin(GL_POLYGON);

		REAL ratio = h_Mesh->h_fSaturation[i] / _param._maxsaturation;
		REAL3 color = ScalarToColor(ratio);
		glColor3f(color.x, color.y, color.z);

		//REAL ratio = h_Mesh->h_fSaturation[i] / _param._maxsaturation;
		//REAL color = 1.0 - ratio;
		//glColor3f(color, color, color);

		glNormal3f(h_Mesh->h_vNormal[ino0].x, h_Mesh->h_vNormal[ino0].y, h_Mesh->h_vNormal[ino0].z);
		glVertex3f(a.x, a.y, a.z);
		glNormal3f(h_Mesh->h_vNormal[ino1].x, h_Mesh->h_vNormal[ino1].y, h_Mesh->h_vNormal[ino1].z);
		glVertex3f(b.x, b.y, b.z);
		glNormal3f(h_Mesh->h_vNormal[ino2].x, h_Mesh->h_vNormal[ino2].y, h_Mesh->h_vNormal[ino2].z);
		glVertex3f(c.x, c.y, c.z);

		glEnd();
	}

	glTranslatef(0.5f, 0.5f, 0.5f);
	glColor3f(1, 1, 1);
	glutSolidSphere(0.28f, 50, 50);

	glEnable(GL_LIGHTING);
}

void PBD_ClothCuda::drawWire(void)
{
	glEnable(GL_LIGHTING);

	glBegin(GL_LINES);
	for (uint i = 0u; i < _param._numEdges; i++)
	{
		uint ino0 = h_Mesh->h_edgeIdx[i].x;
		uint ino1 = h_Mesh->h_edgeIdx[i].y;
		REAL3 a = h_Mesh->h_pos[ino0];
		REAL3 b = h_Mesh->h_pos[ino1];

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
	d_Hash.InitDeviceMem(_param._numVertices);
}

void PBD_ClothCuda::FreeDeviceMem(void)
{
	d_Vertex.FreeDeviceMem();
	d_Face.FreeDeviceMem();
	d_Hash.FreeDeviceMem();
}

void PBD_ClothCuda::copyToDevice(void)
{
	d_Vertex.copyToDevice(*h_Mesh);
	d_Face.copyToDevice(*h_Mesh);
}

void PBD_ClothCuda::copyToHost(void)
{
	d_Vertex.copyToHost(*h_Mesh);
	d_Face.copyToHost(*h_Mesh);
}