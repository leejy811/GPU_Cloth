#include "Mesh.h"

Mesh::Mesh()
{

}

Mesh::~Mesh()
{

}

void Mesh::LoadObj(char* filename, AABB boundary)
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
					boundary._min = boundary._max = x;
					flag = false;
				}
				else addAABB(boundary, x);
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

	moveCenter(0.5, boundary);
	SetMass();
	buildAdjacency();
	computeNormal();
	h_pos1.resize(h_pos.size(), make_REAL3(0.0, 0.0, 0.0));
	h_vel.resize(h_pos.size(), make_REAL3(0.0, 0.0, 0.0));
	h_fSaturation.resize(h_faceIdx.size(), 0.0);
	h_cotWeight.resize(h_edgeIdx.size(), 0.0);
	h_vAngle.resize(h_pos.size(), 0.0);

	REAL3 maxPos = make_REAL3(-100.0, -100.0, -100.0);
	REAL3 minPos = make_REAL3(100.0, 100.0, 100.0);

	for (int i = 0; i < h_pos.size(); i++)
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
}

void Mesh::moveCenter(REAL scale, AABB boundary)
{
	REAL3 size = boundary._max - boundary._min;
	REAL max_length = size.x;
	if (max_length < size.y)
		max_length = size.y;
	if (max_length < size.z)
		max_length = size.z;
	max_length = 2.0 * scale / max_length;

	REAL3 prevCenter = (boundary._min + boundary._max) * (REAL)0.5;

	bool flag = false;
	uint vlen = h_pos.size();
	for (uint i = 0u; i < vlen; i++)
	{
		REAL3 pos = h_pos[i];
		REAL3 grad = pos - prevCenter;
		grad *= max_length;
		pos = grad;
		h_pos[i] = pos;
		if (flag) addAABB(boundary, pos);
		else
		{
			boundary._min = boundary._max = pos;
			flag = true;
		}
	}

	for (int i = 0; i < h_pos.size(); i++)
	{
		h_pos[i] += make_REAL3(0.5 , 0.5, 1.0);

		REAL tmp = h_pos[i].y;
		h_pos[i].y = h_pos[i].z;
		h_pos[i].z = tmp;

		tmp = h_pos[i].x;
		h_pos[i].x = h_pos[i].z;
		h_pos[i].z = tmp;
	}
}

void Mesh::SetMass(void)
{
	for (int i = 0; i < h_pos.size(); i++)
	{
		h_invMass.push_back(1.0);
	}
}

void Mesh::buildAdjacency(void)
{
	buildAdjacency_VF();
	buildEdges();
	buildAdjacency_EF();
	buildAdjacency_FF();
}

void Mesh::buildAdjacency_VF(void)
{
	//Neighbor
	vector<set<uint>> nbFs(h_pos.size());
	vector<set<uint>> nbVs(h_pos.size());

	for (uint i = 0u; i < h_faceIdx.size(); i++)
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

void Mesh::buildEdges(void)
{
	vector<bool> flags(h_pos.size(), false);

	for (int i = 0; i < h_pos.size(); i++)
	{
		for (int j = h_nbVertices._index[i]; j < h_nbVertices._index[i + 1]; j++)
		{
			if (!flags[h_nbVertices._array[j]])
				h_edgeIdx.push_back(make_uint2(i, h_nbVertices._array[j]));
		}
		flags[i] = true;
	}
}

void Mesh::buildAdjacency_EF(void)
{
	vector<set<uint>> nbEF(h_edgeIdx.size());

	for (int i = 0; i < h_edgeIdx.size(); i++)
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

void Mesh::buildAdjacency_FF(void)
{
	vector<set<uint>> nbFF(h_faceIdx.size());

	for (int i = 0; i < h_edgeIdx.size(); i++)
	{
		uint numNbEF = h_nbEFaces._index[i + 1] - h_nbEFaces._index[i];
		if (numNbEF == 2)
		{
			nbFF[h_nbEFaces._array[h_nbEFaces._index[i]]].insert(h_nbEFaces._array[h_nbEFaces._index[i] + 1]);
			nbFF[h_nbEFaces._array[h_nbEFaces._index[i] + 1]].insert(h_nbEFaces._array[h_nbEFaces._index[i]]);
		}
	}

	h_nbFFaces.init(nbFF);
}

void Mesh::computeNormal(void)
{
	h_fNormal.resize(h_faceIdx.size());
	h_vNormal.clear();
	h_vNormal.resize(h_pos.size(), make_REAL3(0.0, 0.0, 0.0));

	for (uint i = 0u; i < h_faceIdx.size(); i++)
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

	for (uint i = 0u; i < h_pos.size(); i++)
	{
		Normalize(h_vNormal[i]);
	}
}