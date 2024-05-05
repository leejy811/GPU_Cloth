#include "Engine.h"

void Engine::init(REAL3& gravity, REAL dt, char* filename, uint num)
{
	_gravity = gravity;
	_dt = dt;
	_invdt = 1.0 / dt;

	_boundary._min = make_REAL3(0.0);
	_boundary._max = make_REAL3(1.0);

	for (int i = 0; i < num; i++)
	{
		_cloths.push_back(new PBD_ClothCuda(filename, 10, 0.99, 0.9, 128, 0.04, 0.1));
	}
	_frame = 0u;
}

void	Engine::simulation(void)
{
	for (auto cloth : _cloths)
	{
		double subDt = _dt / cloth->_iteration;
		double subInvDt = 1.0 / subDt;

		for (int i = 0; i < cloth->_iteration; i++)
		{
			cloth->ComputeFaceNormal_kernel();
			cloth->ComputeVertexNormal_kernel();
			cloth->ComputeExternalForce_kernel(_gravity, subDt);
			cloth->ProjectConstraint_kernel();
			cloth->SetHashTable_kernel();
			cloth->UpdateFaceAABB_Kernel();
			cloth->Colide_kernel();
			cloth->Intergrate_kernel(subInvDt);
			cloth->LevelSetCollision_kernel();
		}
		cloth->copyToHost();
	}
}

void	Engine::reset(void) 
{

}

void	Engine::ApplyWind(REAL3 wind)
{
	for (auto cloth : _cloths)
		cloth->ComputeWind_kernel(wind);
}


void Engine::draw(void)
{
	for (auto cloth : _cloths)
		cloth->draw();
}

void Engine::drawWire(void)
{
	for (auto cloth : _cloths)
		cloth->drawWire();
}