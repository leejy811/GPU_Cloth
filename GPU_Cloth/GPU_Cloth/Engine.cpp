#include "Engine.h"

void Engine::init(REAL3& gravity, REAL dt, char* filename, uint num)
{
	_gravity = gravity;
	_dt = dt;
	_invdt = 1.0 / dt;

	_boundary._min = make_REAL3(-1.5);
	_boundary._max = make_REAL3(1.5);

	for (int i = 0; i < num; i++)
	{
		_cloths.push_back(new PBD_ClothCuda(filename, 20, 0.99, 0.7));
	}
	_frame = 0u;
}

void	Engine::simulation(void)
{
	for (auto cloth : _cloths)
	{
		cloth->ComputeFaceNormal_kernel();
		cloth->ComputeVertexNormal_kernel();
		cloth->ComputeExternalForce_kernel(_gravity, _dt);
		cloth->ProjectConstraint_kernel();
		cloth->Intergrate_kernel(_invdt);

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