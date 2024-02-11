#include "Engine.h"

void Engine::init(REAL3& gravity, REAL dt, char* filename)
{
	_gravity = gravity;
	_dt = dt;
	_invdt = 1.0 / dt;

	_boundary._min = make_REAL3(-1.5);
	_boundary._max = make_REAL3(1.5);

	_cloths = new PBD_ClothCuda(filename, 5, 0.99, 0.9);
	_frame = 0u;
}

void	Engine::simulation(void)
{
	_cloths->computeNormal();
	_cloths->copyToDevice();

	_cloths->ComputeGravityForce_kernel(_gravity, _dt);
	_cloths->Intergrate_kernel(_invdt);

	_cloths->copyToHost();
}

void	Engine::reset(void)
{

}

void Engine::draw(void)
{
	_cloths->draw();
}