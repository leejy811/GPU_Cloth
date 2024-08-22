#include "Engine.h"

void Engine::init(REAL gravity, REAL dt, char* filename, uint num)
{
	_gravity = gravity;
	_dt = dt;
	_invdt = 1.0 / dt;

	_boundary._min = make_REAL3(0.0);
	_boundary._max = make_REAL3(1.0);

	for (int i = 0; i < num; i++)
	{
		_cloths.push_back(new PBD_ClothCuda(filename, _gravity, _dt));
	}

	_cloths.push_back(new PBD_ClothCuda("OBJ\\geoSphere.obj", _gravity, _dt));
	_frame = 0u;

	createFBO();
}

void	Engine::simulation(void)
{
	for (auto cloth : _cloths)
	{
		cloth->Simulation();
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

void Engine::drawBO(const Camera& camera)
{
		glViewport(0, 0, shadowWidth, shadowHeight);
		glBindFramebuffer(GL_FRAMEBUFFER, depthMapFbo);
		glClear(GL_DEPTH_BUFFER_BIT);

		glCullFace(GL_FRONT);
		for (auto cloth : _cloths)
		{
			cloth->_clothRenderer->UpdateShader(*cloth->_clothRenderer->_depthShader, camera);
			cloth->_clothRenderer->RenderScene(cloth->_param._numFaces);
		}
		glCullFace(GL_BACK);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glViewport(0, 0, 800, 800);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glBindTexture(GL_TEXTURE_2D, depthMapTexture);

		for (auto cloth : _cloths)
		{
			cloth->_clothRenderer->UpdateShader(*cloth->_clothRenderer->_shader, camera);
			cloth->_clothRenderer->RenderScene(cloth->_param._numFaces);
		}
}

void Engine::drawWire(void)
{
	for (auto cloth : _cloths)
		cloth->drawWire();
}