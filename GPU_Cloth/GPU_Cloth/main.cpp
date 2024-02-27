#include <Windows.h>
#include <iostream>
#include "GL\glut.h"
#include "Engine.h"
#include "PBD_ObjectCloth.h"
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int width = 800;
int height = 800;
float zoom = 15.0f;
float rotx = 0;
float roty = 0.001f;
float tx = 0;
float ty = 0;
int lastx = 0;
int lasty = 0;
unsigned char Buttons[3] = { 0 };
bool simulation = false;

int frame = 0, curTime, timebase = 0;

Engine* _engine;
vector<PBD_ObjectCloth*> _pbd;

#define CUDA 1

void Init(void)
{
	glEnable(GL_DEPTH_TEST);
	if (CUDA)
	{
		_engine = new Engine(make_REAL3(0.0, -9.81, 0.0), 0.01, "OBJ\\dragon.obj", 1);
	}
	else
	{
		for (int i = 0; i < 1; i++)
		{
			_pbd.push_back(new PBD_ObjectCloth("OBJ\\dragon.obj"));
		}
	}
}

void Draw(void)
{
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	if (CUDA)
	{
		_engine->draw();
	}
	else
	{
		for (auto pbd : _pbd)
		{
			pbd->drawSolid();
		}
	}

	glDisable(GL_LIGHTING);
}

void Capture(char* filename, int width, int height)
{
	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;
	unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
	FILE* file;
	fopen_s(&file, filename, "wb");
	if (image != NULL)
	{
		if (file != NULL)
		{
			glReadPixels(0, 0, width, height, 0x80E0, GL_UNSIGNED_BYTE, image);
			memset(&bf, 0, sizeof(bf));
			memset(&bi, 0, sizeof(bi));
			bf.bfType = 'MB';
			bf.bfSize = sizeof(bf) + sizeof(bi) + width * height * 3;
			bf.bfOffBits = sizeof(bf) + sizeof(bi);
			bi.biSize = sizeof(bi);
			bi.biWidth = width;
			bi.biHeight = height;
			bi.biPlanes = 1;
			bi.biBitCount = 24;
			bi.biSizeImage = width * height * 3;
			fwrite(&bf, sizeof(bf), 1, file);
			fwrite(&bi, sizeof(bi), 1, file);
			fwrite(image, sizeof(unsigned char), height * width * 3, file);
			fclose(file);
		}
		free(image);
	}
}


void Update(void)
{
	if (simulation)
	{
		//if (frame == 0 || frame % 4 == 0) {
		//	static int index = 0;
		//	char filename[100];
		//	sprintf(filename, "capture\\capture-%d.bmp", index++);
		//	Capture(filename, width, height);
		//}
		frame++;
		curTime = glutGet(GLUT_ELAPSED_TIME);

		if (curTime - timebase > 1000)
		{
			double fps = frame * 1000.0 / (curTime - timebase);
			timebase = curTime;
			frame = 0;

			printf("FPS : %f\n", fps);
		}

		if (CUDA)
		{
			_engine->simulation();
		}
		else
		{
			for (auto pbd : _pbd)
			{
				pbd->simulation(0.01);
			}
		}
	}
	::glutPostRedisplay();
}

void Display(void)
{
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	/*
	zoom = 5.699999;
	tx = 0.350000;
	ty = 1.500000;
	rotx = 15.500000;
	roty = -45.499001;
	*/

	glTranslatef(0, 0, -zoom);
	glTranslatef(tx, ty, 0);
	glRotatef(rotx, 1, 0, 0);
	glRotatef(roty, 0, 1, 0);

	//printf("%f, %f, %f, %f, %f\n", -zoom, tx, ty, rotx, roty);

	Draw();

	glutSwapBuffers();
}

void Reshape(int w, int h)
{
	if (w == 0)
	{
		h = 1;
	}
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (float)w / h, 0.1, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void Motion(int x, int y)
{
	int diffx = x - lastx;
	int diffy = y - lasty;
	lastx = x;
	lasty = y;

	if (Buttons[2])
	{
		zoom -= (float)0.05f * diffx;
	}
	else if (Buttons[0])
	{
		rotx += (float)0.5f * diffy;
		roty += (float)0.5f * diffx;
	}
	else if (Buttons[1])
	{
		tx += (float)0.05f * diffx;
		ty -= (float)0.05f * diffy;
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y)
{
	lastx = x;
	lasty = y;
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		Buttons[0] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	case GLUT_MIDDLE_BUTTON:
		Buttons[1] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	case GLUT_RIGHT_BUTTON:
		Buttons[2] = ((GLUT_DOWN == state) ? 1 : 0);
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void SpecialInput(int key, int x, int y)
{
	glutPostRedisplay();
}

void Keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'q':
	case 'Q':
		exit(0);
	case 'r':
	case 'R':
		//_engine->reset();
		break;
	case 'f':
	case 'F':
		if (CUDA)
		{
			_engine->ApplyWind(make_REAL3(-0.5, -0.25, -0.25));
		}
		else
		{
			for (auto pbd : _pbd)
			{
				pbd->applyWind(vec3(-0.5, -0.25, -0.25));
			}
		}
		break;
	case ' ':
		simulation = !simulation;
		break;
	}
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Position Based Dynamics");
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Update);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialInput);
	Init();
	glutMainLoop();
}