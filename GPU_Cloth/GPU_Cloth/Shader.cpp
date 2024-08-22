

#include <iostream>
#include <fstream>
#include "Shader.h"

using namespace std;

static GLuint CreateShader(const string& text, GLenum shaderType);
static string LoadShader(const string& fileName);
static void CheckShaderError(GLuint shader, GLuint flag, bool isProgram, const string& errorMessage);


Shader::Shader(const string& fileName)
{
	_program = glCreateProgram();

	_shaders[0] = CreateShader(LoadShader(fileName + ".vs"), GL_VERTEX_SHADER);
	_shaders[1] = CreateShader(LoadShader(fileName + ".fs"), GL_FRAGMENT_SHADER);

	for (unsigned int i = 0; i < NUM_SHADER; i++)
		glAttachShader(_program, _shaders[i]);

	glBindAttribLocation(_program, 0, "position");
	glBindAttribLocation(_program, 1, "texCoord");
	glBindAttribLocation(_program, 2, "normal");

	glLinkProgram(_program);
	CheckShaderError(_program, GL_LINK_STATUS, true, "Error: Program linking failed: ");

	glValidateProgram(_program);
	CheckShaderError(_program, GL_VALIDATE_STATUS, true, "Error: Program is invalid: ");

	_uniforms[ModelMatrix] = glGetUniformLocation(_program, "ModelMatrix");
	_uniforms[ViewMatrix] = glGetUniformLocation(_program, "ViewMatrix");
	_uniforms[ProjectionMatrix] = glGetUniformLocation(_program, "ProjectionMatrix");
	_uniforms[CameraPos] = glGetUniformLocation(_program, "CameraPosition");
	_uniforms[LightPosition] = glGetUniformLocation(_program, "LightPosition");
	_uniforms[LightMatrix] = glGetUniformLocation(_program, "LightMatrix");
}

Shader::~Shader()
{
	for (unsigned int i = 0; i < NUM_SHADER; i++)
	{
		glDetachShader(_program, _shaders[i]);
		glDeleteShader(_shaders[i]);
	}
	glDeleteProgram(_program);
}

void Shader::bind(void)
{
	glUseProgram(_program);
}

void Shader::update(const Camera& camera)
{
	glm::mat4 model = camera.GetModelMatrix();
	glm::mat4 view = camera.GetViewMatrix();
	glm::mat4 projection = camera.GetModelViewProjection();
	glm::vec3 camPos = camera.GetCameraPos();

	glUniformMatrix4fv(_uniforms[ModelMatrix], 1, GL_FALSE, &model[0][0]);
	glUniformMatrix4fv(_uniforms[ViewMatrix], 1, GL_FALSE, &view[0][0]);
	glUniformMatrix4fv(_uniforms[ProjectionMatrix], 1, GL_FALSE, &projection[0][0]);

	glUniform3fv(_uniforms[CameraPos], 1, &camPos[0]);

	float near_plane = 0.1f, far_plane = 20.0f;
	glm::mat4 lightProjection = glm::ortho(-20.0f, 20.0f, -20.0f, 20.0f,
										near_plane, far_plane);
	glm::vec3 lightPos = glm::vec3(-3.0f, 0.0f, 3.0f);
	glm::mat4 lightView = glm::lookAt(lightPos, 
										glm::vec3(0.0f, 0.0f, 0.0f), 
										glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 lightMatrix = lightProjection * lightView;
	
	glUniformMatrix4fv(_uniforms[LightMatrix], 1, GL_FALSE, &lightMatrix[0][0]);
	glUniform3fv(_uniforms[LightPosition], 1, &lightPos[0]);
}

glm::mat3 Shader::getNormalMat(const glm::mat4& modelViewMatrix)
{
	glm::mat3 tempMatrix;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			tempMatrix[i][j] = modelViewMatrix[i][j];
		}
	}

	return glm::transpose(glm::inverse(tempMatrix));
}

static GLuint CreateShader(const string& text, GLenum shaderType)
{
	GLuint shader = glCreateShader(shaderType);

	if (shader == 0)
		cout << "Error: Shader Creation failed!" << endl;

	const GLchar* shaderSourceString[1];
	GLint shaderSourceStringLength[1];

	shaderSourceString[0] = text.c_str();
	shaderSourceStringLength[0] = (GLint)text.length();

	glShaderSource(shader, 1, shaderSourceString, shaderSourceStringLength);
	glCompileShader(shader);

	CheckShaderError(shader, GL_COMPILE_STATUS, false, "Error: Shader Compilation failed: ");

	return shader;
}

static string LoadShader(const string& fileName)
{
	ifstream file;
	file.open((fileName).c_str());

	string output;
	string line;

	if (file.is_open())
	{
		while (file.good())
		{
			getline(file, line);
			output.append(line + "\n");
		}
	}
	else
	{
		cout << "Unable to load shader: " << fileName << endl;
	}

	return output;
}

static void CheckShaderError(GLuint shader, GLuint flag, bool isProgram, const string& errorMessage)
{
	GLint success = 0;
	GLchar error[1024] = { 0 };

	if (isProgram)
		glGetProgramiv(shader, flag, &success);
	else
		glGetShaderiv(shader, flag, &success);

	if (success == GL_FALSE)
	{
		if (isProgram)
			glGetProgramInfoLog(shader, sizeof(error), NULL, error);
		else
			glGetShaderInfoLog(shader, sizeof(error), NULL, error);

		cout << errorMessage << error << endl;
	}
}