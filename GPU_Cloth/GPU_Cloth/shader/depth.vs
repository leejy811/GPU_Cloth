#version 330 core

layout (location=0) in vec3 VertexPosition;

uniform mat4 ModelMatrix;
uniform mat4 LightMatrix;

void main()
{
	gl_Position = LightMatrix * ModelMatrix * vec4(VertexPosition, 1.0);
}