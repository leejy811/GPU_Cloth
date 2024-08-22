#version 430 compatibility

layout (location=0) in vec3 VertexPosition;
layout (location=1) in vec3 VertexColor;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;
uniform mat3 NormalMatrix;

out vec3 Position;

void main()
{		
	gl_Position = MVP * vec4(VertexPosition, 1.0);
}

