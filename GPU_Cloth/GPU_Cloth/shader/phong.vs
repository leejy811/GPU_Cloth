#version 330 core

layout (location=0) in vec3 VertexPosition;
layout (location=1) in vec3 VertexNormal;
layout (location=2) in vec2 VertexTexCoord;

out vec4 fragPosition;
out vec3 fragNormal;
out vec2 fragTexCoord;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;
uniform mat3 NormalMatrix;

void main()
{
	fragNormal = NormalMatrix * VertexNormal;
	gl_Position = MVP * vec4(VertexPosition, 1.0);
	fragPosition = gl_Position;
}