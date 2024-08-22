#version 430 compatibility

layout (location=0) in vec3 VertexPosition;
layout (location=1) in vec3 VertexNormal;

out vec3 Position;
out vec3 Normal;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;
uniform mat3 NormalMatrix;

void main()
{

	Position = (ModelViewMatrix * vec4(VertexPosition, 1.0)).xyz;
	Normal = normalize(NormalMatrix * VertexNormal);	
	gl_Position = MVP * vec4(VertexPosition, 1.0);
}

