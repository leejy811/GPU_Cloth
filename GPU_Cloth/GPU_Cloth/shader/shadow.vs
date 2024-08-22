#version 330 core

layout (location=0) in vec3 VertexPosition;
layout (location=1) in vec3 VertexNormal;
layout (location=2) in vec2 VertexTexCoord;

out vec3 fragPosition;
out vec3 fragNormal;
out vec4 lightSpacePosition;

uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 LightMatrix;

void main()
{
	fragNormal = transpose(inverse(mat3(ModelMatrix))) * VertexNormal;
	fragPosition = vec3(ModelMatrix * vec4(VertexPosition, 1.0));
	gl_Position = ProjectionMatrix * ViewMatrix * vec4(fragPosition, 1.0);
	lightSpacePosition = LightMatrix * vec4(fragPosition, 1.0);
}