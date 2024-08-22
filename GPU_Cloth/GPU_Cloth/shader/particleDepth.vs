#version 430 compatibility

layout (location=0) in vec3 VertexPosition;
layout (location=1) in vec3 VertexColor;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;
uniform mat3 NormalMatrix;

out vec3 Position;

uniform float pointRadius;
uniform float pointScale;

void main()
{		
	Position = (ModelViewMatrix * vec4(VertexPosition, 1.0)).xyz;
	gl_Position = ProjectionMatrix * vec4(Position, 1.0);
	gl_PointSize = pointScale * (pointRadius / gl_Position.w);
}

