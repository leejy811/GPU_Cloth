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
	//Position = vec3(ModelViewMatrix * vec4(VertexPosition, 1.0));
	Position = (ModelViewMatrix * vec4(VertexPosition, 1.0)).xyz;
	//gl_Position = MVP * vec4(VertexPosition, 1.0);
	gl_Position = MVP * vec4(VertexPosition, 1.0);
	vec3 posEye = vec3(ModelViewMatrix * vec4(VertexPosition, 1.0));
	float dist = length(posEye);
 	gl_PointSize = pointRadius * (pointScale / dist);
 	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_FrontColor = vec4(VertexColor, 1.0);
}

