#version 430

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

//uniform vec4 LightPosition;

uniform sampler2D Diffuse;

const float PI = 3.141592;
const bool Metal = false;
const float Rough = 0.2;
const vec3 Color = vec3(0.4, 0.55, 0.85);
const vec3 LightIntensity = vec3(1.0f, 1.0f, 1.0f);
const vec4 LightPosition = vec4(1000.0f, 1000.0f, 2000.0f, 0.0f);

vec3 SchlickFresnel(float lDotH)
{
	vec3 f0 = vec3(0.04); // Dielectrics
	if (Metal) {
		f0 = Color;
		//f0 = texture(Diffuse, TexCoord).rgb;
	}
	return f0 + (vec3(1) - f0) * pow(1.0 - lDotH, 5);
}

float GeomSmith(float dotProd)
{
	float k = (Rough + 1.0) * (Rough + 1.0) / 8.0;
	float denom = dotProd * (1 - k) + k;

	return 1.0 / denom;
}

float GGXDistribution(float nDotH)
{
	float alpha2 = Rough * Rough * Rough * Rough;
	float d = (nDotH * nDotH) * (alpha2 - 1) + 1;

	return alpha2 / (PI * d * d);
}

vec3 MicroFacetModel(vec3 position, vec3 n)
{
	vec3 diffuseBrdf = vec3(0.0);

	if (!Metal)
	{
		diffuseBrdf = Color;
		//diffuseBrdf = texture(Diffuse, TexCoord).rgb;
	}

	vec3 l = vec3(0.0);
	vec3 lightI = LightIntensity;

	if (LightPosition.w == 0)	// Directional Light
	{
		l = normalize(LightPosition.xyz);
	}
	else
	{
		l = LightPosition.xyz - Position;
		float dist = length(l);
		l = normalize(l);
		lightI /= (dist*dist);
	}

	vec3 v = normalize(-position);
	vec3 h = normalize(v+l);

	float nDotH = dot(n, h);
	float lDotH = dot(l, h);
	float nDotL = max(dot(n, l), 0.0);
	float nDotV = dot(n, v);

	vec3 specBrdf = 0.25 * GGXDistribution(nDotH) * SchlickFresnel(lDotH) * GeomSmith(nDotL) * GeomSmith(nDotV);
	
	return (diffuseBrdf + PI * specBrdf) * lightI * nDotL;
}

void main()
{
	vec3 sum = vec3(0.0);
	vec3 n = normalize(Normal);
	sum = MicroFacetModel(Position, n);
	FragColor = vec4(sum, 1.0);
}