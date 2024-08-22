#version 430

in vec3 Position;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

//uniform vec4 LightPosition;

uniform sampler2D Diffuse;

const float PI = 3.141592;
const bool Metal = false;
const float Rough = 0.5f;
const vec3 Color = vec3(1.0, 0.8, 0.7);
const vec3 LightIntensity = vec3(1.0f, 1.0f, 1.0f);
const vec4 LightPosition = vec4(1000.0f, 1000.0f, 2000.0f, 0.0f);

/*
vec3 SchlickFresnel(float lDotH)
{
	vec3 f0 = vec3(0.04); // Dielectrics
	if (Metal) {
		f0 = Color;
		//f0 = texture(Diffuse, TexCoord).rgb;
	}
	return f0 + (1 - f0) * pow(1.0 - lDotH, 5);
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
*/



void main()
{
#if 1
	mat4 gr = {
	{ 0.087245, -0.266758, -0.245714, -0.528563},
	{-0.266758, -0.087245,  0.033388,  1.477590},
	{-0.245714,  0.033388, -0.695238,  0.193936},
	{-0.528563,  1.477590,  0.193936,  3.584577}
	};

	mat4 gg = {
	{-0.018964, -0.237875, -0.215462, -0.527878},
	{-0.237875,  0.018964, -0.063632,  1.835238},
	{-0.215462, -0.063632, -0.932072,  0.151051},
	{-0.527878,  1.835238,  0.151051,  4.086394}
	};

	mat4 gb = {
	{-0.194005, -0.170235, -0.155918, -0.452784},
	{-0.170235,  0.194005, -0.202208,  2.121897},
	{-0.155918, -0.202208, -1.133733,  0.050433},
	{-0.452784,  2.121897,  0.050433,  4.369998}
	};
	float albedo_2 = 0.06;
	#else	
	mat4 gr = {
  { 0.009098, -0.004780,  0.024033, -0.014947 },
  {-0.004780, -0.009098, -0.011258,  0.020210 },
  { 0.024033, -0.011258, -0.011570, -0.017383 },
  {-0.014947,  0.020210, -0.017383,  0.073787 }
} ;
mat4 gg = {
  {-0.002331, -0.002184,  0.009201, -0.002846 },
  {-0.002184,  0.002331, -0.009611,  0.017903 },
  { 0.009201, -0.009611, -0.007038, -0.009331 },
  {-0.002846,  0.017903, -0.009331,  0.041083 }
} ;
mat4 gb = {
  {-0.013032, -0.005248,  0.005970,  0.000483 },
  {-0.005248,  0.013032, -0.020370,  0.030949 },
  { 0.005970, -0.020370, -0.010948, -0.013784 },
  { 0.000483,  0.030949, -0.013784,  0.051648 }
} ;
float albedo_2 = 2.882;

#endif
	
	
	//vec4 n = vec4(normal, 1.0);
	vec4 n = vec4(normalize(Normal), 1.0);
	float albedo = 0.12;

#if 1
	FragColor = vec4(
		albedo * dot(n, gr * n),
		albedo * dot(n, gg * n),
		albedo * dot(n, gb * n),
		 1.0);
#else
		 	FragColor = vec4(
		albedo_2 * dot(n, gr * n),
		albedo_2 * dot(n, gg * n),
		albedo_2 * dot(n, gb * n),
		 1.0);
#endif


	//vec3 sum = vec3(0.0);
	//vec3 n = normalize(Normal);
	//sum = MicroFacetModel(Position, n);
	//FragColor = vec4(sum, 1.0);
}