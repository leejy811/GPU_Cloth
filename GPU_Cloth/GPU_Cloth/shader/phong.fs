#version 330 core

struct DirectionalLight
{
	vec3 mDirection;
	vec3 mDiffuseColor;
	vec3 mSpecColor;
};

in vec2 fragTexCoord;
in vec3 fragNormal;
in vec4 fragPosition;

uniform vec3 CameraPosition;

out vec4 outColor;

void main()
{
	float specPower = 1024;
	vec3 ambientLight = vec3(1.0f, 1.0f, 1.0f) * 0.3f;
	DirectionalLight dirLight;
	dirLight.mDirection = vec3(1.0f, -1.0f, 0.0f);
	dirLight.mDiffuseColor = vec3(1.0f, 1.0f, 1.0f);
	dirLight.mSpecColor = vec3(1.0f, 1.0f, 1.0f);

	vec3 N = normalize(fragNormal);
	vec3 L = normalize(-dirLight.mDirection);
	vec3 V = normalize(vec3(CameraPosition - vec3(fragPosition)));
	vec3 R = 2*dot(L, N)*N-L;

	vec3 Phong = ambientLight;
	float NdotL = dot(N, L);
	if(NdotL > 0)
	{
		vec3 Diffuse = dirLight.mDiffuseColor * NdotL;
		vec3 Specular = dirLight.mSpecColor * 
						pow(max(0.0, dot(R, V)), specPower);
		Phong += Diffuse + Specular;
	}

	//outColor = vec4(Phong,1.0f);
	outColor = vec4(vec3(gl_FragCoord.z), 1.0);
}