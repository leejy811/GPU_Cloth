#version 430 compatibility

in vec3 Position;
in vec3 Normal;

out vec4 FragColor;

uniform mat4 ModelViewMatrix;

void PointLight(in vec3 eye, in vec3 ecPosition3, in vec3 normal, inout vec4 ambient, inout vec4 diffuse, inout vec4 specular)
{
	float nDotVP;         // normal . light direction
	float nDotHV;         // normal . light half vector
	float pf;             // power factor
	float attenuation;    // computed attenuation factor
	float d;              // distance from surface to light source
	vec3  VP;             // direction from surface to light position
	vec3  halfVector;     // direction of maximum highlights
	
	// user defined lights
	vec3 lightPos = vec3(0.0f, 0.0f, 0.0f);
	float constantAttenuation = 1.0f;
	float linearAttenuation = 0.0f;
	float quadraticAttenuation = 0.0f;
	float shininess = 100.0f;
    
	// Compute vector from surface to light position
	VP = lightPos - ecPosition3;

	// Compute distance between surface and light position
	d = length(VP);

	// Normalize the vector from surface to light position
	VP = normalize(VP);

	// Compute attenuation
	//attenuation = 1.0 / (constantAttenuation + linearAttenuation * d + quadraticAttenuation * d * d);   
	attenuation = 1.0 / constantAttenuation;

	halfVector = normalize(VP + eye);
  
	nDotVP = max(0.0, dot(normal, VP));
	nDotHV = max(0.0, dot(normal, halfVector));

	if (nDotVP == 0.0)
		pf = 0.0;
	else
		pf = pow(nDotHV, shininess);
        
         
	vec4 lightAmbient = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	vec4 lightDiffuse = vec4(1.0f, 1.0f, 0.0f, 1.0f);
	vec4 lightSpecular = vec4(1.0f, 1.0f, 1.0f, 1.0f);

	ambient += lightAmbient * attenuation;
	diffuse += lightDiffuse * nDotVP * attenuation;
	specular += lightSpecular * pf * attenuation;   
}

void SpotLight(in vec3 eye, vec3 ecPosition3, in vec3 normal, inout vec4 ambient, inout vec4 diffuse, inout vec4 specular)
{
	float nDotVP;           // normal . light direction
	float nDotHV;           // normal . light half vector
	float pf;               // power factor
	float spotDot;          // cosine of angle between spotlight
	float spotAttenuation;  // spotlight attenuation factor
	float attenuation;      // computed attenuation factor
	float d;                // distance from surface to light source
	vec3 VP;                // direction from surface to light position
	vec3 halfVector;        // direction of maximum highlights
	
	vec4 lightPos = vec4(1000.0f, 1000.0f, 1000.0f, 1.0f);
	vec3 spotDir = vec3(-1.0f, -1.0f, -1.0f);
	float spotCutoff = 10.0;
	float spotExponent = 1.0f;
	float constantAttenuation = 1.0f;
	float linearAttenuation = 0.0f;
	float quadraticAttenuation = 0.0f;	
	float shininess = 100.0f;
	
	// Compute vector from surface to light position
	VP = vec3(lightPos) - ecPosition3;

	// Compute distance between surface and light position
	d = length(VP);

	// Normalize the vector from surface to light position
	VP = normalize(VP);

	// Compute attenuation
	//attenuation = 1.0 / (constantAttenuation + linearAttenuation * d + quadraticAttenuation * d * d);   
	attenuation = 1.0 / constantAttenuation;
                         
                       
	// See if point on surface is inside cone of illumination
	spotDot = dot(-VP, normalize(spotDir));

	if (spotDot > spotCutoff)
		spotAttenuation = 0.0; // light adds no contribution
	else
		spotAttenuation = pow(spotDot, spotExponent);


	// Combine the spotlight and distance attenuation.
	attenuation *= spotAttenuation;

	halfVector = normalize(VP + eye);

	nDotVP = max(0.0, dot(normal, VP));
	nDotHV = max(0.0, dot(normal, halfVector));

	if (nDotVP == 0.0)
		pf = 0.0;
	else
		pf = pow(nDotHV, shininess);
	
	vec4 lightAmbient = vec4(0.1f, 0.1f, 0.1f, 1.0f);
	vec4 lightDifuse = vec4(1.0f, 1.0f, 0.0f, 1.0f);
	vec4 lightSpecular = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	
	ambient  += lightAmbient * attenuation;
	diffuse  += lightDifuse * nDotVP * attenuation;
	specular += lightSpecular * pf * attenuation;	
}


vec4 calc_lighting_color(in vec3 _pos, in vec3 _normal) 
{
	vec3 eye = vec3(0.0, 0.0, 1.0);
		
	vec4 amb  = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec4 diff = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec4 spec = vec4(0.0f, 0.0f, 0.0f, 1.0f);	

	//PointLight(eye, _pos, normalize(_normal), amb, diff, spec);
	SpotLight(eye, _pos, normalize(_normal), amb, diff, spec);
	
	vec4 color = amb + diff + spec;
	return color;
}

void main()
{
	// Basic code
	vec4 LightPosition = ModelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
	vec3 LightIntensity = vec3(1.0,1.0,1.0);

	vec3 Ka = vec3(0.05,0.05,0.05);
	vec3 Kd = vec3(1.0, 1.0, 1.0);
	vec3 Ks = vec3(0.05,0.05,0.05);
	float shininess = 100.0f;

	vec3 n = normalize(Normal);
	vec3 s = normalize(vec3(LightPosition) - Position);
	vec3 v = normalize(vec3(-Position));
	vec3 r = reflect(-s, n);

	if(gl_FrontFacing) {
		FragColor = vec4 (LightIntensity*(Ka+Kd*max(dot(s,n),0.0)+Ks*pow(max(dot(r,v),0.0), shininess)), 1.0);
	} else {
		discard;
	}
}

