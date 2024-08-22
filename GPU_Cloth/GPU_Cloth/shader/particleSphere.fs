in vec3 Position;

uniform mat4 ModelViewMatrix;

void main()
{ 	
	// Lighting
	vec4 L = ModelViewMatrix * vec4(0.5, 1.5, 1.5, 1.0);
	vec3 lightDir = normalize(vec3(L)-Position);
	
	// Calculate normal from texture coordinates
	vec3 N;
	N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);	
	float mag = dot(N.xy, N.xy);
	
	// Kill pixels outside circle
	if (mag > 1.0) discard;  	
	N.z = sqrt(1.0-mag);
	
	vec3 lightPos = vec3(2, 1, 1);
	vec3 ks = vec3(0.5, 0.5, 0.5);
	float shininess = 128;	
   	vec3 l = normalize(lightPos);
	vec3 v = lightDir;
	vec3 h = normalize(v + l);
	vec3 finalColor = gl_Color * 0.1 + gl_Color * max(dot(l, N), 0.0) + ks * pow(max(dot(h, N), 0.0), shininess);
	gl_FragColor = vec4(finalColor, 1.0);	
}

