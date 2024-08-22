in vec3 Position;

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;

uniform float pointRadius;

void main()
{ 	
	// Calculate normal from texture coordinates
	vec3 normal;
	normal.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);	
	float mag = dot(normal.xy, normal.xy);
	
	if (mag > 1.0)	discard;  	
	normal.z = sqrt(1.0-mag);
	
	//calculate depth
	vec4 pixelPos = vec4(Position + normal * pointRadius, 1.0);
	vec4 clipSpacePos = ProjectionMatrix * pixelPos;
	gl_FragDepth = (clipSpacePos.z / clipSpacePos.w) * 0.5f + 0.5f;
}

