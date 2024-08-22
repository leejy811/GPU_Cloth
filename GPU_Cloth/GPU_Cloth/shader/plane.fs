in vec3 Position;

uniform mat4 ModelViewMatrix;

void main()
{ 	
	vec3 color1 = vec3(0.8);
	vec3 color2 = vec3(0.9);
	vec3 color = mix(color1, color2, 0.5 * mod(floor(Position.x) + floor(Position.y) + floor(Position.z), 2));
	gl_FragColor = vec4(color, 1.0);	
}

