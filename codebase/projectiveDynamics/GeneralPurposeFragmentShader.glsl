#version 330 core

in vec3 Normal_cameraspace;
in vec3 EyeDirection_cameraspace;
in vec3 o_diffuseColor;
in vec3 o_specularColor;
in vec2 UV;
// Ouput data
out vec3 color;

// texture sampler
uniform sampler2D ts;
uniform float renderImg;

void main(){

	
	
		vec3 n = normalize( Normal_cameraspace );
		vec3 l = normalize(EyeDirection_cameraspace );
		float cosTheta = clamp( dot( n,l ), 0,1 );
		
		if(renderImg>0.0f){
			vec3 texture_color = texture(ts, UV ).rgb;
			color = 0.7* texture_color + texture_color*pow(cosTheta,5);
		}
		else{
			
			color = 
			// Diffuse : "color" of the object
			o_diffuseColor +
			// Specular : reflective highlight, like a mirror
			o_specularColor * pow(cosTheta,5);
		}

}