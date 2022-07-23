#version 460
#extension GL_KHR_vulkan_glsl: enable
#define MAX_NS_LIGHTS 30
#define MAX_PLIGHTS 8
#define MAX_DLIGHTS 8
#define MAX_PLIGHT_SHADOW_DISTANCE 200.0
#define MAX_DLIGHT_SHADOW_DISTANCE 200.0
#define MAX_PLIGHT_SHADOW_FALLOFF_DISTANCE 100.0
#define MAX_DLIGHT_SHADOW_FALLOFF_DISTANCE 100.0

#define SMAP_MUL_EPSILON 1.000
#define SMAP_ADD_EPSILON -0.001
#define SMAP_BLUR_EPSILON 0.01
#define SMAP_BLUR_SCALING 3
#define SMAP_BLUR_FALLOFF 0.5
#define SMAP_BLUR_SAMPLES 12

#define SCUBE_MUL_EPSILON 1.05
#define SCUBE_ADD_EPSILON 0.01
#define SCUBE_BLUR_EPSILON 0.007
#define SCUBE_BLUR_SCALING 20
#define SCUBE_BLUR_FALLOFF 0.6
#define SCUBE_BLUR_SAMPLES 45

#define SCUBE_SOFT_MUL_EPSILON 1.00
#define SCUBE_SOFT_ADD_EPSILON 0.01
#define SCUBE_SOFT_BLUR_EPSILON 0.5
#define SCUBE_SOFT_BLUR_SCALING 10
#define SCUBE_SOFT_BLUR_FALLOFF 0.8
#define SCUBE_SOFT_BLUR_SAMPLES 45

#define SCUBE_ANTIBLUR_EPSILON 0.01
#define PI 3.1415926538

#define SHADOW_DARKNESS 0
#define SHADOW_BRIGHTNESS 1
#define SHADOW_MIN_BRIGHTNESS 0.01

#define AMBIENT 0.05

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

struct GPULight {
	vec4 pos;
	vec4 color;
	vec4 dir;
	vec4 props;
	mat4 proj;
	mat4 viewproj;
	ivec4 flags;
};

layout(set = 0, binding = 0) uniform GPUCameraData {
	vec4 camPos;
    vec4 camDir;
	vec4 projprops;
    mat4 viewproj;
} camData;
layout(set = 1, binding = 0) uniform GPUSceneData {
	vec4 fogColor; // w is for exponent
	vec4 fogDistances; //x for min, y for max, zw unused.
	vec4 ambientColor;
	vec4 sunlightPosition;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
	vec4 screenData;
} sceneData;
layout(set = 2, binding = 0) uniform sampler2D texMaps[5];
layout(std140, set = 3, binding = 0) uniform LightBuffer {GPULight lights[MAX_NS_LIGHTS + MAX_PLIGHTS + MAX_DLIGHTS];} lightBuffer;
layout(set = 4, binding = 0) uniform samplerCube shadowCubes[MAX_PLIGHTS];
layout(set = 5, binding = 0) uniform sampler2D shadowMaps[MAX_DLIGHTS];
layout (set = 6, binding = 0) uniform sampler2D smapRVec;
layout (set = 7, binding = 0) uniform rKern{ vec4 data[64];} smapKernel;


//Pre Calculated sin and cos values for smap blur
//float smap_sines[] = {0.0, 0.06540312923014306, 0.13052619222005157, 0.19509032201612825, 0.25881904510252074, 0.3214394653031616, 0.3826834323650898, 0.44228869021900125, 0.49999999999999994, 0.5555702330196022, 0.6087614290087207, 0.6593458151000688, 0.7071067811865476, 0.7518398074789774, 0.7933533402912352, 0.8314696123025451, 0.8660254037844386, 0.8968727415326884, 0.9238795325112867, 0.9469301294951056, 0.9659258262890683, 0.9807852804032304, 0.9914448613738104, 0.9978589232386035};
float smap_sines[] = {0.0, 0.13052619222005157, 0.25881904510252074, 0.3826834323650898, 0.49999999999999994, 0.60876142900872078, 0.7071067811865476, 0.7933533402912352, 0.8660254037844386, 0.9238795325112867, 0.9659258262890683, 0.9914448613738104};
//float smap_cosines[] = {1.0, 0.9978589232386035, 0.9914448613738104, 0.9807852804032304, 0.9659258262890683, 0.9469301294951057, 0.9238795325112867, 0.8968727415326884, 0.8660254037844387, 0.8314696123025452, 0.7933533402912352, 0.7518398074789774, 0.7071067811865476, 0.6593458151000688, 0.6087614290087207, 0.5555702330196024, 0.5000000000000001, 0.44228869021900125, 0.38268343236508984, 0.3214394653031617, 0.25881904510252074, 0.19509032201612833, 0.1305261922200517, 0.06540312923014327};
float smap_cosines[] = {1.0, 0.9914448613738104, 0.9659258262890683, 0.9238795325112867, 0.8660254037844387, 0.7933533402912352, 0.7071067811865476, 0.6087614290087207, 0.5000000000000001, 0.38268343236508984, 0.25881904510252074, 0.1305261922200517};

mat3 TBN;

vec3 norm_vec_in_dir(vec3 v, vec3 dir, float val){
	if (length(dir) == 0) return v;
	vec3 ndir = normalize(dir);
	vec3 tmpo =  v + (val-dot(ndir, v)) * ndir;
	tmpo = tmpo + (val-dot(ndir, tmpo)) * ndir;
	return tmpo;
}

vec4 norm_vec_in_dir(vec4 v, vec4 dir, float val){
	if (length(dir) == 0) return v;
	vec4 ndir = normalize(dir);
	vec4 tmpo =  v + (val-dot(ndir, v)) * ndir;
	tmpo = tmpo + (val-dot(ndir, tmpo)) * ndir;
	return tmpo;
}

float compute_soft_shadow(int lidx, vec3 light_vec){
	vec3 v1, v2;

	float lvl = length(light_vec);
	if (lvl > MAX_PLIGHT_SHADOW_DISTANCE) return 1;
	vec3 lvn = normalize(light_vec);

	if (lvn.y == 0 && lvn.z == 0) {v1=vec3(0, 15, 0);}
	else {v1 = vec3(15,0,0);}

	v2 = cross(lvn, v1);
	//v1 = cross(lvn, v2);
	float light_samples = 0;
	float sharp_depth = texture(shadowCubes[lidx], light_vec).r;
	if(sharp_depth*SCUBE_SOFT_MUL_EPSILON >= lvl){light_samples += 1;}
	vec3 sample_vec;
	float brad = max(SCUBE_SOFT_BLUR_EPSILON*(1-sharp_depth/lvl), SCUBE_BLUR_EPSILON*lvl);
	float sample_weight = SCUBE_SOFT_BLUR_FALLOFF/(SCUBE_SOFT_BLUR_SCALING*brad+1);
	float theta = PI/SCUBE_SOFT_BLUR_SAMPLES;
	for (int i = 0; i < SCUBE_SOFT_BLUR_SAMPLES; i++){
		sample_vec = v1*cos(i*theta) + v2*sin(i*theta);
		sample_vec *= brad;
		if(texture(shadowCubes[lidx], light_vec + sample_vec).r*SCUBE_SOFT_MUL_EPSILON + SCUBE_SOFT_ADD_EPSILON >= lvl){light_samples += sample_weight;}
		if(texture(shadowCubes[lidx], light_vec - sample_vec).r*SCUBE_SOFT_MUL_EPSILON + SCUBE_SOFT_ADD_EPSILON >= lvl){light_samples += sample_weight;}
	}
	return SHADOW_DARKNESS + SHADOW_BRIGHTNESS*(light_samples/(1 + 2*SCUBE_SOFT_BLUR_SAMPLES*sample_weight));
}

float compute_shadow(int lidx, vec3 light_vec){
	int midx = lidx - MAX_NS_LIGHTS;
	vec3 v1, v2;

	float lvl = length(light_vec);
	if (lvl > MAX_PLIGHT_SHADOW_DISTANCE) return SHADOW_DARKNESS;
	vec3 lvn = normalize(light_vec);

	float fader = 0;
	if (lvl > MAX_PLIGHT_SHADOW_FALLOFF_DISTANCE) fader = (SHADOW_BRIGHTNESS)*(lvl - MAX_PLIGHT_SHADOW_FALLOFF_DISTANCE)/(MAX_PLIGHT_SHADOW_DISTANCE - MAX_PLIGHT_SHADOW_FALLOFF_DISTANCE);

	vec3 lvt = texture(smapRVec, inUV * vec2(sceneData.screenData.x/4.0, sceneData.screenData.y/4.0)).xyz;
	lvt = normalize(lvt - dot(lvt, lvn)*lvn);
	vec3 lvb = cross(lvt, lvn);

	mat3 pltbn = mat3(lvt, lvb, lvn);

	float light_samples = 0;
	if(texture(shadowCubes[midx], light_vec).r*SCUBE_MUL_EPSILON >= lvl){light_samples += 1;}
	vec3 sample_vec;
	float brad = SCUBE_BLUR_EPSILON*lvl;
	for (int i = 0; i < 64; i+=4){
		sample_vec = brad * (pltbn * smapKernel.data[i].xyz);
		if(texture(shadowCubes[midx], light_vec + sample_vec).r*SCUBE_MUL_EPSILON + SCUBE_ADD_EPSILON >= lvl){light_samples += 1;}
		if(texture(shadowCubes[midx], light_vec - sample_vec).r*SCUBE_MUL_EPSILON + SCUBE_ADD_EPSILON >= lvl){light_samples += 1;}
	}
	float sprob = smoothstep(0.0, 1.0, light_samples/33.0);
	
	return (SHADOW_DARKNESS) + (SHADOW_BRIGHTNESS - fader)*sprob;
}

float compute_map_shadow(int lidx, vec3 light_vec, vec4 ppos, vec4 normal){
	float lvl = length(light_vec);
	if (lvl > MAX_DLIGHT_SHADOW_DISTANCE) return SHADOW_DARKNESS;
	vec3 lvn = normalize(light_vec);

	float fader = 0;
	if (lvl > MAX_DLIGHT_SHADOW_FALLOFF_DISTANCE) fader = (SHADOW_BRIGHTNESS)*(lvl - MAX_DLIGHT_SHADOW_FALLOFF_DISTANCE)/(MAX_DLIGHT_SHADOW_DISTANCE - MAX_DLIGHT_SHADOW_FALLOFF_DISTANCE);
	
	vec4 smuv = lightBuffer.lights[lidx].viewproj * vec4(ppos.xyz, 1.0f);
	vec4 smuvn = smuv/smuv.w;
	if (smuvn.z <-1.0 || smuvn.z > 1.0 || smuvn.w < 0.0) return SHADOW_DARKNESS + fader;

	int midx = lidx - MAX_PLIGHTS - MAX_NS_LIGHTS;
	float light_samples = 0;
	float total_samples = 1;
	vec2 csample = vec2(smuvn.x/2.0 + 0.5f, -smuvn.y/2.0 + 0.5f);
	float cdist = texture(shadowMaps[midx], csample).r;
	if (cdist + SMAP_ADD_EPSILON >= lvl){light_samples += 1;}

	vec2 sample_vec;
	float brad = max(0.005, SMAP_BLUR_EPSILON*(1 - cdist/lvl));
	
	for (int i = 0; i < 64; i+=2){
		sample_vec = csample + brad*smapKernel.data[i].xy;
		if (sample_vec.x > 0 && sample_vec.x < 1 && sample_vec.y > 0 && sample_vec.y < 1){
			if(texture(shadowMaps[midx], sample_vec).r*SMAP_MUL_EPSILON + SMAP_ADD_EPSILON >= lvl){
				light_samples += 1;
			}
			total_samples += 1;
		}
		sample_vec = csample - brad*smapKernel.data[i].xy;
		if (sample_vec.x > 0 && sample_vec.x < 1 && sample_vec.y > 0 && sample_vec.y < 1){
			if(texture(shadowMaps[midx], sample_vec).r*SMAP_MUL_EPSILON + SMAP_ADD_EPSILON >= lvl){
				light_samples += 1;
			}
			total_samples += 1;
		}
	}
	float sprob = smoothstep(0.0, 1.0, light_samples/total_samples);
	
	return (SHADOW_DARKNESS) + (SHADOW_BRIGHTNESS - fader)*sprob;
}

void main() {
    vec4 texColor = texture(texMaps[0], inUV);
    vec4 fragPos = texture(texMaps[1], inUV);
    vec4 normTex = texture(texMaps[2], inUV);
	normTex = normTex - vec4(0.5);
	normTex = 2 * normTex;

	vec4 seTex = texture(texMaps[3], inUV);

	vec2 movedUV = inUV;
	movedUV.y = 1 - movedUV.y;
	vec4 ambTex = texture(texMaps[4], movedUV);

	vec3 tangent = texture(smapRVec, inUV * vec2(sceneData.screenData.x/4.0, sceneData.screenData.y/4.0)).xyz;
	tangent = normalize(tangent - (dot(tangent, normTex.xyz) * normTex.xyz));
	tangent = normalize(tangent - (dot(tangent, normTex.xyz) * normTex.xyz));
	vec3 bitangent = cross(normTex.xyz, tangent);
	TBN = mat3(tangent, bitangent, normTex.xyz);

	vec4 overall_shade = vec4(0,0,0,0);
	for (int i = 0; i < MAX_NS_LIGHTS + MAX_PLIGHTS + MAX_DLIGHTS; i++){
		vec4 diffuse = vec4(0);
		vec4 specular = vec4(0);
		float shadow = 1;
		vec3 lightVec = lightBuffer.lights[i].pos.xyz - fragPos.xyz;
		vec3 lightVecN = normalize(lightVec);

		bool valid_ls = false;

		if (bool(lightBuffer.lights[i].flags.x & 1)){
			valid_ls = true;
			if (i >= MAX_NS_LIGHTS && i < MAX_NS_LIGHTS + MAX_PLIGHTS){
				shadow = compute_shadow(i, lightVec);
			}
			else if(i >= MAX_NS_LIGHTS + MAX_PLIGHTS){
				shadow = compute_map_shadow(i, lightVec, vec4(fragPos.xyz, 1.0f), normTex);
			}
		}

		shadow = SHADOW_MIN_BRIGHTNESS + (1 - SHADOW_MIN_BRIGHTNESS)*shadow;

		if (bool((lightBuffer.lights[i].flags.x >> 1) & 1)){
			valid_ls = true;
			float lcomp = max(dot(normTex.xyz,lightVecN), 0);
			float ldist = lightBuffer.lights[i].props.y;
			if (ldist < length(lightVec)){
				diffuse = vec4(0);
			}
			else{
				lcomp = lcomp * (1 - (length(lightVec)/ldist));
				diffuse = 1 * lcomp * lcomp * lightBuffer.lights[i].color;
				if (i >= MAX_NS_LIGHTS + MAX_PLIGHTS){
					vec4 smuv = lightBuffer.lights[i].viewproj * vec4(fragPos.xyz, 1.0f);
					vec4 smuvn = smuv/smuv.w;
					if (smuvn.z < -1 || smuvn.z > 1 || smuvn.w < 0 || smuvn.x > 1 || smuvn.x < -1 || smuvn.y > 1 || smuvn.y < -1) diffuse = vec4(0);
				}
			}

			vec3 viewDir = normalize(camData.camPos.xyz - fragPos.xyz);
			vec3 reflectDir = reflect(-lightVecN, normTex.xyz);  
			float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32)/pow(length(lightVec), 2);
			specular = 10 * seTex.x * spec * lightBuffer.lights[i].color;
		}

		if (valid_ls){
			vec4 light_shade = (diffuse + specular) * shadow;
			overall_shade = overall_shade + light_shade;
		}
	}
    
	float ao = 0;
	float brad = 0.01f;

	brad = 0.01/length(camData.camPos.xyz - fragPos.xyz);

	float taow = 0;

	for (int i = 0; i < SMAP_BLUR_SAMPLES ; i+=4){
		vec2 sample_vec = brad*i*vec2(smap_cosines[i], smap_sines[i]);
		vec2 ssvec = movedUV + sample_vec;
		if (ssvec.x >0 && ssvec.x < 1 && ssvec.y > 0 && ssvec.y < 1){
			ao += texture(texMaps[4], ssvec).r;
			taow += 0.5;
		}
		ssvec = movedUV - sample_vec;
		if (ssvec.x >0 && ssvec.x < 1 && ssvec.y > 0 && ssvec.y < 1){
			ao += texture(texMaps[4], ssvec).r;
			taow += 0.5;
		}
		sample_vec = brad*i*vec2(smap_cosines[i], -smap_sines[i]);
		ssvec = movedUV + sample_vec;
		if (ssvec.x >0 && ssvec.x < 1 && ssvec.y > 0 && ssvec.y < 1){
			ao += texture(texMaps[4], ssvec).r;
			taow += 0.5;
		}
		ssvec = movedUV - sample_vec;
		if (ssvec.x >0 && ssvec.x < 1 && ssvec.y > 0 && ssvec.y < 1){
			ao += texture(texMaps[4], ssvec).r;
			taow += 0.5;
		}
	}
	ao = (ao + ambTex.r)/(1 + taow);
	ao = pow(ao, 1);
    //outColor = texColor * ao * ao;
	outColor = texColor * max((AMBIENT + overall_shade)*ao, 10 * seTex.y);
	if (length(vec2((inUV.x - 0.5)*(sceneData.screenData.x/sceneData.screenData.y), inUV.y - 0.5)) < 0.002){
		outColor += vec4(1,1,1,0);
	}
	//outColor = vec4(1,1,1,1);
}