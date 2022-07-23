#version 460
#extension GL_KHR_vulkan_glsl: enable

#define AMBIENT_RAD 2.0

layout (set = 0, binding = 0) uniform sampler2D samplerPositionDepth;
layout (set = 1, binding = 0) uniform sampler2D samplerNormal;
layout (set = 2, binding = 0) uniform GPUCameraData {
	vec4 camPos;
    vec4 camDir;
	vec4 projprops;
    mat4 viewproj;
} camData;
layout (set = 3, binding = 0) uniform sampler2D ambientRVec;
layout (set = 4, binding = 0) uniform rKern{ vec4 data[64];} ambientKernel;

layout (location = 0) in vec2 inUV;

layout (location = 0) out float ambientShade;

void main() 
{
    vec2 mUV = inUV;
	vec3 fragPos = texture(samplerPositionDepth, inUV).rgb;
	vec3 normal = texture(samplerNormal, inUV).rgb;
	normal = normal - vec3(0.5);
	normal = 2 * normal;

	vec3 tangent = texture(ambientRVec, inUV * vec2(1280.0/4.0, 720.0/4.0)).xyz;
	tangent = normalize(tangent - (dot(tangent, normal) * normal));
	tangent = normalize(tangent - (dot(tangent, normal) * normal));
	vec3 bitangent = cross(normal, tangent);

	mat3 TBN = mat3(tangent, bitangent, normal);

	float light_prob = 0;
	float total_prob = 0;
	float bias = 0.05;

	//if (length(fragPos - camData.camPos.xyz) < AMBIENT_RAD){
	//	ambientShade = 1;
	//	return;
	//}

	for (int si = 0; si < 64; si+=1){
		vec3 sample_pos = fragPos + (TBN * ambientKernel.data[si].xyz * AMBIENT_RAD);

		vec4 sample_frag = camData.viewproj * vec4(sample_pos, 1);
		sample_frag.xyz /= sample_frag.w;
		sample_frag.xyz = sample_frag.xyz * 0.5f + 0.5f;
		sample_frag.y = 1 - sample_frag.y;

		if (sample_frag.z > 0 && sample_frag.w > 0 && sample_frag.x > 0 && sample_frag.x < 1 && sample_frag.y > 0 && sample_frag.y < 1){
			float sample_depth = length(sample_pos - camData.camPos.xyz);
			float map_depth = texture(samplerPositionDepth, sample_frag.xy).w;
			float depth_diff = sample_depth - map_depth;

			if (depth_diff >=  bias) {
				light_prob += smoothstep(0, 1, AMBIENT_RAD/abs(depth_diff));
			}
			total_prob += 1;
		}
	}

	ambientShade = pow(1 - (light_prob/total_prob), 2);

}