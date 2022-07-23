#version 460
#extension GL_KHR_vulkan_glsl: enable

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragTangent;
layout(location = 4) in vec3 fragBitangent;
layout(location = 5) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outPosition;
layout(location = 2) out vec4 outNormal;
layout(location = 3) out vec4 outSE;

layout(set = 0, binding = 0) uniform GPUCameraData {
	vec4 camPos;
    vec4 camDir;
	vec4 projprops;
    mat4 viewproj;
} camData;
layout(set = 2, binding = 0) uniform sampler2D texMaps[3];


float linearDepth(float depth)
{
	float z = depth * 2.0f - 1.0f; 
	return (2.0f * camData.projprops.x * camData.projprops.y) / (camData.projprops.y + camData.projprops.x - z * (camData.projprops.y - camData.projprops.x));	
}


void main() {
    vec4 texColor = texture(texMaps[0], fragTexCoord);
    vec4 normTex = texture(texMaps[1], fragTexCoord);
	vec4 seTex = texture(texMaps[2], fragTexCoord);
	
	normTex = normTex - vec4(0.5);
	normTex = 2 * normTex;

	vec4 normVal = vec4(0);

	normVal.x = (fragTangent.x * normTex.x) + (fragNormal.x * normTex.z) + (fragBitangent.x * normTex.y);
	normVal.y = (fragTangent.y * normTex.x) + (fragNormal.y * normTex.z) + (fragBitangent.y * normTex.y);
	normVal.z = (fragTangent.z * normTex.x) + (fragNormal.z * normTex.z) + (fragBitangent.z * normTex.y);

	normVal = normalize(normVal);
	
	normVal = normVal*0.5f + 0.5f;

    outColor = texColor;
    //outPosition = vec4(fragPosition, linearDepth(gl_FragCoord.z));
    outPosition = vec4(fragPosition, length(fragPosition.xyz - camData.camPos.xyz));
    outNormal = normVal;
    outSE = seTex;
}