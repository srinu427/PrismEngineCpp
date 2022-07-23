#version 460
#define MAX_NS_LIGHTS 30
#define MAX_PLIGHTS 8
#define MAX_DLIGHTS 8

struct GPULight {
	vec4 pos;
    vec4 color;
    vec4 dir;
    vec4 props;
    mat4 proj;
    mat4 viewproj;
    ivec4 flags;
};

layout(std140, set = 0, binding = 0) uniform LightBuffer { GPULight lights[MAX_NS_LIGHTS + MAX_PLIGHTS + MAX_DLIGHTS]; } lightBuffer;

layout(push_constant) uniform lpconst {
    ivec4 idx;
    mat4 viewproj;
} lightPC;

struct ObjectData{
	mat4 model;
};

layout(std140,set = 1, binding = 0) readonly buffer ObjectBuffer{
	ObjectData objects[];
} objectBuffer;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inTangent;
layout(location = 3) in vec3 inBitangent;
layout(location = 4) in vec3 inColor;
layout(location = 5) in vec2 inTexCoord;

layout(location = 0) out vec4 fragPos;
layout(location = 1) out vec4 lightPos;

void main() {
    lightPos = lightBuffer.lights[lightPC.idx.x].pos;
    fragPos = objectBuffer.objects[gl_BaseInstance].model * vec4(inPosition, 1.0f);
    vec4 cpos = lightPC.viewproj * lightBuffer.lights[lightPC.idx.x].viewproj * fragPos;
    gl_Position = cpos;
}