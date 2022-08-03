#pragma once

#define VK_USE_PLATFORM_WIN32_KHR
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <array>
#include <optional>
#include <iostream>

#define PRISM_LIGHT_SHADOW_FLAG 0x01
#define PRISM_LIGHT_DIFFUSE_FLAG 0x02
#define PRISM_LIGHT_SPECULAR_FLAG 0x04
#define PRISM_LIGHT_EMISSIVE_FLAG 0x06

static VKAPI_ATTR VkBool32 VKAPI_CALL GPUDebugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

namespace Prism {
	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;
		std::optional<uint32_t> transferFamily;

		bool isComplete() {
			return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value();
		}
	};

	struct SwapChainSupportDetails {
		vk::SurfaceCapabilitiesKHR capabilities;
		std::vector<vk::SurfaceFormatKHR> formats;
		std::vector<vk::PresentModeKHR> presentModes;

		SwapChainSupportDetails(vk::PhysicalDevice device, vk::SurfaceKHR surface);
	};

	struct MeshPushConstants {
		int tidx;
		glm::vec4 data;
		glm::mat4 render_matrix;
	};

	struct Vertex {
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec3 tangent;
		glm::vec3 bitangent;
		glm::vec3 color;
		glm::vec2 texCoord;

		static vk::VertexInputBindingDescription getBindingDescription() {
			vk::VertexInputBindingDescription bindingDescription{ .binding = 0, .stride = sizeof(Vertex) , .inputRate = vk::VertexInputRate::eVertex };
			return bindingDescription;
		}

		static std::array<vk::VertexInputAttributeDescription, 6> getAttributeDescriptions() {
			std::array<vk::VertexInputAttributeDescription, 6> attributeDescriptions{};
			attributeDescriptions[0].binding = 0;
			attributeDescriptions[0].location = 0;
			attributeDescriptions[0].format = vk::Format::eR32G32B32A32Sfloat;
			attributeDescriptions[0].offset = offsetof(Vertex, pos);

			attributeDescriptions[1].binding = 0;
			attributeDescriptions[1].location = 1;
			attributeDescriptions[1].format = vk::Format::eR32G32B32A32Sfloat;
			attributeDescriptions[1].offset = offsetof(Vertex, normal);

			attributeDescriptions[2].binding = 0;
			attributeDescriptions[2].location = 2;
			attributeDescriptions[2].format = vk::Format::eR32G32B32A32Sfloat;
			attributeDescriptions[2].offset = offsetof(Vertex, tangent);

			attributeDescriptions[3].binding = 0;
			attributeDescriptions[3].location = 3;
			attributeDescriptions[3].format = vk::Format::eR32G32B32A32Sfloat;
			attributeDescriptions[3].offset = offsetof(Vertex, bitangent);

			attributeDescriptions[4].binding = 0;
			attributeDescriptions[4].location = 4;
			attributeDescriptions[4].format = vk::Format::eR32G32B32A32Sfloat;
			attributeDescriptions[4].offset = offsetof(Vertex, color);

			attributeDescriptions[5].binding = 0;
			attributeDescriptions[5].location = 5;
			attributeDescriptions[5].format = vk::Format::eR32G32Sfloat;
			attributeDescriptions[5].offset = offsetof(Vertex, texCoord);

			return attributeDescriptions;
		}

		bool operator==(const Vertex& other) const {
			return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal && tangent == other.tangent;
		}
	};

	class Instance : public vk::Instance {
	public:
		void setupDbgMessenger();
		void destroyDbgMessenger();
		Instance();
		Instance(const vk::Instance& vki);
	private:
		VkDebugUtilsMessengerEXT dbgMessenger;
	};

	class PhysicalDevice : public vk::PhysicalDevice {
	public:
		vk::PhysicalDeviceProperties phyDeviceProps;
		vk::PhysicalDeviceMemoryProperties phyDeviceMemProps;

		PhysicalDevice();
		PhysicalDevice(const vk::PhysicalDevice& pd);
		uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags reqProps);
	};

	struct GPUBuffer {
		vk::Buffer _buffer;
		vk::DeviceMemory _bufferMemory;
		vk::DescriptorSet _dSet;
		bool _has_dset = false;
	};

	struct Mesh {
		std::vector<Vertex> _vertices;
		std::vector<uint32_t> _indices;

		void add_vertices(std::vector<Vertex> verts);
		void make_cuboid(glm::vec3 center, glm::vec3 u, glm::vec3 v, float ulen, float vlen, float tlen);
		bool load_from_obj(const char* filename);
		glm::vec3 get_center();
		float get_bound_sphere_radius();
	};

	struct GPUImage {
		vk::Image _image;
		vk::DeviceMemory _imageMemory;
		vk::ImageView _imageView;
	};

	class Device : public vk::Device {
	public:
		void destroyGPUBuffer(GPUBuffer gbuff);
	};

	struct GPUSceneData {
		glm::vec4 fogColor; // w is for exponent
		glm::vec4 fogDistances; //x for min, y for max, zw unused.
		glm::vec4 ambientColor;
		glm::vec4 sunlightPosition;
		glm::vec4 sunlightDirection; //w for sun power
		glm::vec4 sunlightColor;
		glm::vec4 screenData;
	};

	struct GPUCameraData {
		glm::vec4 camPos;
		glm::vec4 camDir;
		glm::vec4 projprops;
		glm::mat4 viewproj;
	};

	struct GPULight {
		glm::vec4 pos = glm::vec4(0);
		glm::vec4 color = glm::vec4(0);
		glm::vec4 dir = glm::vec4(0);
		glm::vec4 props = glm::vec4(0);
		glm::mat4 proj = glm::mat4(1);
		glm::mat4 viewproj = glm::mat4(1);
		glm::ivec4 flags = glm::ivec4(0);

		void set_vp_mat(float fov = glm::radians(90.0f), float aspect = 1.0f, float near_plane = 1.0f, float far_plane = 1000.0f);
	};

	struct ObjectTextures {
		GPUImage color;
		GPUImage normal;
		GPUImage se;
		vk::DescriptorSet dset;
	};

	struct GPUObjectData {
		glm::mat4 model = glm::mat4{ 1.0f };
	};

	class RenderObject {
	public:
		std::string id;
		size_t logic_mgr_id = 0;
		Mesh* mesh;
		std::vector<GPUBuffer> _vertexBuffers;
		std::vector<GPUBuffer> _indexBuffers;
		vk::Sampler _textureSampler;
		ObjectTextures* texmaps;
		GPUObjectData uboData;
		bool renderable = true;
		bool shadowcasting = true;
	};

	struct GPUPushConstant {
		vk::PushConstantRange PCRange;
		void* PCData;
	};

	struct GPULightPC {
		glm::ivec4 idx = glm::ivec4(0);
		glm::mat4 viewproj = glm::mat4(1);
	};

	struct GPUPipeline {
		vk::Pipeline _pipeline;
		vk::PipelineLayout _pipelineLayout;
	};

	class CommandBuffer : public vk::CommandBuffer {
	public:
		CommandBuffer();
		CommandBuffer(const vk::CommandBuffer& cb);
		void drawRenderObjectMesh(RenderObject* robj, size_t obj_idx);
	};

	

	struct GPUFrameData {
		std::unordered_map<std::string, GPUBuffer> setBuffers;

		vk::CommandPool commandPool;
		Prism::CommandBuffer commandBuffer;

		GPUImage swapChainImage;
		GPUImage colorImage;
		GPUImage positionImage;
		GPUImage normalImage;
		GPUImage seImage;
		GPUImage ambientImage;
		GPUImage pShadowMapTemp;
		GPUImage dShadowMapTemp;
		std::vector<GPUImage> shadow_cube_maps;
		std::vector<GPUImage> shadow_dir_maps;

		vk::Framebuffer swapChainFrameBuffer;
		vk::Framebuffer gbufferFrameBuffer;
		vk::Framebuffer ambientFrameBuffer;
		vk::Framebuffer pShadowFrameBuffer;
		vk::Framebuffer dShadowFrameBuffer;

		vk::DescriptorSet shadow_cube_dset;
		vk::DescriptorSet shadow_dir_dset;
		vk::DescriptorSet positionImageDset;
		vk::DescriptorSet normalImageDset;
		vk::DescriptorSet finalComposeDset;

		vk::Semaphore presentSemaphore, renderSemaphore;
		vk::Fence renderFence;
	};
}

namespace std {
	template<> struct hash<Prism::Vertex> {
		size_t operator()(Prism::Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}