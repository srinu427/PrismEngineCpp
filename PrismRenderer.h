#pragma once

#include "ModelStructs.h"

#include <mutex>
#include <vector>

class PrismRenderer {
public:
	const size_t MAX_NS_LIGHTS = 30;
	const size_t MAX_POINT_LIGHTS = 8;
	const size_t MAX_DIRECTIONAL_LIGHTS = 8;

	vk::Extent2D swapChainExtent = { 1920, 1080 };
	bool framebufferResized = false;

	std::vector<Prism::RenderObject> renderObjects;
	std::vector<Prism::GPULight> lights;

	Prism::GPUSceneData currentScene;
	Prism::GPUCameraData currentCamera;

	void (*uboUpdateCallback) (float framedeltat, PrismRenderer* renderer, uint32_t frameNo);

	PrismRenderer(GLFWwindow* glfwWindow, void(*nextFrameCallback)(float framedeltat, PrismRenderer* renderer, uint32_t frameNo));
	void run();
	void addRenderObj(
		PrismModelData* lModelObj,
		size_t lModelIdx,
		bool include_in_final_render = true,
		bool include_in_shadow_map = true
	);
	void refreshMeshVB(PrismModelData* pmdl, int fno);
private:
#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif
	const size_t MAX_OBJECTS = 1000;

	vk::ClearValue COLOR_CLEAR_VAL_DEFAULT{ .color = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f }) };
	vk::ClearValue DEPTH_CLEAR_VAL_DEFAULT{ .depthStencil = vk::ClearDepthStencilValue(1.0f, 0) };

	std::vector<const char*> validationLayers;
	std::vector<const char*> deviceExtensions;

	GLFWwindow* window;
	Prism::Instance instance;
	Prism::Device device;
	Prism::PhysicalDevice physicalDevice;
	vk::PhysicalDeviceProperties phyDeviceProps;
	vk::PhysicalDeviceMemoryProperties phyDeviceMemProps;

	Prism::QueueFamilyIndices queueFamilyIndices;
	vk::Queue graphicsQueue, presentQueue, transferQueue;
	vk::SurfaceKHR surface;

	size_t MAX_FRAMES_IN_FLIGHT = 3;
	size_t currentFrame = 0;
	bool time_refresh = true;
	std::chrono::steady_clock::time_point startTime;
	std::chrono::steady_clock::time_point lastFrameTime;

	vk::SwapchainKHR swapChain;
	vk::Format swapChainImageFormat;
	std::vector<vk::Fence> imagesInFlight;

	Prism::GPUImage gbufferDepthImage;
	Prism::GPUImage dShadowDepthImage;
	Prism::GPUImage pShadowDepthImage;
	Prism::GPUImage shadowNoiseImage;
	Prism::GPUImage ambientNoiseImage;
	vk::DescriptorSet shadowNoiseImageDset;
	vk::DescriptorSet ambientNoiseImageDset;
	Prism::GPUBuffer shadowKernel;
	Prism::GPUBuffer ambientKernel;
	vk::Format depthFormat;

	
	vk::Extent2D plight_smap_extent = { 1 << 10, 1 << 10 };
	vk::Extent2D dlight_smap_extent = { 1 << 11, 1 << 11 };

	vk::DescriptorPool descriptorPool;
	std::unordered_map<std::string, vk::DescriptorSetLayout> dSetLayouts;

	vk::RenderPass gbufferRenderPass;
	vk::RenderPass shadowRenderPass;
	vk::RenderPass ambientRenderPass;
	vk::RenderPass finalRenderPass;

	std::vector<vk::ClearValue> gbufferClearValues = { COLOR_CLEAR_VAL_DEFAULT , COLOR_CLEAR_VAL_DEFAULT , COLOR_CLEAR_VAL_DEFAULT , COLOR_CLEAR_VAL_DEFAULT , DEPTH_CLEAR_VAL_DEFAULT };
	std::vector<vk::ClearValue> smapClearValues = { COLOR_CLEAR_VAL_DEFAULT , DEPTH_CLEAR_VAL_DEFAULT };
	std::vector<vk::ClearValue> fullscreenClearValues = { COLOR_CLEAR_VAL_DEFAULT };

	vk::CommandPool uploadCmdPool;

	std::unordered_map<std::string, vk::Sampler> texSamplers;
	std::vector<Prism::GPUFrameData> frameDatas;

	std::unordered_map<std::string, Prism::GPUPipeline> pipelines;
	std::unordered_map<std::string, Prism::ObjectTextures> objTextures;

	std::mutex spawn_mut;

	void initVulkan();
	void mainLoop();
	void cleanup();
	void drawFrame();
	void updateUBOs(size_t frameNo);
	void recreateSwapChain();
	void cleanupSwapChain(bool partial_cleanup);

	void getVkInstance();
	void createSurface();
	void getVkLogicalDevice();

	void createSwapChain();
	void createGBufferAttachmentImages();
	void createDSMapAttachmentImages();
	void createPSMapAttachmentImages();
	void createAmbientAttachmentImages();

	void createGBufferDepthImage();
	void createDSMapDepthImage();
	void createPSMapDepthImage();
	void createNoiseImages();

	void createTexSamplers();

	void createFrameCmdPools();
	void createUploadCmdPool();

	void createDescriptorPool();
	void addDsetLayout(std::string name, uint32_t binding, vk::DescriptorType dType, uint32_t dCount, vk::ShaderStageFlags stageFlag);
	void createDSetLayouts();
	void createGeneralDSets();
	void createSMapDSets();
	void createScreenSizeDSets();

	void createRenderPasses();
	void createGBufferFrameBuffers();
	void createDSMapFrameBuffers();
	void createPSMapFrameBuffers();
	void createAmbientFrameBuffers();
	void createFinalFrameBuffers();
	void createGBufferPipeline();
	void createDSMapPipeline();
	void createPSMapPipeline();
	void createAmbientPipeline();
	void createFinalPipeline();

	void createSyncObjects();

	void createRenderCmdBuffers();
	void updateRenderCmds(size_t frameNo);

	void addGBufferRenderCmds(size_t frameNo);
	void addDSMapRenderCmds(size_t frameNo);
	void addPSMapRenderCmds(size_t frameNo);
	void addAmbientRenderCmds(size_t frameNo);
	void addFinalRenderCmds(size_t frameNo);

	Prism::GPUImage create2DGPUImage(vk::Format format, vk::Extent3D extent, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::ImageLayout initLayout, vk::ImageAspectFlags aspectFlags, vk::MemoryPropertyFlags memFlag);
	Prism::GPUImage createCubeGPUImage(vk::Format format, vk::Extent3D extent, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::ImageLayout initLayout, vk::ImageAspectFlags aspectFlags, vk::MemoryPropertyFlags memFlag);
	void destroyGPUImage(Prism::GPUImage image, bool freeMem = true);

	Prism::GPUBuffer createGPUBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memProps);
	Prism::GPUBuffer createSetBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memProps, std::string dsetlayout_name, vk::DescriptorType dType);
	void destroyGPUBuffer(Prism::GPUBuffer gbuff);

	vk::DescriptorSet createGPUImageDSet(std::vector<Prism::GPUImage> images, vk::ImageLayout imglayout, std::string sampler_name, std::string dsetlayout_name);
	void CPUtoGPUBufferCopy(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize, Prism::GPUBuffer gbuff);
	Prism::CommandBuffer beginOneTimeCmds();
	void endOneTimeCmds(Prism::CommandBuffer cmdbuff);
	void copyDatatoImage(void* data, vk::DeviceSize dataSize, Prism::GPUImage image, vk::Extent3D imgExtent, vk::Offset3D imgOffset = { 0,0,0 });
	
	vk::ShaderModule loadShader(std::string shaderFilePath);
	Prism::GPUImage loadSingleTexture(std::string texPath, vk::Format imgFormat=vk::Format::eR8G8B8A8Srgb);
	Prism::ObjectTextures* loadObjTextures(
		std::string colorTexPath,
		std::string normalTexPath,
		std::string seTexPath,
		std::string texSamplerType
	);
	void destroyGPUPipeline(std::string pipeline_name);
};