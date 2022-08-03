#include "PrismRenderer.h"

#include <set>
#include <random>
#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

static vk::AccessFlags getAccessFlagsForLayout(vk::ImageLayout ilay) {
	switch (ilay)
	{
	case vk::ImageLayout::eUndefined:
		return vk::AccessFlagBits::eNone;
	case vk::ImageLayout::ePreinitialized:
		return vk::AccessFlagBits::eHostWrite;
	case vk::ImageLayout::eColorAttachmentOptimal:
		return vk::AccessFlagBits::eColorAttachmentWrite;
	case vk::ImageLayout::eDepthStencilAttachmentOptimal:
		return vk::AccessFlagBits::eDepthStencilAttachmentWrite;
	case vk::ImageLayout::eTransferSrcOptimal:
		return vk::AccessFlagBits::eTransferRead;
	case vk::ImageLayout::eTransferDstOptimal:
		return vk::AccessFlagBits::eTransferWrite;
	case vk::ImageLayout::eShaderReadOnlyOptimal:
		return vk::AccessFlagBits::eShaderRead;
	default:
		throw std::invalid_argument("unsupported layout transition!");
	}
	return vk::AccessFlagBits::eNone;
}

PrismRenderer::PrismRenderer(GLFWwindow* glfwWindow, void (*nextFrameCallback) (float framedeltat, PrismRenderer* renderer, uint32_t frameNo))
{
	window = glfwWindow;
	uboUpdateCallback = nextFrameCallback;
	lights.resize(MAX_NS_LIGHTS + MAX_DIRECTIONAL_LIGHTS + MAX_POINT_LIGHTS);
	plight_render_list.resize(MAX_POINT_LIGHTS * (MAX_OBJECTS + 1));
	dlight_render_list.resize(MAX_DIRECTIONAL_LIGHTS * (MAX_OBJECTS + 1));
	cam_render_list.resize(MAX_OBJECTS + 1);

	// Layers and Extensions needed
	validationLayers = {
	"VK_LAYER_KHRONOS_validation"
	};

	deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	"VK_KHR_maintenance1",
	"VK_KHR_shader_draw_parameters",
	"VK_EXT_shader_viewport_index_layer"
	};

	//renderer_tpool = new SimpleThreadPooler(RENDERER_THREADS);

	initVulkan();
}

void PrismRenderer::initVulkan()
{
	getVkInstance();
	createSurface();
	getVkLogicalDevice();

	createSwapChain();

	createUploadCmdPool();

	createGBufferAttachmentImages();
	createDSMapAttachmentImages();
	createPSMapAttachmentImages();
	createAmbientAttachmentImages();

	std::vector<vk::Format> depthFormatCandidates = { vk::Format::eD24UnormS8Uint };
	for (vk::Format dfc : depthFormatCandidates) {
		vk::FormatProperties props = physicalDevice.getFormatProperties(dfc);
		if ((props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)) {
			depthFormat = dfc;
			break;
		}
	}

	createGBufferDepthImage();
	createDSMapDepthImage();
	createPSMapDepthImage();
	createNoiseImages();
	createTexSamplers();

	createFrameCmdPools();

	createDescriptorPool();
	createDSetLayouts();
	createGeneralDSets();
	createSMapDSets();
	createScreenSizeDSets();

	createRenderPasses();
	createGBufferFrameBuffers();
	createDSMapFrameBuffers();
	createPSMapFrameBuffers();
	createAmbientFrameBuffers();
	createFinalFrameBuffers();

	createGBufferPipeline();
	createDSMapPipeline();
	createPSMapPipeline();
	createAmbientPipeline();
	createFinalPipeline();

	createRenderCmdBuffers();

	createSyncObjects();
}

void PrismRenderer::run()
{
	mainLoop();
	cleanup();
}

void PrismRenderer::cleanup()
{
	cleanupSwapChain(false);

	for (auto it : dSetLayouts) device.destroyDescriptorSetLayout(it.second);
	dSetLayouts.clear();
	device.destroyDescriptorPool(descriptorPool);
	for (auto it : texSamplers) device.destroySampler(it.second);
	texSamplers.clear();
	for (auto it : objTextures) {
		destroyGPUImage(it.second.color);
		destroyGPUImage(it.second.normal);
		destroyGPUImage(it.second.se);
	}
	objTextures.clear();
	for (auto it : pipelines) {
		device.destroyPipeline(it.second._pipeline);
		device.destroyPipelineLayout(it.second._pipelineLayout);
	}
	pipelines.clear();
	for (auto it : renderObjects) {
		for (size_t jt = 0; jt < MAX_FRAMES_IN_FLIGHT; jt++) {
			destroyGPUBuffer(it._vertexBuffers[jt]);
			destroyGPUBuffer(it._indexBuffers[jt]);
		}
	}
	renderObjects.clear();
	destroyGPUBuffer(ambientKernel);
	destroyGPUBuffer(shadowKernel);
	destroyGPUImage(ambientNoiseImage);
	destroyGPUImage(shadowNoiseImage);
	device.destroyCommandPool(uploadCmdPool);
	device.destroyRenderPass(shadowRenderPass);
	device.destroyRenderPass(gbufferRenderPass);
	device.destroyRenderPass(ambientRenderPass);
	device.destroyRenderPass(finalRenderPass);
	device.destroy();
	if (enableValidationLayers) instance.destroyDbgMessenger();
	instance.destroySurfaceKHR(surface);
	instance.destroy();
	glfwDestroyWindow(window);
	glfwTerminate();
}

Prism::GPUImage PrismRenderer::loadSingleTexture(std::string texPath, vk::Format imgFormat)
{
	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load(texPath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	vk::DeviceSize imageSize = texWidth * texHeight * 4;

	if (!pixels) throw std::runtime_error("failed to load texture image!");

	Prism::GPUImage tmp_gimg = create2DGPUImage(
		imgFormat,
		{ (uint32_t)texWidth, (uint32_t)texHeight, 1 },
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageAspectFlagBits::eColor,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);
	copyDatatoImage(
		pixels,
		imageSize,
		tmp_gimg,
		{ (uint32_t)texWidth, (uint32_t)texHeight, 1 }
	);
	stbi_image_free(pixels);
	Prism::CommandBuffer cmdbuffer = beginOneTimeCmds();
	vk::ImageMemoryBarrier imbarrier{
			.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
			.dstAccessMask = vk::AccessFlagBits::eShaderRead,
			.oldLayout = vk::ImageLayout::eTransferDstOptimal,
			.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			.image = tmp_gimg._image,
			.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	cmdbuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands,
		vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlagBits::eByRegion,
		{}, {}, { imbarrier }
	);
	endOneTimeCmds(cmdbuffer);
	return tmp_gimg;
}

Prism::ObjectTextures* PrismRenderer::loadObjTextures(
	std::string colorTexPath,
	std::string normalTexPath,
	std::string seTexPath,
	std::string texSamplerType
) {
	auto texit = objTextures.find(colorTexPath + "_" + normalTexPath + "_" + seTexPath);
	if (texit == objTextures.end()) {
		Prism::ObjectTextures gts;
		gts.color = loadSingleTexture(colorTexPath);
		gts.normal = loadSingleTexture(normalTexPath, vk::Format::eR8G8B8A8Unorm);
		gts.se = loadSingleTexture(seTexPath, vk::Format::eR8G8B8A8Unorm);
		gts.dset = createGPUImageDSet(
			{ gts.color, gts.normal, gts.se },
			vk::ImageLayout::eShaderReadOnlyOptimal,
			texSamplerType,
			"frag_sampler_3"
		);

		objTextures[colorTexPath + "_" + normalTexPath + "_" + seTexPath] = gts;
	}
	return &objTextures[colorTexPath + "_" + normalTexPath + "_" + seTexPath];
}

void PrismRenderer::addRenderObj(PrismModelData* lModelObj, size_t lModelIdx, bool include_in_final_render, bool include_in_shadow_map)
{
	spawn_mut.lock();
	Prism::RenderObject robj;
	robj.id = lModelObj->id;
	robj.logic_mgr_id = lModelIdx;
	robj.renderable = include_in_final_render;
	robj.shadowcasting = include_in_shadow_map;
	robj.uboData.model = lModelObj->initLRS.getTMatrix();

	robj._vertexBuffers.resize(3);
	robj._indexBuffers.resize(3);

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		robj._vertexBuffers[i] = createGPUBuffer(
			sizeof(Prism::Vertex) * lModelObj->mesh._vertices.size(),
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
		);
		CPUtoGPUBufferCopy(lModelObj->mesh._vertices.data(), 0, sizeof(Prism::Vertex) * lModelObj->mesh._vertices.size(), robj._vertexBuffers[i]);

		robj._indexBuffers[i] = createGPUBuffer(
			sizeof(uint32_t) * lModelObj->mesh._indices.size(),
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible
		);
		CPUtoGPUBufferCopy(lModelObj->mesh._indices.data(), 0, sizeof(uint32_t) * lModelObj->mesh._indices.size(), robj._indexBuffers[i]);
	}

	robj.mesh = &lModelObj->mesh;
	robj.texmaps = loadObjTextures(lModelObj->texFilePath, lModelObj->nmapFilePath, lModelObj->semapFilePath, "linear_filtered");
	lModelObj->renderer_id = renderObjects.size();
	lModelObj->pushed_to_renderer = true;
	renderObjects.push_back(robj);
	spawn_mut.unlock();
}

void PrismRenderer::refreshMeshVB(PrismModelData* pmdl, int fno)
{
	CPUtoGPUBufferCopy(pmdl->mesh._vertices.data(), 0, sizeof(Prism::Vertex) * pmdl->mesh._vertices.size(), renderObjects[pmdl->renderer_id]._vertexBuffers[fno]);
	CPUtoGPUBufferCopy(pmdl->mesh._indices.data(), 0, sizeof(uint32_t) * pmdl->mesh._indices.size(), renderObjects[pmdl->renderer_id]._indexBuffers[fno]);
}

void PrismRenderer::mainLoop()
{
	spawn_mut.lock();
	while (!glfwWindowShouldClose(window)) {
		drawFrame();
	}
	device.waitIdle();
	spawn_mut.unlock();
}

void PrismRenderer::drawFrame()
{
	device.waitForFences({ frameDatas[currentFrame].renderFence }, VK_TRUE, UINT64_MAX);

	vk::ResultValue<uint32_t> iqResult = device.acquireNextImageKHR(swapChain, UINT64_MAX, frameDatas[currentFrame].presentSemaphore, VK_NULL_HANDLE);
	uint32_t imageIndex = iqResult.value;
	imageIndex = imageIndex % MAX_FRAMES_IN_FLIGHT;

	if (iqResult.result == vk::Result::eErrorOutOfDateKHR) {
		recreateSwapChain();
		return;
	}
	else if (iqResult.result != vk::Result::eSuccess && iqResult.result != vk::Result::eSuboptimalKHR) {
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	// Check if a previous frame is using this image (i.e. there is its fence to wait on)
	if (imagesInFlight[imageIndex]) {
		device.waitForFences(1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
	}
	// Mark the image as now being in use by this frame
	imagesInFlight[imageIndex] = frameDatas[currentFrame].renderFence;

	spawn_mut.unlock();

	updateUBOs(imageIndex);

	//inputmgr.clearMOffset();

	spawn_mut.lock();

	updateRenderCmds(currentFrame);
	//delegate_gen_final_cmd_bufs(this, currentFrame);
	//renderer_tpool->add_task(&delegate_gen_final_cmd_bufs, this, currentFrame);;
	//renderer_tpool->wait_till_done();

	vk::Semaphore waitSemaphores[] = { frameDatas[currentFrame].presentSemaphore };
	vk::Semaphore signalSemaphores[] = { frameDatas[currentFrame].renderSemaphore };
	vk::PipelineStageFlags waitStages[1] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

	vk::SubmitInfo submitInfo{
		.waitSemaphoreCount = 1, .pWaitSemaphores = waitSemaphores,
		.pWaitDstStageMask = waitStages,
		.commandBufferCount = 1, .pCommandBuffers = &frameDatas[currentFrame].commandBuffer,
		.signalSemaphoreCount = 1, .pSignalSemaphores = signalSemaphores
	};

	device.resetFences(1, &frameDatas[currentFrame].renderFence);

	graphicsQueue.submit(1, &submitInfo, frameDatas[currentFrame].renderFence);

	vk::PresentInfoKHR presentInfo{
		.waitSemaphoreCount = 1, .pWaitSemaphores = signalSemaphores,
		.swapchainCount = 1, .pSwapchains = &swapChain,
		.pImageIndices = &imageIndex,
		.pResults = NULL
	};

	vk::Result prresult = presentQueue.presentKHR(&presentInfo);

	if (prresult == vk::Result::eErrorOutOfDateKHR || prresult == vk::Result::eSuboptimalKHR || framebufferResized) {
		framebufferResized = false;
		recreateSwapChain();
	}
	else if (prresult != vk::Result::eSuccess) {
		throw std::runtime_error("failed to present swap chain image!");
	}

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void PrismRenderer::updateUBOs(size_t frameNo)
{
	std::chrono::steady_clock::time_point currentFrameTime;

	float time = 0;
	float framedeltat = 0;

	if (time_refresh) {
		startTime = std::chrono::high_resolution_clock::now();
		time_refresh = false;
		lastFrameTime = startTime;
		currentFrameTime = startTime;
	}
	else {
		currentFrameTime = std::chrono::high_resolution_clock::now();
		time = std::chrono::duration<float, std::chrono::seconds::period>(currentFrameTime - startTime).count();
		framedeltat = std::chrono::duration<float, std::chrono::seconds::period>(currentFrameTime - lastFrameTime).count();
		lastFrameTime = currentFrameTime;
	}

	uboUpdateCallback(framedeltat, this, frameNo);

	

	for (size_t pli = 0; pli < MAX_POINT_LIGHTS; pli++) lights[MAX_NS_LIGHTS + pli].viewproj = glm::translate(glm::mat4(1.0f), -glm::vec3(lights[MAX_NS_LIGHTS + pli].pos));
	for (size_t pli = 0; pli < MAX_DIRECTIONAL_LIGHTS; pli++) {
		lights[MAX_NS_LIGHTS + MAX_POINT_LIGHTS + pli].viewproj = lights[MAX_NS_LIGHTS + MAX_POINT_LIGHTS + pli].proj * glm::lookAt(
			glm::vec3(lights[MAX_NS_LIGHTS + MAX_POINT_LIGHTS + pli].pos),
			glm::vec3(lights[MAX_NS_LIGHTS + MAX_POINT_LIGHTS + pli].pos + glm::normalize(lights[MAX_NS_LIGHTS + MAX_POINT_LIGHTS + pli].dir)),
			glm::vec3(0, 1, 0)
		);
	}
	size_t objCount = renderObjects.size();
	std::vector<Prism::GPUObjectData> objubo;
	for (size_t i = 0; i < objCount; i++) {
		objubo.push_back(renderObjects[i].uboData);
	}
	currentScene.screenData.x = swapChainExtent.width;
	currentScene.screenData.y = swapChainExtent.height;
	CPUtoGPUBufferCopy(&currentScene, 0, sizeof(Prism::GPUSceneData), frameDatas[frameNo].setBuffers["scene"]);
	CPUtoGPUBufferCopy(&currentCamera, 0, sizeof(Prism::GPUCameraData), frameDatas[frameNo].setBuffers["camera"]);
	CPUtoGPUBufferCopy(lights.data(), 0, lights.size() * sizeof(Prism::GPULight), frameDatas[frameNo].setBuffers["light"]);
	if (objCount > 0) CPUtoGPUBufferCopy(objubo.data(), 0, objCount * sizeof(Prism::GPUObjectData), frameDatas[frameNo].setBuffers["object"]);
}

void PrismRenderer::recreateSwapChain()
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(window, &width, &height);
	}
	device.waitIdle();

	Prism::SwapChainSupportDetails scdetails(physicalDevice, surface);
	uint32_t imageCount = scdetails.capabilities.minImageCount + 1;
	if (scdetails.capabilities.maxImageCount > 0 && imageCount > scdetails.capabilities.maxImageCount) {
		imageCount = scdetails.capabilities.maxImageCount;
	}
	bool partial_cleanup = frameDatas.size() == imageCount;
	cleanupSwapChain(partial_cleanup);

	createSwapChain();

	if (partial_cleanup) {
		createGBufferAttachmentImages();
		createGBufferDepthImage();

		createAmbientAttachmentImages();
		
		createGBufferFrameBuffers();
		createAmbientFrameBuffers();
		createFinalFrameBuffers();

		createGBufferPipeline();
		createAmbientPipeline();
		createFinalPipeline();

		createScreenSizeDSets();
	}
	else {
		imagesInFlight.resize(frameDatas.size());

		createGBufferAttachmentImages();
		createGBufferDepthImage();

		createDSMapAttachmentImages();
		createDSMapDepthImage();

		createPSMapAttachmentImages();
		createPSMapDepthImage();

		createAmbientAttachmentImages();

		createGBufferFrameBuffers();
		createDSMapFrameBuffers();
		createPSMapFrameBuffers();
		createAmbientFrameBuffers();
		createFinalFrameBuffers();

		createGBufferPipeline();
		createDSMapPipeline();
		createPSMapPipeline();
		createAmbientPipeline();
		createFinalPipeline();

		createRenderCmdBuffers();

		createSMapDSets();
		createScreenSizeDSets();
		createSyncObjects();
	}
}

void PrismRenderer::cleanupSwapChain(bool partial_cleanup)
{
	if (partial_cleanup) {
		destroyGPUImage(gbufferDepthImage);
		for (Prism::GPUFrameData fdata : frameDatas) {
			destroyGPUImage(fdata.colorImage);
			destroyGPUImage(fdata.positionImage);
			destroyGPUImage(fdata.normalImage);
			destroyGPUImage(fdata.seImage);
			
			destroyGPUImage(fdata.ambientImage);

			device.destroyImageView(fdata.swapChainImage._imageView);

			device.destroyFramebuffer(fdata.gbufferFrameBuffer);
			device.destroyFramebuffer(fdata.ambientFrameBuffer);
			device.destroyFramebuffer(fdata.swapChainFrameBuffer);

			destroyGPUPipeline("gbuffer");
			destroyGPUPipeline("ambient");
			destroyGPUPipeline("final");
		}
	}
	else {
		destroyGPUImage(gbufferDepthImage);
		destroyGPUImage(dShadowDepthImage);
		destroyGPUImage(pShadowDepthImage);
		destroyGPUPipeline("gbuffer");
		destroyGPUPipeline("dsmap");
		destroyGPUPipeline("psmap");
		destroyGPUPipeline("ambient");
		destroyGPUPipeline("final");

		for (Prism::GPUFrameData fdata : frameDatas) {
			destroyGPUImage(fdata.colorImage);
			destroyGPUImage(fdata.positionImage);
			destroyGPUImage(fdata.normalImage);
			destroyGPUImage(fdata.seImage);
			
			destroyGPUImage(fdata.dShadowMapTemp);
			for (Prism::GPUImage t : fdata.shadow_dir_maps) destroyGPUImage(t);
			fdata.shadow_dir_maps.clear();
			
			destroyGPUImage(fdata.pShadowMapTemp);
			for (Prism::GPUImage t : fdata.shadow_cube_maps) destroyGPUImage(t);
			fdata.shadow_cube_maps.clear();

			destroyGPUImage(fdata.ambientImage);

			device.destroyImageView(fdata.swapChainImage._imageView);

			device.destroyFramebuffer(fdata.gbufferFrameBuffer);
			device.destroyFramebuffer(fdata.dShadowFrameBuffer);
			device.destroyFramebuffer(fdata.pShadowFrameBuffer);
			device.destroyFramebuffer(fdata.ambientFrameBuffer);
			device.destroyFramebuffer(fdata.swapChainFrameBuffer);

			device.freeCommandBuffers(fdata.commandPool, { fdata.commandBuffer });

			device.destroySemaphore(fdata.renderSemaphore);
			device.destroySemaphore(fdata.presentSemaphore);
			device.destroyFence(fdata.renderFence);
			imagesInFlight.clear();

			device.destroyCommandPool(fdata.commandPool);

			for (auto t : fdata.setBuffers) destroyGPUBuffer(t.second);
			fdata.setBuffers.clear();
		}
	}
	device.destroySwapchainKHR(swapChain);
}

void PrismRenderer::getVkInstance()
{
	//Struct with app info
	vk::ApplicationInfo appInfo{
		.pApplicationName = "Prism App",
		.applicationVersion = 104,
		.pEngineName = "Prism Engine",
		.engineVersion = 104,
		.apiVersion = VK_API_VERSION_1_2
	};

	instance = vkutils::createVKInstance(
		appInfo,
		enableValidationLayers,
		validationLayers
	);
}

void PrismRenderer::createSurface() {
	if (glfwCreateWindowSurface(instance, window, NULL, reinterpret_cast<VkSurfaceKHR*>(&surface)) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface!");
	}
}

void PrismRenderer::getVkLogicalDevice()
{
	physicalDevice = vkutils::pickPhysicalDevice(instance, surface, deviceExtensions, true);
	phyDeviceProps = physicalDevice.getProperties();
	phyDeviceMemProps = physicalDevice.getMemoryProperties();
	queueFamilyIndices = vkutils::findQueueFamilies(physicalDevice, surface);

	//Struct to create required Queues
	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = { queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.presentFamily.value() , queueFamilyIndices.transferFamily.value() };

	float queuePriority = 1.0f;
	for (uint32_t queueFamily : uniqueQueueFamilies) {
		vk::DeviceQueueCreateInfo queueCreateInfo{
			.queueFamilyIndex = queueFamily,
			.queueCount = 1,
			.pQueuePriorities = &queuePriority
		};
		queueCreateInfos.push_back(queueCreateInfo);
	}

	//Struct to create logical device
	vk::PhysicalDeviceFeatures deviceFeatures{
		.samplerAnisotropy = VK_TRUE
	};

	vk::PhysicalDeviceVulkan11Features deviceFeatures11{
		.shaderDrawParameters = VK_TRUE
	};

	vk::DeviceCreateInfo createInfo{
		.pNext = &deviceFeatures11,
		.queueCreateInfoCount = uint32_t(queueCreateInfos.size()),
		.pQueueCreateInfos = queueCreateInfos.data(),
		.enabledExtensionCount = uint32_t(deviceExtensions.size()),
		.ppEnabledExtensionNames = deviceExtensions.data(),
		.pEnabledFeatures = &deviceFeatures
	};

	if (enableValidationLayers) {
		createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}
	//Creating the logical device
	device = Prism::Device(physicalDevice.createDevice(createInfo));

	//Get queue handles
	graphicsQueue = device.getQueue(queueFamilyIndices.graphicsFamily.value(), 0);
	presentQueue = device.getQueue(queueFamilyIndices.presentFamily.value(), 0);
	transferQueue = device.getQueue(queueFamilyIndices.transferFamily.value(), 0);
}

void PrismRenderer::createSwapChain()
{
	Prism::SwapChainSupportDetails swapChainSupport(physicalDevice, surface);
	vk::SurfaceFormatKHR surfaceFormat = vkutils::chooseSwapSurfaceFormat(swapChainSupport.formats);
	vk::PresentModeKHR presentMode = vkutils::chooseSwapPresentMode(swapChainSupport.presentModes);
	vk::Extent2D extent = vkutils::chooseSwapExtent(window, swapChainSupport.capabilities);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	//Struct to create swap chain
	vk::SwapchainCreateInfoKHR createInfo{
		.surface = surface,
		.minImageCount = imageCount,
		.imageFormat = surfaceFormat.format,
		.imageColorSpace = surfaceFormat.colorSpace,
		.imageExtent = extent,
		.imageArrayLayers = 1,
		.imageUsage = vk::ImageUsageFlagBits::eColorAttachment
	};

	uint32_t qfindices[] = { queueFamilyIndices.graphicsFamily.value(), queueFamilyIndices.presentFamily.value(), queueFamilyIndices.transferFamily.value() };

	if (queueFamilyIndices.graphicsFamily != queueFamilyIndices.presentFamily && queueFamilyIndices.graphicsFamily != queueFamilyIndices.transferFamily) {
		createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
		createInfo.queueFamilyIndexCount = 3;
		createInfo.pQueueFamilyIndices = qfindices;
	}
	else {
		createInfo.imageSharingMode = vk::SharingMode::eExclusive;
		createInfo.queueFamilyIndexCount = 0; // Optional
		createInfo.pQueueFamilyIndices = NULL; // Optional
	}

	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;

	//Create swapchain
	swapChain = device.createSwapchainKHR(createInfo);

	MAX_FRAMES_IN_FLIGHT = imageCount;
	frameDatas.resize(imageCount);

	std::vector<vk::Image> swapChainImages = device.getSwapchainImagesKHR(swapChain);

	swapChainImageFormat = surfaceFormat.format;
	swapChainExtent = extent;

	vk::ImageViewCreateInfo iviewinfo{
		.viewType = vk::ImageViewType::e2D,
		.format = swapChainImageFormat,
		.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
	};

	for (size_t i = 0; i < imageCount; i++) {
		frameDatas[i].swapChainImage._image = swapChainImages[i];
		iviewinfo.image = swapChainImages[i];
		frameDatas[i].swapChainImage._imageView = device.createImageView(iviewinfo);
	}
}

void PrismRenderer::createFrameCmdPools()
{
	for (int i = 0; i < frameDatas.size(); i++) {
		vk::CommandPoolCreateInfo poolInfo{
			.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
		};
		frameDatas[i].commandPool = device.createCommandPool(poolInfo);
	}
}

void PrismRenderer::createUploadCmdPool()
{
	vk::CommandPoolCreateInfo poolInfo{ .queueFamilyIndex = queueFamilyIndices.transferFamily.value() };
	uploadCmdPool = device.createCommandPool(poolInfo);
}

void PrismRenderer::createDescriptorPool() {
	std::vector<vk::DescriptorPoolSize> poolSizes =
	{
		{ vk::DescriptorType::eUniformBuffer, 200 },
		{ vk::DescriptorType::eUniformBufferDynamic, 200 },
		{ vk::DescriptorType::eStorageBuffer, 200 },
		{ vk::DescriptorType::eCombinedImageSampler, 200 }
	};

	vk::DescriptorPoolCreateInfo poolInfo{
		.maxSets = 400,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};

	descriptorPool = device.createDescriptorPool(poolInfo);
}

void PrismRenderer::addDsetLayout(std::string name, uint32_t binding, vk::DescriptorType dType, uint32_t dCount, vk::ShaderStageFlags stageFlag)
{
	vk::DescriptorSetLayoutBinding uboLayoutBinding{
		.binding = binding,
		.descriptorType = dType,
		.descriptorCount = dCount,
		.stageFlags = stageFlag
	};
	vk::DescriptorSetLayoutCreateInfo ubolayoutInfo{ .bindingCount = 1, .pBindings = &uboLayoutBinding };
	dSetLayouts[name] = device.createDescriptorSetLayout(ubolayoutInfo);

}

void PrismRenderer::createDSetLayouts()
{
	addDsetLayout("vert_uniform", 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);
	addDsetLayout("vert_storage", 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex);
	addDsetLayout("frag_uniform", 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment);
	addDsetLayout("frag_sampler_1", 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);
	addDsetLayout("frag_sampler_3", 0, vk::DescriptorType::eCombinedImageSampler, 3, vk::ShaderStageFlagBits::eFragment);
	addDsetLayout("frag_sampler_5", 0, vk::DescriptorType::eCombinedImageSampler, 5, vk::ShaderStageFlagBits::eFragment);
	addDsetLayout("vert_frag_uniform", 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex);
	addDsetLayout("frag_plight_sampler", 0, vk::DescriptorType::eCombinedImageSampler, uint32_t(MAX_POINT_LIGHTS), vk::ShaderStageFlagBits::eFragment);
	addDsetLayout("frag_dlight_sampler", 0, vk::DescriptorType::eCombinedImageSampler, uint32_t(MAX_DIRECTIONAL_LIGHTS), vk::ShaderStageFlagBits::eFragment);
}

Prism::GPUBuffer PrismRenderer::createGPUBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memProps)
{
	Prism::GPUBuffer gbuff;

	vk::BufferCreateInfo buffInfo{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
	gbuff._buffer = device.createBuffer(buffInfo);

	vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(gbuff._buffer);
	vk::MemoryAllocateInfo memAllocInfo{ .allocationSize = memReqs.size, .memoryTypeIndex = physicalDevice.findMemoryType(memReqs.memoryTypeBits, memProps) };
	gbuff._bufferMemory = device.allocateMemory(memAllocInfo);
	device.bindBufferMemory(gbuff._buffer, gbuff._bufferMemory, 0);

	gbuff._has_dset = false;

	return gbuff;
}

Prism::GPUBuffer PrismRenderer::createSetBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memProps, std::string dsetlayout_name, vk::DescriptorType dType)
{
	Prism::GPUBuffer gbuff;

	vk::BufferCreateInfo buffInfo{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
	gbuff._buffer = device.createBuffer(buffInfo);

	vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(gbuff._buffer);
	vk::MemoryAllocateInfo memAllocInfo{ .allocationSize = memReqs.size, .memoryTypeIndex = physicalDevice.findMemoryType(memReqs.memoryTypeBits, memProps) };
	gbuff._bufferMemory = device.allocateMemory(memAllocInfo);
	device.bindBufferMemory(gbuff._buffer, gbuff._bufferMemory, 0);

	vk::DescriptorSetAllocateInfo dsetAllocInfo{ .descriptorPool = descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &dSetLayouts[dsetlayout_name] };
	gbuff._dSet = device.allocateDescriptorSets(dsetAllocInfo)[0];
	vk::DescriptorBufferInfo dBufferInfo{ .buffer = gbuff._buffer, .offset = 0, .range = size };
	vk::WriteDescriptorSet dSetWrite{
		.dstSet = gbuff._dSet, .dstBinding =0, .dstArrayElement =0,
		.descriptorCount = 1,
		.descriptorType = dType,
		.pBufferInfo = &dBufferInfo
	};
	device.updateDescriptorSets({ dSetWrite }, NULL);

	gbuff._has_dset = true;

	return gbuff;
}

void PrismRenderer::destroyGPUBuffer(Prism::GPUBuffer gbuff)
{
	device.destroyBuffer(gbuff._buffer);
	device.freeMemory(gbuff._bufferMemory);
	//if (gbuff._has_dset) device.freeDescriptorSets(descriptorPool, { gbuff._dSet });
}

vk::DescriptorSet PrismRenderer::createGPUImageDSet(std::vector<Prism::GPUImage> images, vk::ImageLayout imglayout, std::string sampler_name, std::string dsetlayout_name)
{
	vk::DescriptorSet dset;
	vk::DescriptorSetAllocateInfo dsetAllocInfo{ .descriptorPool = descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &dSetLayouts[dsetlayout_name] };
	dset = device.allocateDescriptorSets(dsetAllocInfo)[0];

	std::vector<vk::DescriptorImageInfo> imageInfos = std::vector<vk::DescriptorImageInfo>(images.size());
	vk::Sampler isampler = texSamplers[sampler_name];
	for (size_t i = 0; i < images.size(); i++) {
		imageInfos[i].sampler = isampler;
		imageInfos[i].imageView = images[i]._imageView;
		imageInfos[i].imageLayout = imglayout;
	}

	vk::WriteDescriptorSet dSetWrite{
		.dstSet = dset, .dstBinding = 0, .dstArrayElement = 0,
		.descriptorCount = uint32_t(images.size()),
		.descriptorType = vk::DescriptorType::eCombinedImageSampler,
		.pImageInfo = imageInfos.data()
	};
	device.updateDescriptorSets({ dSetWrite }, NULL);
	return dset;
}

void PrismRenderer::createGeneralDSets()
{
	std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0]
	std::default_random_engine generator;

	std::vector<glm::vec4> ssaoKernel;
	for (unsigned int i = 0; i < 64; ++i)
	{
		glm::vec4 sample(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator),
			0.0f
		);
		sample = glm::normalize(sample);
		//sample *= randomFloats(generator);
		float scale = (float)i / 64.0;
		scale = 0.1f + (scale * scale * 0.9f);
		sample *= scale;
		ssaoKernel.push_back(sample);
	}
	ambientKernel = createSetBuffer(
		sizeof(glm::vec4) * ssaoKernel.size(),
		vk::BufferUsageFlagBits::eUniformBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		"frag_uniform",
		vk::DescriptorType::eUniformBuffer
	);
	CPUtoGPUBufferCopy(ssaoKernel.data(), 0, sizeof(glm::vec4) * ssaoKernel.size(), ambientKernel);
	ambientNoiseImageDset = createGPUImageDSet({ ambientNoiseImage }, vk::ImageLayout::eShaderReadOnlyOptimal, "linear_unfiltered", "frag_sampler_1");

	std::vector<glm::vec4> shadowKernelData;
	for (unsigned int i = 0; i < 64; ++i)
	{
		glm::vec4 sample(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) * 2.0 - 1.0,
			0.0f,
			0.0f
		);
		sample = glm::normalize(sample);
		sample *= randomFloats(generator);
		//float scale = (float)i / 64.0;
		//scale = 0.1f + (scale * scale * 0.9f);
		//sample *= scale;
		shadowKernelData.push_back(sample);
	}

	shadowKernel = createSetBuffer(
		sizeof(glm::vec4) * shadowKernelData.size(),
		vk::BufferUsageFlagBits::eUniformBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		"frag_uniform",
		vk::DescriptorType::eUniformBuffer
	);
	CPUtoGPUBufferCopy(shadowKernelData.data(), 0, sizeof(glm::vec4) * shadowKernelData.size(), shadowKernel);
	shadowNoiseImageDset = createGPUImageDSet({ shadowNoiseImage }, vk::ImageLayout::eShaderReadOnlyOptimal, "linear_unfiltered", "frag_sampler_1");

	for (size_t i = 0; i < frameDatas.size(); i++) {
		//GBuffer and SMap dsets
		frameDatas[i].setBuffers["scene"] = createSetBuffer(
			sizeof(Prism::GPUSceneData),
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			"frag_uniform",
			vk::DescriptorType::eUniformBuffer
		);
		frameDatas[i].setBuffers["light"] = createSetBuffer(
			sizeof(Prism::GPULight) * (MAX_NS_LIGHTS + MAX_DIRECTIONAL_LIGHTS + MAX_POINT_LIGHTS),
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			"vert_frag_uniform",
			vk::DescriptorType::eUniformBuffer
		);
		frameDatas[i].setBuffers["camera"] = createSetBuffer(
			sizeof(Prism::GPUCameraData),
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			"vert_frag_uniform",
			vk::DescriptorType::eUniformBuffer
		);
		frameDatas[i].setBuffers["object"] = createSetBuffer(
			sizeof(Prism::GPUObjectData) * MAX_OBJECTS,
			vk::BufferUsageFlagBits::eStorageBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			"vert_storage",
			vk::DescriptorType::eStorageBuffer
		);
	}
}

void PrismRenderer::createSMapDSets()
{
	for (size_t i = 0; i < frameDatas.size(); i++) {
		frameDatas[i].shadow_dir_dset = createGPUImageDSet(
			frameDatas[i].shadow_dir_maps,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			"linear_clamped_unfiltered",
			"frag_dlight_sampler"
		);
		frameDatas[i].shadow_cube_dset = createGPUImageDSet(
			frameDatas[i].shadow_cube_maps,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			"linear_clamped_unfiltered",
			"frag_plight_sampler"
		);
	}
}

void PrismRenderer::createScreenSizeDSets()
{
	for (size_t i = 0; i < frameDatas.size(); i++) {
		frameDatas[i].positionImageDset = createGPUImageDSet(
			{ frameDatas[i].positionImage },
			vk::ImageLayout::eShaderReadOnlyOptimal,
			"linear_clamped_unfiltered",
			"frag_sampler_1"
		);
		frameDatas[i].normalImageDset = createGPUImageDSet(
			{ frameDatas[i].normalImage },
			vk::ImageLayout::eShaderReadOnlyOptimal,
			"linear_clamped_unfiltered",
			"frag_sampler_1"
		);
		frameDatas[i].finalComposeDset = createGPUImageDSet(
			{ frameDatas[i].colorImage, frameDatas[i].positionImage, frameDatas[i].normalImage, frameDatas[i].seImage, frameDatas[i].ambientImage },
			vk::ImageLayout::eShaderReadOnlyOptimal,
			"linear_clamped_unfiltered",
			"frag_sampler_5"
		);
	}
}

void PrismRenderer::createRenderPasses()
{
	std::vector<vk::AttachmentDescription> attachments;
	std::vector<vk::AttachmentReference> attachmentRefs;
	std::vector<vk::SubpassDependency> subpassDependencies;

	for (size_t ai = 0; ai < 4; ai++) {
		vk::AttachmentDescription tadesc{
			.flags = vk::AttachmentDescriptionFlagBits::eMayAlias,
			.samples = vk::SampleCountFlagBits::e1,
			.loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
			.stencilLoadOp = vk::AttachmentLoadOp::eDontCare, .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
			.initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal, .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal
		};
		vk::AttachmentReference taref{ .attachment = uint32_t(ai), .layout = vk::ImageLayout::eColorAttachmentOptimal };
		attachments.push_back(tadesc);
		attachmentRefs.push_back(taref);
	}
	attachments[0].format = vk::Format::eR8G8B8A8Unorm;
	attachments[1].format = vk::Format::eR32G32B32A32Sfloat;
	attachments[2].format = vk::Format::eR8G8B8A8Unorm;
	attachments[3].format = vk::Format::eR8G8B8A8Unorm;

	vk::AttachmentDescription depthAttachment{
		.format = depthFormat,
		.samples = vk::SampleCountFlagBits::e1,
		.loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eDontCare,
		.stencilLoadOp = vk::AttachmentLoadOp::eDontCare, .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		.initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal, .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
	};
	vk::AttachmentReference depthAttachmentRef{ .attachment = 4, .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal };
	attachments.push_back(depthAttachment);
	attachmentRefs.push_back(depthAttachmentRef);

	vk::SubpassDescription subpass{
		.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		.colorAttachmentCount = 4,
		.pColorAttachments = attachmentRefs.data(),
		.pDepthStencilAttachment = &attachmentRefs[4]
	};

	vk::SubpassDependency tsdep1{
		.srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
		.srcStageMask = vk::PipelineStageFlagBits::eFragmentShader, .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
		.srcAccessMask = vk::AccessFlagBits::eShaderRead, .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		.dependencyFlags = vk::DependencyFlagBits::eByRegion
	};
	subpassDependencies.push_back(tsdep1);
	vk::SubpassDependency tsdep2{
		.srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
		.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput, .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
		.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .dstAccessMask = vk::AccessFlagBits::eShaderRead,
		.dependencyFlags = vk::DependencyFlagBits::eByRegion
	};
	subpassDependencies.push_back(tsdep2);

	vk::RenderPassCreateInfo renderPassInfo{
		.attachmentCount = uint32_t(attachments.size()),
		.pAttachments = attachments.data(),
		.subpassCount = 1,
		.pSubpasses = &subpass,
		.dependencyCount = uint32_t(subpassDependencies.size()),
		.pDependencies = subpassDependencies.data()
	};
	gbufferRenderPass = device.createRenderPass(renderPassInfo);

	attachments.clear();
	attachmentRefs.clear();
	subpassDependencies.clear();

	//SMap RenderPass
	for (size_t ai = 0; ai < 1; ai++) {
		vk::AttachmentDescription tadesc{
			.format = vk::Format::eR32Sfloat,
			.samples = vk::SampleCountFlagBits::e1,
			.loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
			.stencilLoadOp = vk::AttachmentLoadOp::eDontCare, .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
			.initialLayout = vk::ImageLayout::eTransferSrcOptimal, .finalLayout = vk::ImageLayout::eTransferSrcOptimal
		};
		vk::AttachmentReference taref{ .attachment = uint32_t(ai), .layout = vk::ImageLayout::eColorAttachmentOptimal };
		attachments.push_back(tadesc);
		attachmentRefs.push_back(taref);
	}

	depthAttachment = {
		.format = depthFormat,
		.samples = vk::SampleCountFlagBits::e1,
		.loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eDontCare,
		.stencilLoadOp = vk::AttachmentLoadOp::eDontCare, .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
		.initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal, .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
	};
	depthAttachmentRef = { .attachment = 1, .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal };
	attachments.push_back(depthAttachment);
	attachmentRefs.push_back(depthAttachmentRef);

	subpass = {
		.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		.colorAttachmentCount = 1,
		.pColorAttachments = attachmentRefs.data(),
		.pDepthStencilAttachment = &attachmentRefs[1]
	};

	tsdep1 = {
		.srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
		.srcStageMask = vk::PipelineStageFlagBits::eFragmentShader, .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
		.srcAccessMask = vk::AccessFlagBits::eShaderRead, .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		.dependencyFlags = vk::DependencyFlagBits::eByRegion
	};
	subpassDependencies.push_back(tsdep1);
	tsdep2 = {
		.srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
		.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput, .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
		.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .dstAccessMask = vk::AccessFlagBits::eShaderRead,
		.dependencyFlags = vk::DependencyFlagBits::eByRegion
	};
	subpassDependencies.push_back(tsdep2);

	renderPassInfo = {
		.attachmentCount = uint32_t(attachments.size()),
		.pAttachments = attachments.data(),
		.subpassCount = 1,
		.pSubpasses = &subpass,
		.dependencyCount = uint32_t(subpassDependencies.size()),
		.pDependencies = subpassDependencies.data()
	};
	shadowRenderPass = device.createRenderPass(renderPassInfo);

	attachments.clear();
	attachmentRefs.clear();
	subpassDependencies.clear();

	//Ambient RenderPass
	for (size_t ai = 0; ai < 1; ai++) {
		vk::AttachmentDescription tadesc{
			.format = vk::Format::eR8Unorm,
			.samples = vk::SampleCountFlagBits::e1,
			.loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
			.stencilLoadOp = vk::AttachmentLoadOp::eDontCare, .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
			.initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal, .finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal
		};
		vk::AttachmentReference taref{ .attachment = uint32_t(ai), .layout = vk::ImageLayout::eColorAttachmentOptimal };
		attachments.push_back(tadesc);
		attachmentRefs.push_back(taref);
	}

	subpass = {
		.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		.colorAttachmentCount = 1,
		.pColorAttachments = attachmentRefs.data(),
		.pDepthStencilAttachment = NULL
	};

	tsdep1 = {
		.srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
		.srcStageMask = vk::PipelineStageFlagBits::eFragmentShader, .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
		.srcAccessMask = vk::AccessFlagBits::eShaderRead, .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		.dependencyFlags = vk::DependencyFlagBits::eByRegion
	};
	subpassDependencies.push_back(tsdep1);
	tsdep2 = {
		.srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
		.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput, .dstStageMask = vk::PipelineStageFlagBits::eFragmentShader,
		.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .dstAccessMask = vk::AccessFlagBits::eShaderRead,
		.dependencyFlags = vk::DependencyFlagBits::eByRegion
	};
	subpassDependencies.push_back(tsdep2);

	renderPassInfo = {
		.attachmentCount = uint32_t(attachments.size()),
		.pAttachments = attachments.data(),
		.subpassCount = 1,
		.pSubpasses = &subpass,
		.dependencyCount = uint32_t(subpassDependencies.size()),
		.pDependencies = subpassDependencies.data()
	};
	ambientRenderPass = device.createRenderPass(renderPassInfo);

	attachments.clear();
	attachmentRefs.clear();
	subpassDependencies.clear();

	//Final RenderPass
	for (size_t ai = 0; ai < 1; ai++) {
		vk::AttachmentDescription tadesc{
			.format = swapChainImageFormat,
			.samples = vk::SampleCountFlagBits::e1,
			.loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
			.stencilLoadOp = vk::AttachmentLoadOp::eDontCare, .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
			.initialLayout = vk::ImageLayout::eUndefined, .finalLayout = vk::ImageLayout::ePresentSrcKHR
		};
		vk::AttachmentReference taref{ .attachment = uint32_t(ai), .layout = vk::ImageLayout::eColorAttachmentOptimal };
		attachments.push_back(tadesc);
		attachmentRefs.push_back(taref);
	}

	subpass = {
		.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		.colorAttachmentCount = 1,
		.pColorAttachments = attachmentRefs.data(),
		.pDepthStencilAttachment = NULL
	};

	tsdep1 = {
		.srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0,
		.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests, .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
		.srcAccessMask = vk::AccessFlagBits::eNone, .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
		.dependencyFlags = vk::DependencyFlagBits::eByRegion
	};
	subpassDependencies.push_back(tsdep1);
	tsdep2 = {
		.srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL,
		.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput, .dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe,
		.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .dstAccessMask = vk::AccessFlagBits::eNone,
		.dependencyFlags = vk::DependencyFlagBits::eByRegion
	};
	subpassDependencies.push_back(tsdep2);

	renderPassInfo = {
		.attachmentCount = uint32_t(attachments.size()),
		.pAttachments = attachments.data(),
		.subpassCount = 1,
		.pSubpasses = &subpass,
		.dependencyCount = uint32_t(subpassDependencies.size()),
		.pDependencies = subpassDependencies.data()
	};
	finalRenderPass = device.createRenderPass(renderPassInfo);

	attachments.clear();
	attachmentRefs.clear();
	subpassDependencies.clear();
}

void PrismRenderer::createGBufferFrameBuffers()
{
	for (size_t i = 0; i < frameDatas.size(); i++) {
		std::vector<vk::ImageView> attachments;
		//GBuffer RP
		attachments = { frameDatas[i].colorImage._imageView, frameDatas[i].positionImage._imageView, frameDatas[i].normalImage._imageView, frameDatas[i].seImage._imageView, gbufferDepthImage._imageView };
		vk::FramebufferCreateInfo fbcreateinfo{
			.renderPass = gbufferRenderPass,
			.attachmentCount = uint32_t(attachments.size()), .pAttachments = attachments.data(),
			.width = swapChainExtent.width, .height = swapChainExtent.height, .layers = 1
		};
		frameDatas[i].gbufferFrameBuffer = device.createFramebuffer(fbcreateinfo);
	}
}

void PrismRenderer::createDSMapFrameBuffers()
{
	for (size_t i = 0; i < frameDatas.size(); i++) {
		std::vector<vk::ImageView> attachments;
		//DSMap RP
		attachments = { frameDatas[i].dShadowMapTemp._imageView, dShadowDepthImage._imageView };
		vk::FramebufferCreateInfo fbcreateinfo{
			.renderPass = shadowRenderPass,
			.attachmentCount = uint32_t(attachments.size()), .pAttachments = attachments.data(),
			.width = dlight_smap_extent.width, .height = dlight_smap_extent.height, .layers = 1
		};
		frameDatas[i].dShadowFrameBuffer = device.createFramebuffer(fbcreateinfo);
	}
}

void PrismRenderer::createPSMapFrameBuffers()
{
	for (size_t i = 0; i < frameDatas.size(); i++) {
		std::vector<vk::ImageView> attachments;
		//PSMap RP
		attachments = { frameDatas[i].pShadowMapTemp._imageView, pShadowDepthImage._imageView };
		vk::FramebufferCreateInfo fbcreateinfo{
			.renderPass = shadowRenderPass,
			.attachmentCount = uint32_t(attachments.size()), .pAttachments = attachments.data(),
			.width = plight_smap_extent.width, .height = plight_smap_extent.height, .layers = 1
		};
		frameDatas[i].pShadowFrameBuffer = device.createFramebuffer(fbcreateinfo);
	}
}

void PrismRenderer::createAmbientFrameBuffers()
{
	for (size_t i = 0; i < frameDatas.size(); i++) {
		std::vector<vk::ImageView> attachments;
		//Ambient RP
		attachments = { frameDatas[i].ambientImage._imageView };
		vk::FramebufferCreateInfo fbcreateinfo{
			.renderPass = ambientRenderPass,
			.attachmentCount = uint32_t(attachments.size()), .pAttachments = attachments.data(),
			.width = swapChainExtent.width/2, .height = swapChainExtent.height/2, .layers = 1
		};
		frameDatas[i].ambientFrameBuffer = device.createFramebuffer(fbcreateinfo);
	}
}

void PrismRenderer::createFinalFrameBuffers()
{
	for (size_t i = 0; i < frameDatas.size(); i++) {
		std::vector<vk::ImageView> attachments;
		//Final RP
		attachments = { frameDatas[i].swapChainImage._imageView };
		vk::FramebufferCreateInfo fbcreateinfo{
			.renderPass = finalRenderPass,
			.attachmentCount = uint32_t(attachments.size()), .pAttachments = attachments.data(),
			.width = swapChainExtent.width, .height = swapChainExtent.height, .layers = 1
		};
		frameDatas[i].swapChainFrameBuffer = device.createFramebuffer(fbcreateinfo);
	}
}

vk::ShaderModule PrismRenderer::loadShader(std::string shaderFilePath)
{
	std::ifstream file(shaderFilePath, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	char* buffer = (char*)malloc(sizeof(char) * fileSize);

	file.seekg(0);
	file.read(buffer, fileSize);
	file.close();

	vk::ShaderModuleCreateInfo createInfo{ .codeSize = fileSize, .pCode = reinterpret_cast<const uint32_t*>(buffer) };
	vk::ShaderModule retval = device.createShaderModule(createInfo);
	free(buffer);
	return retval;
}

void PrismRenderer::createGBufferPipeline()
{
	vk::ShaderModule shaders[2];
	vk::PipelineShaderStageCreateInfo shaderStageInfos[2] = { {} };
	shaders[0] = loadShader("shaders/gbuffer.vert.spv");
	shaders[1] = loadShader("shaders/gbuffer.frag.spv");
	shaderStageInfos[0] = { .stage = vk::ShaderStageFlagBits::eVertex, .module = shaders[0], .pName = "main" };
	shaderStageInfos[1] = { .stage = vk::ShaderStageFlagBits::eFragment, .module = shaders[1], .pName = "main" };

	auto bindingDescription = Prism::Vertex::getBindingDescription();
	auto attributeDescriptions = Prism::Vertex::getAttributeDescriptions();

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDescription,
		.vertexAttributeDescriptionCount = uint32_t(attributeDescriptions.size()), .pVertexAttributeDescriptions = attributeDescriptions.data()
	};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = VK_FALSE };

	vk::Viewport viewport{
		.x = 0.0f, .y = float(swapChainExtent.height),
		.width = float(swapChainExtent.width), .height = -float(swapChainExtent.height),
		.minDepth = 0.0f, .maxDepth = 1.0f
	};
	vk::Rect2D scissor{ .offset = { 0, 0 }, .extent = swapChainExtent };
	vk::PipelineViewportStateCreateInfo viewportState{
		.viewportCount = 1, .pViewports = &viewport,
		.scissorCount = 1, .pScissors = &scissor
	};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};
	vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = VK_FALSE };
	
	std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments(4,
		{.blendEnable = VK_FALSE,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
		});

	vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = VK_FALSE, .attachmentCount = uint32_t(colorBlendAttachments.size()), .pAttachments = colorBlendAttachments.data()};
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
	.depthTestEnable = VK_TRUE,
	.depthWriteEnable = VK_TRUE,
	.depthCompareOp = vk::CompareOp::eLess,
	.depthBoundsTestEnable = VK_FALSE,
	.stencilTestEnable = VK_FALSE,
	};

	std::vector<vk::DescriptorSetLayout> pipelineSetLayouts;
	pipelineSetLayouts.push_back(dSetLayouts["vert_frag_uniform"]);
	pipelineSetLayouts.push_back(dSetLayouts["vert_storage"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_sampler_3"]);
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = uint32_t(pipelineSetLayouts.size()), .pSetLayouts = pipelineSetLayouts.data() };

	Prism::GPUPipeline gPipeline;
	gPipeline._pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.stageCount = 2, .pStages = shaderStageInfos,
		.pVertexInputState = &vertexInputInfo, .pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlending,
		.layout = gPipeline._pipelineLayout,
		.renderPass = gbufferRenderPass
	};
	gPipeline._pipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo).value;
	pipelines["gbuffer"] = gPipeline;
	for (vk::ShaderModule smod : shaders) device.destroyShaderModule(smod, NULL);
}

void PrismRenderer::createDSMapPipeline()
{
	vk::ShaderModule shaders[2];
	vk::PipelineShaderStageCreateInfo shaderStageInfos[2] = { {} };
	shaders[0] = loadShader("shaders/light_smap.vert.spv");
	shaders[1] = loadShader("shaders/light_smap.frag.spv");
	shaderStageInfos[0] = { .stage = vk::ShaderStageFlagBits::eVertex, .module = shaders[0], .pName = "main"};
	shaderStageInfos[1] = { .stage = vk::ShaderStageFlagBits::eFragment, .module = shaders[1], .pName = "main"};

	auto bindingDescription = Prism::Vertex::getBindingDescription();
	auto attributeDescriptions = Prism::Vertex::getAttributeDescriptions();

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDescription,
		.vertexAttributeDescriptionCount = uint32_t(attributeDescriptions.size()), .pVertexAttributeDescriptions = attributeDescriptions.data()
	};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = VK_FALSE };

	vk::Viewport viewport{
		.x = 0.0f, .y = float(dlight_smap_extent.height),
		.width = float(dlight_smap_extent.width), .height = -float(dlight_smap_extent.height),
		.minDepth = 0.0f, .maxDepth = 1.0f
	};
	vk::Rect2D scissor{ .offset = { 0, 0 }, .extent = dlight_smap_extent };
	vk::PipelineViewportStateCreateInfo viewportState{
		.viewportCount = 1, .pViewports = &viewport,
		.scissorCount = 1, .pScissors = &scissor
	};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};
	vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = VK_FALSE };

	vk::PipelineColorBlendAttachmentState colorBlendAttachments{
		.blendEnable = VK_FALSE,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = VK_FALSE, .attachmentCount = 1, .pAttachments = &colorBlendAttachments };
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
	.depthTestEnable = VK_TRUE,
	.depthWriteEnable = VK_TRUE,
	.depthCompareOp = vk::CompareOp::eLess,
	.depthBoundsTestEnable = VK_FALSE,
	.stencilTestEnable = VK_FALSE,
	};

	std::vector<vk::DescriptorSetLayout> pipelineSetLayouts;
	pipelineSetLayouts.push_back(dSetLayouts["vert_frag_uniform"]);
	pipelineSetLayouts.push_back(dSetLayouts["vert_storage"]);
	vk::PushConstantRange pcRange{ .stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = sizeof(Prism::GPULightPC) };
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount = uint32_t(pipelineSetLayouts.size()), .pSetLayouts = pipelineSetLayouts.data(),
		.pushConstantRangeCount = 1, .pPushConstantRanges = &pcRange
	};
	Prism::GPUPipeline gPipeline;
	gPipeline._pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.stageCount = 2, .pStages = shaderStageInfos,
		.pVertexInputState = &vertexInputInfo, .pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlending,
		.layout = gPipeline._pipelineLayout,
		.renderPass = shadowRenderPass
	};
	gPipeline._pipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo).value;
	pipelines["dsmap"] = gPipeline;
	for (vk::ShaderModule smod : shaders) device.destroyShaderModule(smod, NULL);
}

void PrismRenderer::createPSMapPipeline()
{
	vk::ShaderModule shaders[2];
	vk::PipelineShaderStageCreateInfo shaderStageInfos[2] = { {} };
	shaders[0] = loadShader("shaders/light_smap.vert.spv");
	shaders[1] = loadShader("shaders/light_smap.frag.spv");
	shaderStageInfos[0] = { .stage = vk::ShaderStageFlagBits::eVertex, .module = shaders[0], .pName = "main" };
	shaderStageInfos[1] = { .stage = vk::ShaderStageFlagBits::eFragment, .module = shaders[1], .pName = "main" };

	auto bindingDescription = Prism::Vertex::getBindingDescription();
	auto attributeDescriptions = Prism::Vertex::getAttributeDescriptions();

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bindingDescription,
		.vertexAttributeDescriptionCount = uint32_t(attributeDescriptions.size()), .pVertexAttributeDescriptions = attributeDescriptions.data()
	};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = VK_FALSE };

	vk::Viewport viewport{
		.x = 0.0f, .y = float(plight_smap_extent.height),
		.width = float(plight_smap_extent.width), .height = -float(plight_smap_extent.height),
		.minDepth = 0.0f, .maxDepth = 1.0f
	};
	vk::Rect2D scissor{ .offset = { 0, 0 }, .extent = plight_smap_extent };
	vk::PipelineViewportStateCreateInfo viewportState{
		.viewportCount = 1, .pViewports = &viewport,
		.scissorCount = 1, .pScissors = &scissor
	};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};
	vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = VK_FALSE };

	vk::PipelineColorBlendAttachmentState colorBlendAttachments{
		.blendEnable = VK_FALSE,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = VK_FALSE, .attachmentCount = 1, .pAttachments = &colorBlendAttachments };
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
	.depthTestEnable = VK_TRUE,
	.depthWriteEnable = VK_TRUE,
	.depthCompareOp = vk::CompareOp::eLess,
	.depthBoundsTestEnable = VK_FALSE,
	.stencilTestEnable = VK_FALSE,
	};

	std::vector<vk::DescriptorSetLayout> pipelineSetLayouts;
	pipelineSetLayouts.push_back(dSetLayouts["vert_frag_uniform"]);
	pipelineSetLayouts.push_back(dSetLayouts["vert_storage"]);
	vk::PushConstantRange pcRange{ .stageFlags = vk::ShaderStageFlagBits::eVertex, .offset = 0, .size = sizeof(Prism::GPULightPC) };
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
		.setLayoutCount = uint32_t(pipelineSetLayouts.size()), .pSetLayouts = pipelineSetLayouts.data(),
		.pushConstantRangeCount = 1, .pPushConstantRanges = &pcRange
	};
	Prism::GPUPipeline gPipeline;
	gPipeline._pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.stageCount = 2, .pStages = shaderStageInfos,
		.pVertexInputState = &vertexInputInfo, .pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlending,
		.layout = gPipeline._pipelineLayout,
		.renderPass = shadowRenderPass
	};
	gPipeline._pipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo).value;
	pipelines["psmap"] = gPipeline;
	for (vk::ShaderModule smod : shaders) device.destroyShaderModule(smod, NULL);
}

void PrismRenderer::createAmbientPipeline()
{
	vk::ShaderModule shaders[2];
	vk::PipelineShaderStageCreateInfo shaderStageInfos[2] = { {} };
	shaders[0] = loadShader("shaders/screensize.vert.spv");
	shaders[1] = loadShader("shaders/ambient.frag.spv");
	shaderStageInfos[0] = { .stage = vk::ShaderStageFlagBits::eVertex, .module = shaders[0], .pName = "main" };
	shaderStageInfos[1] = { .stage = vk::ShaderStageFlagBits::eFragment, .module = shaders[1], .pName = "main" };

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = VK_FALSE };

	vk::Viewport viewport{
		.x = 0.0f, .y = float(swapChainExtent.height/2),
		.width = float(swapChainExtent.width/2), .height = -float(swapChainExtent.height/2),
		.minDepth = 0.0f, .maxDepth = 1.0f
	};
	vk::Rect2D scissor{ .offset = { 0, 0 }, .extent = { swapChainExtent.width / 2, swapChainExtent.height / 2 } };
	vk::PipelineViewportStateCreateInfo viewportState{
		.viewportCount = 1, .pViewports = &viewport,
		.scissorCount = 1, .pScissors = &scissor
	};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};
	vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = VK_FALSE };

	vk::PipelineColorBlendAttachmentState colorBlendAttachments{
		.blendEnable = VK_FALSE,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = VK_FALSE, .attachmentCount = 1, .pAttachments = &colorBlendAttachments };
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
	.depthTestEnable = VK_FALSE,
	.depthWriteEnable = VK_FALSE,
	.depthCompareOp = vk::CompareOp::eLess,
	.depthBoundsTestEnable = VK_FALSE,
	.stencilTestEnable = VK_FALSE,
	};

	std::vector<vk::DescriptorSetLayout> pipelineSetLayouts;
	pipelineSetLayouts.push_back(dSetLayouts["frag_sampler_1"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_sampler_1"]);
	pipelineSetLayouts.push_back(dSetLayouts["vert_frag_uniform"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_sampler_1"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_uniform"]);
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = uint32_t(pipelineSetLayouts.size()), .pSetLayouts = pipelineSetLayouts.data() };
	
	Prism::GPUPipeline gPipeline;
	gPipeline._pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.stageCount = 2, .pStages = shaderStageInfos,
		.pVertexInputState = &vertexInputInfo, .pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlending,
		.layout = gPipeline._pipelineLayout,
		.renderPass = ambientRenderPass
	};
	gPipeline._pipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo).value;
	pipelines["ambient"] = gPipeline;
	for (vk::ShaderModule smod : shaders) device.destroyShaderModule(smod, NULL);
}

void PrismRenderer::createFinalPipeline()
{
	vk::ShaderModule shaders[2];
	vk::PipelineShaderStageCreateInfo shaderStageInfos[2] = { {} };
	shaders[0] = loadShader("shaders/screensize.vert.spv");
	shaders[1] = loadShader("shaders/finalimage.frag.spv");
	shaderStageInfos[0] = { .stage = vk::ShaderStageFlagBits::eVertex, .module = shaders[0], .pName = "main" };
	shaderStageInfos[1] = { .stage = vk::ShaderStageFlagBits::eFragment, .module = shaders[1], .pName = "main" };

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList, .primitiveRestartEnable = VK_FALSE };

	vk::Viewport viewport{
		.x = 0.0f, .y = 0.0f,
		.width = float(swapChainExtent.width), .height = float(swapChainExtent.height),
		.minDepth = 0.0f, .maxDepth = 1.0f
	};
	vk::Rect2D scissor{ .offset = { 0, 0 }, .extent = swapChainExtent };
	vk::PipelineViewportStateCreateInfo viewportState{
		.viewportCount = 1, .pViewports = &viewport,
		.scissorCount = 1, .pScissors = &scissor
	};

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = vk::PolygonMode::eFill,
		.cullMode = vk::CullModeFlagBits::eFront,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};
	vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = VK_FALSE };

	vk::PipelineColorBlendAttachmentState colorBlendAttachments{
		.blendEnable = VK_FALSE,
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = VK_FALSE, .attachmentCount = 1, .pAttachments = &colorBlendAttachments };
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
	.depthTestEnable = VK_FALSE,
	.depthWriteEnable = VK_FALSE,
	.depthCompareOp = vk::CompareOp::eLess,
	.depthBoundsTestEnable = VK_FALSE,
	.stencilTestEnable = VK_FALSE,
	};

	std::vector<vk::DescriptorSetLayout> pipelineSetLayouts;
	pipelineSetLayouts.push_back(dSetLayouts["vert_frag_uniform"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_uniform"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_sampler_5"]);
	pipelineSetLayouts.push_back(dSetLayouts["vert_frag_uniform"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_plight_sampler"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_dlight_sampler"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_sampler_1"]);
	pipelineSetLayouts.push_back(dSetLayouts["frag_uniform"]);
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = uint32_t(pipelineSetLayouts.size()), .pSetLayouts = pipelineSetLayouts.data() };

	Prism::GPUPipeline gPipeline;
	gPipeline._pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
	vk::GraphicsPipelineCreateInfo pipelineInfo{
		.stageCount = 2, .pStages = shaderStageInfos,
		.pVertexInputState = &vertexInputInfo, .pInputAssemblyState = &inputAssembly,
		.pViewportState = &viewportState,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depthStencil,
		.pColorBlendState = &colorBlending,
		.layout = gPipeline._pipelineLayout,
		.renderPass = finalRenderPass
	};
	gPipeline._pipeline = device.createGraphicsPipeline(VK_NULL_HANDLE, pipelineInfo).value;
	pipelines["final"] = gPipeline;
	for (vk::ShaderModule smod : shaders) device.destroyShaderModule(smod, NULL);
}

void PrismRenderer::createSyncObjects()
{
	imagesInFlight.resize(frameDatas.size());

	vk::SemaphoreCreateInfo semaphoreInfo{};
	vk::FenceCreateInfo fenceInfo{ .flags = vk::FenceCreateFlagBits::eSignaled };

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		frameDatas[i].presentSemaphore = device.createSemaphore(semaphoreInfo);
		frameDatas[i].renderSemaphore = device.createSemaphore(semaphoreInfo);
		frameDatas[i].renderFence = device.createFence(fenceInfo);
	}
}

void PrismRenderer::createRenderCmdBuffers()
{
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		frameDatas[i].commandBuffer = device.allocateCommandBuffers(
			{ .commandPool = frameDatas[i].commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 }
		)[0];
	}
}

void PrismRenderer::updateRenderCmds(size_t frameNo)
{
	vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eRenderPassContinue };
	frameDatas[frameNo].commandBuffer.begin(beginInfo);
	addGBufferRenderCmds(frameNo);
	addDSMapRenderCmds(frameNo);
	addPSMapRenderCmds(frameNo);
	addAmbientRenderCmds(frameNo);
	addFinalRenderCmds(frameNo);
	frameDatas[frameNo].commandBuffer.end();
}

void PrismRenderer::addGBufferRenderCmds(size_t frameNo)
{
	Prism::CommandBuffer cmdbuffer = frameDatas[frameNo].commandBuffer;

	vk::RenderPassBeginInfo rpbegininfo{
		.renderPass = gbufferRenderPass,
		.framebuffer = frameDatas[frameNo].gbufferFrameBuffer,
		.renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
		.clearValueCount = uint32_t(gbufferClearValues.size()),
		.pClearValues = gbufferClearValues.data(),
	};
	cmdbuffer.beginRenderPass(rpbegininfo, vk::SubpassContents::eInline);

	Prism::GPUPipeline gbuffer_pipeline = pipelines["gbuffer"];
	cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, gbuffer_pipeline._pipeline);

	size_t robjCount = cam_render_list[0];

	for (size_t ro_idx = 0; ro_idx < robjCount; ro_idx++) {
		Prism::RenderObject* robj = &renderObjects[cam_render_list[ro_idx + 1]];

		vk::Buffer vertexBuffers[] = { robj->_vertexBuffers[frameNo]._buffer};
		vk::DeviceSize offsets[] = { 0 };
		cmdbuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
		cmdbuffer.bindIndexBuffer(robj->_indexBuffers[frameNo]._buffer, 0, vk::IndexType::eUint32);
		cmdbuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, gbuffer_pipeline._pipelineLayout, 0,
			{
				frameDatas[frameNo].setBuffers["camera"]._dSet,
				frameDatas[frameNo].setBuffers["object"]._dSet,
				robj->texmaps->dset,
			},
			{}
		);
		cmdbuffer.drawRenderObjectMesh(robj, cam_render_list[ro_idx + 1]);
	}
	cmdbuffer.endRenderPass();
	//std::cout << "gbuffer objs drawn: " << robjCount << "\n";
}

void PrismRenderer::addDSMapRenderCmds(size_t frameNo)
{
	Prism::CommandBuffer cmdbuffer = frameDatas[frameNo].commandBuffer;

	std::vector<vk::ClearValue> clearValues(2, COLOR_CLEAR_VAL_DEFAULT);
	clearValues[1] = DEPTH_CLEAR_VAL_DEFAULT;

	vk::RenderPassBeginInfo rpbegininfo{
		.renderPass = shadowRenderPass,
		.framebuffer = frameDatas[frameNo].dShadowFrameBuffer,
		.renderArea = {.offset = { 0, 0 }, .extent = dlight_smap_extent },
		.clearValueCount = uint32_t(clearValues.size()),
		.pClearValues = clearValues.data(),
	};

	Prism::GPUPipeline dLightPipeline = pipelines["dsmap"];
	Prism::GPULightPC tlpc;
	tlpc.viewproj = glm::mat4(1);

	for (size_t lidx = 0; lidx < MAX_DIRECTIONAL_LIGHTS; lidx++) {
		if (!(lights[MAX_NS_LIGHTS + MAX_POINT_LIGHTS + lidx].flags.x & 1)) continue;

		tlpc.idx.x = int(MAX_NS_LIGHTS + MAX_POINT_LIGHTS + lidx);
	
		cmdbuffer.beginRenderPass(rpbegininfo, vk::SubpassContents::eInline);
		cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, dLightPipeline._pipeline);
		cmdbuffer.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, dLightPipeline._pipelineLayout, 0,
			{ frameDatas[frameNo].setBuffers["light"]._dSet, frameDatas[frameNo].setBuffers["object"]._dSet }, {}
		);
		cmdbuffer.pushConstants(dLightPipeline._pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(Prism::GPULightPC), &tlpc);
		size_t drcount = 0;
		for (size_t do_idx = 0; do_idx < dlight_render_list[MAX_OBJECTS * lidx]; do_idx++) {
			size_t ro_idx = dlight_render_list[MAX_OBJECTS * lidx + do_idx + 1];
			Prism::RenderObject robj = renderObjects[ro_idx];
			if (!robj.shadowcasting) continue;

			vk::Buffer vertexBuffers[] = { robj._vertexBuffers[frameNo]._buffer };
			vk::DeviceSize offsets[] = { 0 };
			cmdbuffer.bindVertexBuffers(0, vertexBuffers, offsets);
			cmdbuffer.bindIndexBuffer(robj._indexBuffers[frameNo]._buffer, 0, vk::IndexType::eUint32);
			cmdbuffer.drawRenderObjectMesh(&robj, ro_idx);
			drcount++;
		}
		cmdbuffer.endRenderPass();
		//std::cout << "dlight" << lidx << " objs drawn: " << drcount << "\n";

		vk::ImageMemoryBarrier imbarrier{
			.srcAccessMask = vk::AccessFlagBits::eShaderRead,
			.dstAccessMask = vk::AccessFlagBits::eTransferWrite,
			.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			.newLayout = vk::ImageLayout::eTransferDstOptimal,
			.image = frameDatas[frameNo].shadow_dir_maps[lidx]._image,
			.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		};
		cmdbuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eAllCommands,
			vk::PipelineStageFlagBits::eAllCommands,
			vk::DependencyFlagBits::eByRegion,
			{}, {}, { imbarrier }
		);

		vk::ImageCopy copyRegion = {
			.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
			.srcOffset = {0,0,0},
			.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
			.dstOffset = {0,0,0},
			.extent = {dlight_smap_extent.width, dlight_smap_extent.height, 1}
		};
		cmdbuffer.copyImage(
			frameDatas[frameNo].dShadowMapTemp._image,
			vk::ImageLayout::eTransferSrcOptimal,
			frameDatas[frameNo].shadow_dir_maps[lidx]._image,
			vk::ImageLayout::eTransferDstOptimal,
			{ copyRegion }
		);

		imbarrier = {
			.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
			.dstAccessMask = vk::AccessFlagBits::eShaderRead,
			.oldLayout = vk::ImageLayout::eTransferDstOptimal,
			.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			.image = frameDatas[frameNo].shadow_dir_maps[lidx]._image,
			.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
		};
		cmdbuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eAllCommands,
			vk::PipelineStageFlagBits::eAllCommands,
			vk::DependencyFlagBits::eByRegion,
			{}, {}, { imbarrier }
		);
	}
}

void PrismRenderer::addPSMapRenderCmds(size_t frameNo)
{
	Prism::CommandBuffer cmdbuffer = frameDatas[frameNo].commandBuffer;

	std::vector<vk::ClearValue> clearValues(2, COLOR_CLEAR_VAL_DEFAULT);
	clearValues[1] = DEPTH_CLEAR_VAL_DEFAULT;

	vk::RenderPassBeginInfo rpbegininfo{
		.renderPass = shadowRenderPass,
		.framebuffer = frameDatas[frameNo].pShadowFrameBuffer,
		.renderArea = {.offset = { 0, 0 }, .extent = plight_smap_extent },
		.clearValueCount = uint32_t(clearValues.size()),
		.pClearValues = clearValues.data(),
	};

	Prism::GPUPipeline pLightPipeline = pipelines["psmap"];
	Prism::GPULightPC tlpc;
	
	glm::mat4 projMatrix = glm::perspective(glm::radians(90.0f), float(plight_smap_extent.width) / float(plight_smap_extent.height), 0.01f, 1000.0f);

	for (size_t lidx = 0; lidx < MAX_POINT_LIGHTS; lidx++) {
		if (!(lights[MAX_NS_LIGHTS + lidx].flags.x & 1)) continue;
		tlpc.idx.x = int(MAX_NS_LIGHTS + lidx);

		size_t drcount = 0;
		
		for (uint32_t face = 0; face < 6; face++) {
			glm::mat4 viewMatrix = glm::mat4(1);
			switch (face)
			{
			case 0: // POSITIVE_X
				viewMatrix = glm::lookAt(glm::vec3(0), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
				break;
			case 1:	// NEGATIVE_X
				viewMatrix = glm::lookAt(glm::vec3(0), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
				break;
			case 2:	// POSITIVE_Y
				viewMatrix = glm::lookAt(glm::vec3(0), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
				break;
			case 3:	// NEGATIVE_Y
				viewMatrix = glm::lookAt(glm::vec3(0), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
				break;
			case 4:	// POSITIVE_Z
				viewMatrix = glm::lookAt(glm::vec3(0), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
				break;
			case 5:	// NEGATIVE_Z
				viewMatrix = glm::lookAt(glm::vec3(0), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
				break;
			}

			tlpc.viewproj = projMatrix * viewMatrix;

			cmdbuffer.beginRenderPass(rpbegininfo, vk::SubpassContents::eInline);
			cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pLightPipeline._pipeline);
			cmdbuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics, pLightPipeline._pipelineLayout, 0,
				{ frameDatas[frameNo].setBuffers["light"]._dSet, frameDatas[frameNo].setBuffers["object"]._dSet }, {}
			);
			cmdbuffer.pushConstants(pLightPipeline._pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(Prism::GPULightPC), &tlpc);

			for (size_t po_idx = 0; po_idx < plight_render_list[MAX_OBJECTS * lidx]; po_idx++) {
				size_t ro_idx = dlight_render_list[MAX_OBJECTS * lidx + po_idx + 1];
				Prism::RenderObject robj = renderObjects[ro_idx];
				if (!robj.shadowcasting) continue;

				vk::Buffer vertexBuffers[] = { robj._vertexBuffers[frameNo]._buffer };
				vk::DeviceSize offsets[] = { 0 };
				cmdbuffer.bindVertexBuffers(0, vertexBuffers, offsets);
				cmdbuffer.bindIndexBuffer(robj._indexBuffers[frameNo]._buffer, 0, vk::IndexType::eUint32);
				cmdbuffer.drawRenderObjectMesh(&robj, ro_idx);
				drcount++;
			}
			cmdbuffer.endRenderPass();

			vk::ImageMemoryBarrier imbarrier{
				.srcAccessMask = vk::AccessFlagBits::eShaderRead,
				.dstAccessMask = vk::AccessFlagBits::eTransferWrite,
				.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
				.newLayout = vk::ImageLayout::eTransferDstOptimal,
				.image = frameDatas[frameNo].shadow_cube_maps[lidx]._image,
				.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, face, 1}
			};
			cmdbuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eAllCommands,
				vk::PipelineStageFlagBits::eAllCommands,
				vk::DependencyFlagBits::eByRegion,
				{}, {}, { imbarrier }
			);

			vk::ImageCopy copyRegion = {
				.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
				.srcOffset = {0,0,0},
				.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, face, 1},
				.dstOffset = {0,0,0},
				.extent = {plight_smap_extent.width, plight_smap_extent.height, 1}
			};
			cmdbuffer.copyImage(
				frameDatas[frameNo].pShadowMapTemp._image,
				vk::ImageLayout::eTransferSrcOptimal,
				frameDatas[frameNo].shadow_cube_maps[lidx]._image,
				vk::ImageLayout::eTransferDstOptimal,
				{ copyRegion }
			);

			imbarrier = {
				.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
				.dstAccessMask = vk::AccessFlagBits::eShaderRead,
				.oldLayout = vk::ImageLayout::eTransferDstOptimal,
				.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
				.image = frameDatas[frameNo].shadow_cube_maps[lidx]._image,
				.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, face, 1}
			};
			cmdbuffer.pipelineBarrier(
				vk::PipelineStageFlagBits::eAllCommands,
				vk::PipelineStageFlagBits::eAllCommands,
				vk::DependencyFlagBits::eByRegion,
				{}, {}, { imbarrier }
			);
		}

		//std::cout << "plight" << lidx << " objs drawn: " << drcount << "\n";
	}
}

void PrismRenderer::addAmbientRenderCmds(size_t frameNo)
{
	Prism::CommandBuffer cmdbuffer = frameDatas[frameNo].commandBuffer;

	std::vector<vk::ClearValue> clearValues(1, COLOR_CLEAR_VAL_DEFAULT);

	Prism::GPUPipeline ambient_pipeline = pipelines["ambient"];

	vk::RenderPassBeginInfo rpbegininfo{
		.renderPass = ambientRenderPass,
		.framebuffer = frameDatas[frameNo].ambientFrameBuffer,
		.renderArea = {.offset = { 0, 0 }, .extent = { swapChainExtent.width/2, swapChainExtent.height/2 } },
		.clearValueCount = uint32_t(clearValues.size()),
		.pClearValues = clearValues.data(),
	};

	cmdbuffer.beginRenderPass(rpbegininfo, vk::SubpassContents::eInline);
	cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ambient_pipeline._pipeline);
	cmdbuffer.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics, ambient_pipeline._pipelineLayout, 0,
		{
			frameDatas[frameNo].positionImageDset,
			frameDatas[frameNo].normalImageDset,
			frameDatas[frameNo].setBuffers["camera"]._dSet,
			ambientNoiseImageDset,
			ambientKernel._dSet
		}, {}
	);
	cmdbuffer.draw(3, 1, 0, 0);
	cmdbuffer.endRenderPass();
}

void PrismRenderer::addFinalRenderCmds(size_t frameNo)
{
	Prism::CommandBuffer cmdbuffer = frameDatas[frameNo].commandBuffer;

	std::vector<vk::ClearValue> clearValues(1, COLOR_CLEAR_VAL_DEFAULT);

	Prism::GPUPipeline final_pipeline = pipelines["final"];

	vk::RenderPassBeginInfo rpbegininfo{
		.renderPass = finalRenderPass,
		.framebuffer = frameDatas[frameNo].swapChainFrameBuffer,
		.renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
		.clearValueCount = uint32_t(clearValues.size()),
		.pClearValues = clearValues.data(),
	};

	cmdbuffer.beginRenderPass(rpbegininfo, vk::SubpassContents::eInline);
	cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, final_pipeline._pipeline);
	cmdbuffer.bindDescriptorSets(
		vk::PipelineBindPoint::eGraphics, final_pipeline._pipelineLayout, 0,
		{
			frameDatas[frameNo].setBuffers["camera"]._dSet,
			frameDatas[frameNo].setBuffers["scene"]._dSet,
			frameDatas[frameNo].finalComposeDset,
			frameDatas[frameNo].setBuffers["light"]._dSet,
			frameDatas[frameNo].shadow_cube_dset,
			frameDatas[frameNo].shadow_dir_dset,
			shadowNoiseImageDset,
			shadowKernel._dSet,
		}, {}
	);
	cmdbuffer.draw(3, 1, 0, 0);
	cmdbuffer.endRenderPass();
}

void PrismRenderer::createTexSamplers()
{
	//Create Basic Texture Sampler
	vk::SamplerCreateInfo samplerInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.addressModeU = vk::SamplerAddressMode::eRepeat,
		.addressModeV = vk::SamplerAddressMode::eRepeat,
		.addressModeW = vk::SamplerAddressMode::eRepeat,
		.anisotropyEnable = VK_TRUE,
		.maxAnisotropy = phyDeviceProps.limits.maxSamplerAnisotropy,
		.compareEnable = VK_FALSE,
		.compareOp = vk::CompareOp::eAlways,
		.borderColor = vk::BorderColor::eIntOpaqueBlack,
		.unnormalizedCoordinates = VK_FALSE,
	};
	texSamplers["linear_filtered"] = device.createSampler(samplerInfo);
	samplerInfo.anisotropyEnable = VK_FALSE;
	texSamplers["linear_unfiltered"] = device.createSampler(samplerInfo);

	//Create Basic Texture Sampler Clamped
	samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToBorder;
	samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToBorder;
	samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToBorder;
	texSamplers["linear_clamped_unfiltered"] = device.createSampler(samplerInfo);
	samplerInfo.anisotropyEnable = VK_TRUE;
	texSamplers["linear_clamped_filtered"] = device.createSampler(samplerInfo);
}

Prism::GPUImage PrismRenderer::create2DGPUImage(vk::Format format, vk::Extent3D extent, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::ImageLayout initLayout, vk::ImageAspectFlags aspectFlags, vk::MemoryPropertyFlags memFlag)
{
	Prism::GPUImage timg;

	vk::ImageCreateInfo imageInfo{
		.imageType = vk::ImageType::e2D,
		.format = format,
		.extent = extent,
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = vk::SampleCountFlagBits::e1,
		.tiling = tiling,
		.usage = usage,
		.sharingMode = vk::SharingMode::eExclusive,
		.initialLayout = vk::ImageLayout::eUndefined,
	};
	timg._image = device.createImage(imageInfo);

	vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(timg._image);
	vk::MemoryAllocateInfo allocInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = physicalDevice.findMemoryType(memRequirements.memoryTypeBits, memFlag)
	};
	timg._imageMemory = device.allocateMemory(allocInfo);
	device.bindImageMemory(timg._image, timg._imageMemory, 0);

	vk::ImageViewCreateInfo viewInfo{
		.image = timg._image,
		.viewType = vk::ImageViewType::e2D,
		.format = format,
		.subresourceRange = {aspectFlags, 0, 1, 0, 1},
	};
	if (format == vk::Format::eR32Sfloat) viewInfo.components = { vk::ComponentSwizzle::eR };
	timg._imageView = device.createImageView(viewInfo);

	Prism::CommandBuffer cmdbuffer = beginOneTimeCmds();
	vk::ImageMemoryBarrier imbarrier{
			.srcAccessMask = vk::AccessFlagBits::eNone,
			.dstAccessMask = getAccessFlagsForLayout(initLayout),
			.oldLayout = vk::ImageLayout::eUndefined,
			.newLayout = initLayout,
			.image = timg._image,
			.subresourceRange = {aspectFlags, 0, 1, 0, 1}
	};
	cmdbuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands,
		vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlagBits::eByRegion,
		{}, {}, { imbarrier }
	);
	endOneTimeCmds(cmdbuffer);

	return timg;
}

Prism::GPUImage PrismRenderer::createCubeGPUImage(vk::Format format, vk::Extent3D extent, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::ImageLayout initLayout, vk::ImageAspectFlags aspectFlags, vk::MemoryPropertyFlags memFlag)
{
	Prism::GPUImage timg;

	vk::ImageCreateInfo imageInfo{
		.flags = vk::ImageCreateFlagBits::eCubeCompatible,
		.imageType = vk::ImageType::e2D,
		.format = format,
		.extent = extent,
		.mipLevels = 1,
		.arrayLayers = 6,
		.samples = vk::SampleCountFlagBits::e1,
		.tiling = tiling,
		.usage = usage,
		.sharingMode = vk::SharingMode::eExclusive,
		.initialLayout = vk::ImageLayout::eUndefined,
	};
	timg._image = device.createImage(imageInfo);

	vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(timg._image);
	vk::MemoryAllocateInfo allocInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = physicalDevice.findMemoryType(memRequirements.memoryTypeBits, memFlag)
	};
	timg._imageMemory = device.allocateMemory(allocInfo);
	device.bindImageMemory(timg._image, timg._imageMemory, 0);

	vk::ImageViewCreateInfo viewInfo{
		.image = timg._image,
		.viewType = vk::ImageViewType::eCube,
		.format = format,
		.subresourceRange = {aspectFlags, 0, 1, 0, 6},
	};
	if (format == vk::Format::eR32Sfloat) viewInfo.components = { vk::ComponentSwizzle::eR };
	timg._imageView = device.createImageView(viewInfo);

	Prism::CommandBuffer cmdbuffer = beginOneTimeCmds();
	vk::ImageMemoryBarrier imbarrier{
			.srcAccessMask = vk::AccessFlagBits::eNone,
			.dstAccessMask = getAccessFlagsForLayout(initLayout),
			.oldLayout = vk::ImageLayout::eUndefined,
			.newLayout = initLayout,
			.image = timg._image,
			.subresourceRange = {aspectFlags, 0, 1, 0, 6}
	};
	cmdbuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands,
		vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlagBits::eByRegion,
		{}, {}, { imbarrier }
	);
	endOneTimeCmds(cmdbuffer);

	return timg;
}

void PrismRenderer::destroyGPUImage(Prism::GPUImage image, bool freeMem)
{
	device.destroyImageView(image._imageView);
	device.destroyImage(image._image);
	if (freeMem) device.freeMemory(image._imageMemory);
}

void PrismRenderer::CPUtoGPUBufferCopy(void* data, vk::DeviceSize offset, vk::DeviceSize dataSize, Prism::GPUBuffer gbuff)
{
	void* tmp;
	device.mapMemory(gbuff._bufferMemory, offset, dataSize, (vk::MemoryMapFlags)0, &tmp);
	memcpy(tmp, data, dataSize);
	device.unmapMemory(gbuff._bufferMemory);
}

Prism::CommandBuffer PrismRenderer::beginOneTimeCmds()
{
	vk::CommandBufferAllocateInfo allocInfo{
		.commandPool = uploadCmdPool,
		.level = vk::CommandBufferLevel::ePrimary,
		.commandBufferCount = 1,
	};
	Prism::CommandBuffer cmdBuffer = Prism::CommandBuffer(device.allocateCommandBuffers(allocInfo)[0]);
	vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
	cmdBuffer.begin(beginInfo);
	return cmdBuffer;
}

void PrismRenderer::endOneTimeCmds(Prism::CommandBuffer cmdbuff)
{
	cmdbuff.end();
	vk::SubmitInfo otsubinfo{ .commandBufferCount = 1, .pCommandBuffers = &cmdbuff };
	transferQueue.submit(otsubinfo);
	transferQueue.waitIdle();

	device.freeCommandBuffers(uploadCmdPool, { cmdbuff });
}

void PrismRenderer::copyDatatoImage(void* data, vk::DeviceSize dataSize, Prism::GPUImage image, vk::Extent3D imgExtent, vk::Offset3D imgOffset)
{
	Prism::GPUBuffer stageBuff = createGPUBuffer(dataSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	CPUtoGPUBufferCopy(data, 0, dataSize, stageBuff);
	Prism::CommandBuffer cmdBuff = beginOneTimeCmds();
	vk::BufferImageCopy region{
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
		.imageOffset = imgOffset,
		.imageExtent = imgExtent
	};
	cmdBuff.copyBufferToImage(stageBuff._buffer, image._image, vk::ImageLayout::eTransferDstOptimal, { region });
	endOneTimeCmds(cmdBuff);
	device.destroyGPUBuffer(stageBuff);
}

void PrismRenderer::destroyGPUPipeline(std::string pipeline_name)
{
	device.destroyPipelineLayout(pipelines[pipeline_name]._pipelineLayout);
	device.destroyPipeline(pipelines[pipeline_name]._pipeline);
	pipelines.erase(pipeline_name);
}

void PrismRenderer::createGBufferDepthImage()
{
	gbufferDepthImage = create2DGPUImage(
		depthFormat,
		{ swapChainExtent.width, swapChainExtent.height, 1 },
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::ImageLayout::eDepthStencilAttachmentOptimal,
		vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);
}

void PrismRenderer::createDSMapDepthImage()
{
	dShadowDepthImage = create2DGPUImage(
		depthFormat,
		{ dlight_smap_extent.width, dlight_smap_extent.height, 1 },
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::ImageLayout::eDepthStencilAttachmentOptimal,
		vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);
}

void PrismRenderer::createPSMapDepthImage()
{
	pShadowDepthImage = create2DGPUImage(
		depthFormat,
		{ plight_smap_extent.width, plight_smap_extent.height, 1 },
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::ImageLayout::eDepthStencilAttachmentOptimal,
		vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);
}

void PrismRenderer::createNoiseImages()
{
	std::uniform_real_distribution<float> randomFloats(0.0, 1.0); // random floats between [0.0, 1.0]
	std::default_random_engine generator;

	std::vector<glm::vec4> ssaoNoise;
	for (unsigned int i = 0; i < 16; i++)
	{
		glm::vec4 noise(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) * 2.0 - 1.0,
			0.0f,
			0.0f);
		ssaoNoise.push_back(noise);
	}

	ambientNoiseImage = create2DGPUImage(
		vk::Format::eR32G32B32A32Sfloat,
		{ 4,4,1 },
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageAspectFlagBits::eColor,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);
	copyDatatoImage(ssaoNoise.data(), sizeof(glm::vec4) * ssaoNoise.size(), ambientNoiseImage, { 4,4,1 });
	Prism::CommandBuffer cmdbuffer = beginOneTimeCmds();
	vk::ImageMemoryBarrier imbarrier{
			.srcAccessMask = getAccessFlagsForLayout(vk::ImageLayout::eTransferDstOptimal),
			.dstAccessMask = getAccessFlagsForLayout(vk::ImageLayout::eShaderReadOnlyOptimal),
			.oldLayout = vk::ImageLayout::eTransferDstOptimal,
			.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			.image = ambientNoiseImage._image,
			.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	cmdbuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands,
		vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlagBits::eByRegion,
		{}, {}, { imbarrier }
	);
	endOneTimeCmds(cmdbuffer);

	std::vector<glm::vec4> shadowNoise;
	for (unsigned int i = 0; i < 16; i++)
	{
		glm::vec4 noise(
			randomFloats(generator) * 2.0 - 1.0,
			randomFloats(generator) * 2.0 - 1.0,
			0.0f,
			0.0f);
		shadowNoise.push_back(noise);
	}

	shadowNoiseImage = create2DGPUImage(
		vk::Format::eR32G32B32A32Sfloat,
		{ 4,4,1 },
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageAspectFlagBits::eColor,
		vk::MemoryPropertyFlagBits::eDeviceLocal
	);
	copyDatatoImage(shadowNoise.data(), sizeof(glm::vec4) * shadowNoise.size(), shadowNoiseImage, { 4,4,1 });
	cmdbuffer = beginOneTimeCmds();
	imbarrier = {
			.srcAccessMask = getAccessFlagsForLayout(vk::ImageLayout::eTransferDstOptimal),
			.dstAccessMask = getAccessFlagsForLayout(vk::ImageLayout::eShaderReadOnlyOptimal),
			.oldLayout = vk::ImageLayout::eTransferDstOptimal,
			.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
			.image = shadowNoiseImage._image,
			.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
	};
	cmdbuffer.pipelineBarrier(
		vk::PipelineStageFlagBits::eAllCommands,
		vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlagBits::eByRegion,
		{}, {}, { imbarrier }
	);
	endOneTimeCmds(cmdbuffer);
}

void PrismRenderer::createGBufferAttachmentImages()
{
	for (uint32_t i = 0; i < frameDatas.size(); i++) {
		// GBuffer attachments
		frameDatas[i].colorImage = create2DGPUImage(
			vk::Format::eR8G8B8A8Unorm,
			{ swapChainExtent.width, swapChainExtent.height, 1 },
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageAspectFlagBits::eColor,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		frameDatas[i].positionImage = create2DGPUImage(
			vk::Format::eR32G32B32A32Sfloat,
			{ swapChainExtent.width, swapChainExtent.height, 1 },
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageAspectFlagBits::eColor,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		frameDatas[i].normalImage = create2DGPUImage(
			vk::Format::eR8G8B8A8Unorm,
			{ swapChainExtent.width, swapChainExtent.height, 1 },
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageAspectFlagBits::eColor,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
		frameDatas[i].seImage = create2DGPUImage(
			vk::Format::eR8G8B8A8Unorm,
			{ swapChainExtent.width, swapChainExtent.height, 1 },
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageAspectFlagBits::eColor,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
	}
}

void PrismRenderer::createDSMapAttachmentImages()
{
	for (uint32_t i = 0; i < frameDatas.size(); i++) {
		// DSMAP attachments
		for (uint32_t j = 0; j < MAX_DIRECTIONAL_LIGHTS; j++) {
			Prism::GPUImage timg = create2DGPUImage(
				vk::Format::eR32Sfloat,
				{ dlight_smap_extent.width, dlight_smap_extent.height , 1 },
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
				vk::ImageLayout::eShaderReadOnlyOptimal,
				vk::ImageAspectFlagBits::eColor,
				vk::MemoryPropertyFlagBits::eDeviceLocal
			);
			frameDatas[i].shadow_dir_maps.push_back(timg);
		}
		frameDatas[i].dShadowMapTemp = create2DGPUImage(
			vk::Format::eR32Sfloat,
			{ dlight_smap_extent.width, dlight_smap_extent.height , 1 },
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eColorAttachment,
			vk::ImageLayout::eTransferSrcOptimal,
			vk::ImageAspectFlagBits::eColor,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
	}
}

void PrismRenderer::createPSMapAttachmentImages()
{
	for (uint32_t i = 0; i < frameDatas.size(); i++) {
		// PSMAP attachments
		for (uint32_t j = 0; j < MAX_POINT_LIGHTS; j++) {
			Prism::GPUImage timg = createCubeGPUImage(
				vk::Format::eR32Sfloat,
				{ plight_smap_extent.width, plight_smap_extent.height , 1 },
				vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
				vk::ImageLayout::eShaderReadOnlyOptimal,
				vk::ImageAspectFlagBits::eColor,
				vk::MemoryPropertyFlagBits::eDeviceLocal
			);
			frameDatas[i].shadow_cube_maps.push_back(timg);
		}
		frameDatas[i].pShadowMapTemp = create2DGPUImage(
			vk::Format::eR32Sfloat,
			{ plight_smap_extent.width, plight_smap_extent.height , 1 },
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eColorAttachment,
			vk::ImageLayout::eTransferSrcOptimal,
			vk::ImageAspectFlagBits::eColor,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
	}
}

void PrismRenderer::createAmbientAttachmentImages()
{
	for (uint32_t i = 0; i < frameDatas.size(); i++) {
		// Ambient attachments
		frameDatas[i].ambientImage = create2DGPUImage(
			vk::Format::eR8Unorm,
			{ swapChainExtent.width/2, swapChainExtent.height/2, 1 },
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageAspectFlagBits::eColor,
			vk::MemoryPropertyFlagBits::eDeviceLocal
		);
	}
}
