#include "vkutils.h"

#include <set>
#include <iostream>

bool checkValidationLayerSupport(const std::vector<const char*> vlayer_list) {
	std::vector<vk::LayerProperties> availLayersRes = vk::enumerateInstanceLayerProperties();

	for (const vk::LayerProperties vlp : availLayersRes) {
		std::cout << vlp.layerName << std::endl;
	}

	for (const char* layerName : vlayer_list) {
		bool layerFound = false;

		for (const auto& layerProperties : availLayersRes) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}

	return true;
}

std::vector<const char*> getRequiredExtensions(bool enableValidationLayers) {
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

bool checkExtensionSupport(std::vector<const char*> ext_list)
{
	//Get Supported extensions
	std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();

	for (uint32_t i = 0; i < ext_list.size(); i++) {
		bool found = false;
		for (const auto& ext : extensions) {
			if (!strcmp(ext_list[i], ext.extensionName)) {
				found = true;
				break;
			}
		}
		if (!found) return false;
	}

	return true;
}

Prism::Instance vkutils::createVKInstance(vk::ApplicationInfo appInfo, bool enableValidationLayers, std::vector<const char*> validationLayers)
{
	//Check Validation Layer support (only used if debug build)
	if (enableValidationLayers && !checkValidationLayerSupport(validationLayers)) {
		throw std::runtime_error("validation layers requested, but not available!");
	}

	//Query extensions needed for glfw
	std::vector<const char*> extensions = getRequiredExtensions(enableValidationLayers);

	//Verify if extentions are supported
	if (!checkExtensionSupport(extensions)) {
		throw std::runtime_error("Extentions required by glfw not supported");
	}

	//Struct to create instance
	vk::InstanceCreateInfo createInfo{
		.pApplicationInfo = &appInfo,
		.enabledExtensionCount = uint32_t(extensions.size()),
		.ppEnabledExtensionNames = extensions.data()
	};

	if (enableValidationLayers) {
		createInfo.enabledLayerCount = uint32_t(validationLayers.size());
		createInfo.ppEnabledLayerNames = validationLayers.data();
		vk::DebugUtilsMessengerCreateInfoEXT dmcreateinfo{
			.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
			.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
			.pfnUserCallback = GPUDebugCallback
		};
		createInfo.pNext = &dmcreateinfo;
	}
	else {
		createInfo.enabledLayerCount = 0;
		createInfo.pNext = NULL;
	}

	//Create instance
	Prism::Instance instance = Prism::Instance(vk::createInstance(createInfo));
	if (enableValidationLayers) instance.setupDbgMessenger();

	return instance;
}

Prism::QueueFamilyIndices vkutils::findQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
	Prism::QueueFamilyIndices indices;
	std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) indices.graphicsFamily = i;
		else if (queueFamily.queueFlags & vk::QueueFlagBits::eTransfer) indices.transferFamily = i;
		if (device.getSurfaceSupportKHR(i, surface)) {
			indices.presentFamily = i;
		}
		if (indices.isComplete()) break;
		i++;
	}
	return indices;
}

bool checkDeviceExtensionSupport(vk::PhysicalDevice device, std::vector<const char*> requiredExtensions) {

	std::vector<vk::ExtensionProperties> availableExtensions = device.enumerateDeviceExtensionProperties();

	std::set<std::string> leftExtensions(requiredExtensions.begin(), requiredExtensions.end());

	for (const auto& extension : availableExtensions) {
		leftExtensions.erase(extension.extensionName);
	}
	return leftExtensions.empty();
}

bool isDeviceSuitable(
	vk::PhysicalDevice device,
	vk::SurfaceKHR surface,
	std::vector<const char*> requiredExtensions,
	bool allow_integrated
)
{
	vk::PhysicalDeviceProperties supportedProperties = device.getProperties();
	vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();
	Prism::QueueFamilyIndices indices = vkutils::findQueueFamilies(device, surface);

	bool extensionsSupported = checkDeviceExtensionSupport(device, requiredExtensions);

	bool swapChainAdequate = false;
	if (extensionsSupported) {
		Prism::SwapChainSupportDetails swapChainSupport(device, surface);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	return (allow_integrated || supportedProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) &&
		supportedFeatures.geometryShader && indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

Prism::PhysicalDevice vkutils::pickPhysicalDevice(Prism::Instance instance, vk::SurfaceKHR surface, std::vector<const char*> requiredExtensions, bool allowIntegrated)
{
	std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
	if (devices.size() == 0) {
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}

	bool integrated_status = false;
	if (devices.size() == 1) integrated_status = true;

	vk::PhysicalDevice physicalDevice{};
	for (const auto& device : devices) {
		if (isDeviceSuitable(device, surface, requiredExtensions, integrated_status)) {
			physicalDevice = device;
			break;
		}
	}
	if (!physicalDevice) throw std::runtime_error("failed to find a suitable GPU!");
	return Prism::PhysicalDevice(physicalDevice);
}

vk::SurfaceFormatKHR vkutils::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return availableFormat;
		}
	}
	return availableFormats[0];
}

vk::PresentModeKHR vkutils::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
	for (const auto& availablePresentMode : availablePresentModes) {
		if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
			return availablePresentMode;
		}
	}
	return vk::PresentModeKHR::eFifo;
}

vk::Extent2D vkutils::chooseSwapExtent(GLFWwindow* window, const vk::SurfaceCapabilitiesKHR& capabilities) {
	if (capabilities.currentExtent.width != UINT32_MAX) {
		return capabilities.currentExtent;
	}
	else {
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		vk::Extent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}