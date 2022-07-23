#pragma once
#include "vkstructs.h"

namespace vkutils {
	Prism::Instance createVKInstance(
		vk::ApplicationInfo appInfo,
		bool enableValidationLayers,
		std::vector<const char*> validationLayers
	);

	Prism::QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device, vk::SurfaceKHR surface);

	Prism::PhysicalDevice pickPhysicalDevice(
		Prism::Instance instance,
		vk::SurfaceKHR surface,
		std::vector<const char*> requiredExtensions,
		bool allowIntegrated
	);

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);

	vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);

	vk::Extent2D chooseSwapExtent(GLFWwindow* window, const vk::SurfaceCapabilitiesKHR& capabilities);
}