/*
 * Copyright © 2011 Benjamin Franzke
 * Copyright (c) 2012 Arvin Schnell <arvin.schnell@gmail.com>
 * Copyright (c) 2012 Rob Clark <rob@ti.com>
 * Copyright © 2015 Intel Corporation
 * Copyright © 2023 Erico Nunes
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "config.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <signal.h>

#include <linux/input.h>

#include <wayland-client.h>
#include <wayland-cursor.h>

#include "fractional-scale-v1-client-protocol.h"
#include "viewporter-client-protocol.h"
#include "xdg-shell-client-protocol.h"
#include "tearing-control-v1-client-protocol.h"

#include <sys/types.h>
#include <unistd.h>

#include <libweston/matrix.h>
#include "shared/helpers.h"
#include "shared/platform.h"
#include "shared/xalloc.h"

#define VK_USE_PLATFORM_WAYLAND_KHR
#define VK_PROTOTYPES
#include <vulkan/vulkan.h>

#define MAX_NUM_IMAGES 4

struct window;
struct seat;

struct display {
	struct wl_display *display;
	struct wl_registry *registry;
	struct wl_compositor *compositor;
	struct xdg_wm_base *wm_base;
	struct wl_seat *seat;
	struct wl_pointer *pointer;
	struct wl_touch *touch;
	struct wl_keyboard *keyboard;
	struct wl_shm *shm;
	struct wl_cursor_theme *cursor_theme;
	struct wl_cursor *default_cursor;
	struct wl_surface *cursor_surface;
	struct wp_tearing_control_manager_v1 *tearing_manager;
	struct wp_viewporter *viewporter;
	struct wp_fractional_scale_manager_v1 *fractional_scale_manager;
	struct window *window;

	struct wl_list output_list; /* struct output::link */
};

struct geometry {
	int width, height;
};

struct window_buffer {
	VkImageView view;
	VkFramebuffer framebuffer;
	VkFence fence;
	VkCommandBuffer cmd_buffer;
};

struct window {
	struct display *display;
	struct geometry window_size;
	struct geometry logical_size;
	struct geometry buffer_size;
	int32_t buffer_scale;
	double fractional_buffer_scale;
	enum wl_output_transform buffer_transform;
	bool needs_buffer_geometry_update;

	struct {
		VkSwapchainKHR swap_chain;

		VkInstance instance;
		VkPhysicalDevice physical_device;
		VkPhysicalDeviceMemoryProperties memory_properties;
		VkDevice device;
		VkRenderPass render_pass;
		VkQueue queue;
		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;
		VkDeviceMemory mem;
		VkBuffer buffer;
		VkDescriptorSet descriptor_set;
		VkSemaphore image_semaphore;
		VkSemaphore render_semaphore;
		VkCommandPool cmd_pool;

		void *map;
		uint32_t vertex_offset, colors_offset;

		VkSurfaceKHR surface;
		VkFormat image_format;
		struct window_buffer buffers[MAX_NUM_IMAGES];
		uint32_t image_count;

		struct {
			float rotation[16];
		} ubo;
	} vk;

	VkPresentModeKHR present_mode;
	uint32_t frames;
	uint32_t initial_frame_time;
	uint32_t benchmark_time;
	struct wl_surface *surface;
	struct xdg_surface *xdg_surface;
	struct xdg_toplevel *xdg_toplevel;
	int fullscreen, maximized, opaque, delay;
	struct wp_tearing_control_v1 *tear_control;
	struct wp_viewport *viewport;
	struct wp_fractional_scale_v1 *fractional_scale_obj;
	bool tearing, toggled_tearing, tear_enabled;
	bool fullscreen_ratio;
	bool wait_for_configure;

	struct wl_list window_output_list; /* struct window_output::link */
};

struct output {
	struct display *display;
	struct wl_output *wl_output;
	uint32_t name;
	struct wl_list link; /* struct display::output_list */
	enum wl_output_transform transform;
	int32_t scale;
};

struct window_output {
	struct output *output;
	struct wl_list link; /* struct window::window_output_list */
};

static uint32_t vs_spirv_source[] = {
#include "simple-vulkan.vert.spv.h"
};

static uint32_t fs_spirv_source[] = {
#include "simple-vulkan.frag.spv.h"
};

static int running = 1;

static int
find_host_coherent_memory(struct window *window, unsigned allowed)
{
	for (unsigned i = 0; (1u << i) <= allowed && i <= window->vk.memory_properties.memoryTypeCount; ++i) {
		if ((allowed & (1u << i)) &&
				(window->vk.memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
				(window->vk.memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
			return i;
	}
	return -1;
}

static int32_t
compute_buffer_scale(struct window *window)
{
	struct window_output *window_output;
	int32_t scale = 1;

	wl_list_for_each(window_output, &window->window_output_list, link) {
		if (window_output->output->scale > scale)
			scale = window_output->output->scale;
	}

	return scale;
}

static enum wl_output_transform
compute_buffer_transform(struct window *window)
{
	struct window_output *window_output;
	enum wl_output_transform transform = WL_OUTPUT_TRANSFORM_NORMAL;

	wl_list_for_each(window_output, &window->window_output_list, link) {
		/* If the surface spans over multiple outputs the optimal
		 * transform value can be ambiguous. Thus just return the value
		 * from the oldest entered output.
		 */
		transform = window_output->output->transform;
		break;
	}

	return transform;
}

static void
create_swapchain(struct window *window)
{
	VkSurfaceCapabilitiesKHR surface_caps;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(window->vk.physical_device, window->vk.surface,
			&surface_caps);
	assert(surface_caps.supportedCompositeAlpha &
			VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR);

	VkBool32 supported;
	vkGetPhysicalDeviceSurfaceSupportKHR(window->vk.physical_device, 0, window->vk.surface,
			&supported);
	assert(supported);

	uint32_t present_mode_count;
	vkGetPhysicalDeviceSurfacePresentModesKHR(window->vk.physical_device,
			window->vk.surface, &present_mode_count, NULL);
	VkPresentModeKHR present_modes[present_mode_count];
	vkGetPhysicalDeviceSurfacePresentModesKHR(window->vk.physical_device,
			window->vk.surface, &present_mode_count, present_modes);

	assert(window->present_mode >= 0 && window->present_mode < 4);
	supported = false;
	for (size_t i = 0; i < present_mode_count; ++i) {
		if (present_modes[i] == window->present_mode) {
			supported = true;
			break;
		}
	}

	if (!supported) {
		fprintf(stderr, "Present mode %d unsupported\n", window->present_mode);
		abort();
	}

	uint32_t min_image_count = 2;
	if (min_image_count < surface_caps.minImageCount) {
		if (surface_caps.minImageCount > MAX_NUM_IMAGES)
			fprintf(stderr, "surface_caps.min_image_count is too large (is: %d, max: %d)",
					surface_caps.minImageCount, MAX_NUM_IMAGES);
		min_image_count = surface_caps.minImageCount;
	}

	if (surface_caps.maxImageCount > 0 &&
			min_image_count > surface_caps.maxImageCount) {
		min_image_count = surface_caps.maxImageCount;
	}

	const uint32_t queue_family_indices[] = { 0 };
	const VkSwapchainCreateInfoKHR swapchain_create_info = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.flags = 0,
		.surface = window->vk.surface,
		.minImageCount = min_image_count,
		.imageFormat = window->vk.image_format,
		.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
		.imageExtent = { window->buffer_size.width, window->buffer_size.height },
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 1,
		.pQueueFamilyIndices = queue_family_indices,
		.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.compositeAlpha = window->opaque ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
		.presentMode = window->present_mode,
	};
	vkCreateSwapchainKHR(window->vk.device, &swapchain_create_info, NULL, &window->vk.swap_chain);

	vkGetSwapchainImagesKHR(window->vk.device, window->vk.swap_chain,
			&window->vk.image_count, NULL);
	assert(window->vk.image_count > 0);
	VkImage swap_chain_images[window->vk.image_count];
	vkGetSwapchainImagesKHR(window->vk.device, window->vk.swap_chain,
			&window->vk.image_count, swap_chain_images);

	assert(window->vk.image_count <= MAX_NUM_IMAGES);
	for (uint32_t i = 0; i < window->vk.image_count; i++) {
		struct window_buffer *b = &window->vk.buffers[i];

		const VkImageViewCreateInfo imageview_info = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swap_chain_images[i],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = window->vk.image_format,
			.components = {
				.r = VK_COMPONENT_SWIZZLE_R,
				.g = VK_COMPONENT_SWIZZLE_G,
				.b = VK_COMPONENT_SWIZZLE_B,
				.a = VK_COMPONENT_SWIZZLE_A,
			},
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCreateImageView(window->vk.device, &imageview_info, NULL, &b->view);

		const VkFramebufferCreateInfo framebuffer_create_info = {
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = window->vk.render_pass,
			.attachmentCount = 1,
			.pAttachments = &b->view,
			.width = window->buffer_size.width,
			.height = window->buffer_size.height,
			.layers = 1
		};
		vkCreateFramebuffer(window->vk.device, &framebuffer_create_info, NULL, &b->framebuffer);

		const VkFenceCreateInfo fence_create_info = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT
		};
		vkCreateFence(window->vk.device, &fence_create_info, NULL, &b->fence);

		const VkCommandBufferAllocateInfo commandbuffer_allocate_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = window->vk.cmd_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		vkAllocateCommandBuffers(window->vk.device, &commandbuffer_allocate_info, &b->cmd_buffer);
	}

	VkSemaphoreCreateInfo semaphore_create_info = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
	};
	vkCreateSemaphore(window->vk.device, &semaphore_create_info, NULL, &window->vk.image_semaphore);
	vkCreateSemaphore(window->vk.device, &semaphore_create_info, NULL, &window->vk.render_semaphore);
}

static void
destroy_swapchain(struct window *window)
{
	VkSwapchainKHR old_chain = window->vk.swap_chain;

	for (uint32_t i = 0; i < window->vk.image_count; i++) {
		struct window_buffer *b = &window->vk.buffers[i];
		vkFreeCommandBuffers(window->vk.device, window->vk.cmd_pool, 1, &b->cmd_buffer);
		vkDestroyFence(window->vk.device, b->fence, NULL);
		vkDestroyFramebuffer(window->vk.device, b->framebuffer, NULL);
		vkDestroyImageView(window->vk.device, b->view, NULL);
	}

	vkDestroySwapchainKHR(window->vk.device, old_chain, NULL);

	vkDestroySemaphore(window->vk.device, window->vk.image_semaphore, NULL);
	vkDestroySemaphore(window->vk.device, window->vk.render_semaphore, NULL);
}

static void
recreate_swapchain(struct window *window)
{
	destroy_swapchain(window);
	create_swapchain(window);
}

static void
update_buffer_geometry(struct window *window)
{
	enum wl_output_transform new_buffer_transform;
	struct geometry new_buffer_size;
	struct geometry new_viewport_dest_size;

	new_buffer_transform = compute_buffer_transform(window);
	if (window->buffer_transform != new_buffer_transform) {
		window->buffer_transform = new_buffer_transform;
		wl_surface_set_buffer_transform(window->surface,
						window->buffer_transform);
	}

	switch (window->buffer_transform) {
	case WL_OUTPUT_TRANSFORM_NORMAL:
	case WL_OUTPUT_TRANSFORM_180:
	case WL_OUTPUT_TRANSFORM_FLIPPED:
	case WL_OUTPUT_TRANSFORM_FLIPPED_180:
		new_buffer_size.width = window->logical_size.width;
		new_buffer_size.height = window->logical_size.height;
		break;
	case WL_OUTPUT_TRANSFORM_90:
	case WL_OUTPUT_TRANSFORM_270:
	case WL_OUTPUT_TRANSFORM_FLIPPED_90:
	case WL_OUTPUT_TRANSFORM_FLIPPED_270:
		new_buffer_size.width = window->logical_size.height;
		new_buffer_size.height = window->logical_size.width;
		break;
	}

	if (window->fractional_buffer_scale > 0.0) {
		if (window->buffer_scale > 1) {
			window->buffer_scale = 1;
			wl_surface_set_buffer_scale(window->surface,
						    window->buffer_scale);
		}

		new_buffer_size.width = ceil(new_buffer_size.width *
					     window->fractional_buffer_scale);
		new_buffer_size.height = ceil(new_buffer_size.height *
					      window->fractional_buffer_scale);
	} else {
		int32_t new_buffer_scale;

		new_buffer_scale = compute_buffer_scale(window);
		if (window->buffer_scale != new_buffer_scale) {
			window->buffer_scale = new_buffer_scale;
			wl_surface_set_buffer_scale(window->surface,
						    window->buffer_scale);
		}

		new_buffer_size.width *= window->buffer_scale;
		new_buffer_size.height *= window->buffer_scale;
	}

	if (window->fullscreen && window->fullscreen_ratio) {
		int new_buffer_size_min;
		int new_viewport_dest_size_min;

		new_buffer_size_min = MIN(new_buffer_size.width,
					  new_buffer_size.height);
		new_buffer_size.width = new_buffer_size_min;
		new_buffer_size.height = new_buffer_size_min;

		new_viewport_dest_size_min = MIN(window->logical_size.width,
						 window->logical_size.height);
		new_viewport_dest_size.width = new_viewport_dest_size_min;
		new_viewport_dest_size.height = new_viewport_dest_size_min;
	} else {
		new_viewport_dest_size.width = window->logical_size.width;
		new_viewport_dest_size.height = window->logical_size.height;
	}

	if (window->buffer_size.width != new_buffer_size.width ||
	    window->buffer_size.height != new_buffer_size.height) {
		window->buffer_size = new_buffer_size;
	}

	if (window->fractional_buffer_scale > 0.0)
		wp_viewport_set_destination(window->viewport,
					    new_viewport_dest_size.width,
					    new_viewport_dest_size.height);

	window->needs_buffer_geometry_update = false;
}

static VkFormat
choose_surface_format(struct window *window)
{
	uint32_t num_formats = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(window->vk.physical_device, window->vk.surface,
			&num_formats, NULL);
	assert(num_formats > 0);

	VkSurfaceFormatKHR formats[num_formats];

	vkGetPhysicalDeviceSurfaceFormatsKHR(window->vk.physical_device, window->vk.surface,
			&num_formats, formats);

	VkFormat format = VK_FORMAT_UNDEFINED;
	for (int i = 0; i < (int)num_formats; i++) {
		switch (formats[i].format) {
			case VK_FORMAT_B8G8R8A8_UNORM:
				format = formats[i].format;
				break;
			default:
				continue;
		}
	}

	assert(format != VK_FORMAT_UNDEFINED);

	return format;
}

static void
init_vulkan(struct window *window)
{
	VkResult result;

	const char *extension = VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;

	if (window->needs_buffer_geometry_update)
		update_buffer_geometry(window);

	const VkApplicationInfo application_info = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "window",
		.apiVersion = VK_MAKE_VERSION(1, 1, 0),
	};
	const VkInstanceCreateInfo instance_create_info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &application_info,
		.enabledExtensionCount = extension ? 2 : 0,
		.ppEnabledExtensionNames = (const char *[2]) {
			VK_KHR_SURFACE_EXTENSION_NAME,
			extension,
		},
	};

	result = vkCreateInstance(&instance_create_info, NULL, &window->vk.instance);
	if (result != VK_SUCCESS)
		fprintf(stderr, "Failed to create Vulkan instance.\n");

	uint32_t count;
	result = vkEnumeratePhysicalDevices(window->vk.instance, &count, NULL);
	if (result != VK_SUCCESS || count == 0) {
		fprintf(stderr, "No Vulkan devices found.\n");
		abort();
	}

	VkPhysicalDevice pd[count];
	vkEnumeratePhysicalDevices(window->vk.instance, &count, pd);
	window->vk.physical_device = pd[0];

	vkGetPhysicalDeviceMemoryProperties(window->vk.physical_device, &window->vk.memory_properties);

	vkGetPhysicalDeviceQueueFamilyProperties(window->vk.physical_device, &count, NULL);
	assert(count > 0);
	VkQueueFamilyProperties props[count];
	vkGetPhysicalDeviceQueueFamilyProperties(window->vk.physical_device, &count, props);
	assert(props[0].queueFlags & VK_QUEUE_GRAPHICS_BIT);

	const float queue_priority = 1.0f;
	const VkDeviceQueueCreateInfo device_queue_create_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = 0,
		.queueCount = 1,
		.flags = 0,
		.pQueuePriorities = &queue_priority,
	};
	const char * const enabled_extension_names[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	const VkDeviceCreateInfo device_create_info = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &device_queue_create_info,
		.enabledExtensionCount = 1,
		.ppEnabledExtensionNames = enabled_extension_names,
	};
	vkCreateDevice(window->vk.physical_device, &device_create_info, NULL, &window->vk.device);

	vkGetDeviceQueue(window->vk.device, 0, 0, &window->vk.queue);

	PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR get_wayland_presentation_support =
		(PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR)
		vkGetInstanceProcAddr(window->vk.instance, "vkGetPhysicalDeviceWaylandPresentationSupportKHR");
	PFN_vkCreateWaylandSurfaceKHR create_wayland_surface =
		(PFN_vkCreateWaylandSurfaceKHR)
		vkGetInstanceProcAddr(window->vk.instance, "vkCreateWaylandSurfaceKHR");

	if (!get_wayland_presentation_support(window->vk.physical_device, 0,
				window->display->display)) {
		fprintf(stderr, "Vulkan not supported on given Wayland surface");
	}

	const VkWaylandSurfaceCreateInfoKHR wayland_surface_create_info = {
		.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
		.display = window->display->display,
		.surface = window->surface,
	};
	create_wayland_surface(window->vk.instance, &wayland_surface_create_info, NULL, &window->vk.surface);

	window->vk.image_format = choose_surface_format(window);

	const VkAttachmentDescription attachment_description[] = {
		{
			.format = window->vk.image_format,
			.samples = 1,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		},
	};
	const VkSubpassDescription subpass_description[] = {
		{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.colorAttachmentCount = 1,
			.pColorAttachments = (VkAttachmentReference []) {
				{
					.attachment = 0,
					.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
				}
			},
			.pResolveAttachments = (VkAttachmentReference []) {
				{
					.attachment = VK_ATTACHMENT_UNUSED,
					.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
				}
			},
			.pDepthStencilAttachment = NULL,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL,
		}
	};
	const VkRenderPassCreateInfo renderpass_create_info = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = attachment_description,
		.subpassCount = 1,
		.pSubpasses = subpass_description,
	};
	vkCreateRenderPass(window->vk.device, &renderpass_create_info, NULL, &window->vk.render_pass);

	VkDescriptorSetLayout set_layout;
	const VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = (VkDescriptorSetLayoutBinding[]) {
			{
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			}
		}
	};
	vkCreateDescriptorSetLayout(window->vk.device, &descriptor_set_layout_create_info, NULL, &set_layout);

	const VkPipelineLayoutCreateInfo pipeline_layout_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &set_layout,
	};
	vkCreatePipelineLayout(window->vk.device, &pipeline_layout_create_info, NULL, &window->vk.pipeline_layout);

	VkShaderModule vs_module;
	const VkShaderModuleCreateInfo vs_shader_module_create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(vs_spirv_source),
		.pCode = (uint32_t *)vs_spirv_source,
	};
	vkCreateShaderModule(window->vk.device, &vs_shader_module_create_info, NULL, &vs_module);

	VkShaderModule fs_module;
	const VkShaderModuleCreateInfo fs_shader_module_create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = sizeof(fs_spirv_source),
		.pCode = (uint32_t *)fs_spirv_source,
	};
	vkCreateShaderModule(window->vk.device, &fs_shader_module_create_info, NULL, &fs_module);

	const VkPipelineVertexInputStateCreateInfo pipeline_vertex_input_state_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 2,
		.pVertexBindingDescriptions = (VkVertexInputBindingDescription[]) {
			{
				.binding = 0,
				.stride = 3 * sizeof(float),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
			},
			{
				.binding = 1,
				.stride = 3 * sizeof(float),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
			},
		},
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]) {
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = 0
			},
			{
				.location = 1,
				.binding = 1,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = 0
			},
		}
	};
	const VkPipelineInputAssemblyStateCreateInfo pipeline_input_assembly_state_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = false,
	};
	const VkPipelineViewportStateCreateInfo pipeline_viewport_state_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo pipeline_rasterization_state_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.rasterizerDiscardEnable = false,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthClampEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo pipeline_multisample_state_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = 1,
	};
	const VkPipelineColorBlendStateCreateInfo pipeline_color_blend_state_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = (VkPipelineColorBlendAttachmentState []) {
			{ .colorWriteMask = VK_COLOR_COMPONENT_A_BIT |
				VK_COLOR_COMPONENT_R_BIT |
					VK_COLOR_COMPONENT_G_BIT |
					VK_COLOR_COMPONENT_B_BIT },
		}
	};
	const VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_create_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = (VkDynamicState[]) {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
		},
	};

	const VkGraphicsPipelineCreateInfo graphics_pipeline_create_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = (VkPipelineShaderStageCreateInfo[]) {
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_VERTEX_BIT,
				.module = vs_module,
				.pName = "main",
			},
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
				.module = fs_module,
				.pName = "main",
			},
		},
		.pVertexInputState = &pipeline_vertex_input_state_create_info,
		.pInputAssemblyState = &pipeline_input_assembly_state_create_info,
		.pViewportState = &pipeline_viewport_state_create_info,
		.pRasterizationState = &pipeline_rasterization_state_create_info,
		.pMultisampleState = &pipeline_multisample_state_create_info,
		.pColorBlendState = &pipeline_color_blend_state_create_info,
		.pDynamicState = &pipeline_dynamic_state_create_info,

		.flags = 0,
		.layout = window->vk.pipeline_layout,
		.renderPass = window->vk.render_pass,
		.subpass = 0,
	};
	vkCreateGraphicsPipelines(window->vk.device, VK_NULL_HANDLE, 1, &graphics_pipeline_create_info, NULL, &window->vk.pipeline);

	static const float vVertices[] = {
		-0.5f, -0.5f, 0.0,
		 0.5f, -0.5f, 0.0,
		 0.0f,  0.5f, 0.0,
	};

	static const float vColors[] = {
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
	};

	window->vk.vertex_offset = sizeof(window->vk.ubo);
	window->vk.colors_offset = window->vk.vertex_offset + sizeof(vVertices);
	uint32_t mem_size = window->vk.colors_offset + sizeof(vColors);

	const VkBufferCreateInfo buffer_create_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = mem_size,
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.flags = 0
	};
	vkCreateBuffer(window->vk.device, &buffer_create_info, NULL, &window->vk.buffer);

	VkMemoryRequirements reqs;
	vkGetBufferMemoryRequirements(window->vk.device, window->vk.buffer, &reqs);

	int memory_type = find_host_coherent_memory(window, reqs.memoryTypeBits);
	if (memory_type < 0)
		fprintf(stderr, "find_host_coherent_memory failed");

	const VkMemoryAllocateInfo memory_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_size,
		.memoryTypeIndex = memory_type,
	};
	vkAllocateMemory(window->vk.device, &memory_allocate_info, NULL, &window->vk.mem);

	result = vkMapMemory(window->vk.device, window->vk.mem, 0, mem_size, 0, &window->vk.map);
	if (result != VK_SUCCESS)
		fprintf(stderr, "vkMapMemory failed");
	memcpy(window->vk.map + window->vk.vertex_offset, vVertices, sizeof(vVertices));
	memcpy(window->vk.map + window->vk.colors_offset, vColors, sizeof(vColors));

	vkBindBufferMemory(window->vk.device, window->vk.buffer, window->vk.mem, 0);

	VkDescriptorPool desc_pool;
	const VkDescriptorPoolCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = (VkDescriptorPoolSize[]) {
			{
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1
			},
		}
	};

	vkCreateDescriptorPool(window->vk.device, &info, NULL, &desc_pool);

	const VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = desc_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &set_layout,
	};
	vkAllocateDescriptorSets(window->vk.device, &descriptor_set_allocate_info, &window->vk.descriptor_set);

	const VkDescriptorBufferInfo descriptor_buffer_info = {
		.buffer = window->vk.buffer,
		.offset = 0,
		.range = sizeof(window->vk.ubo),
	};
	vkUpdateDescriptorSets(window->vk.device, 1,
			(VkWriteDescriptorSet []) {
			{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = window->vk.descriptor_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &descriptor_buffer_info,
			}
			},
			0, NULL);

	const VkCommandPoolCreateInfo command_pool_create_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.queueFamilyIndex = 0,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
	};
	vkCreateCommandPool(window->vk.device, &command_pool_create_info, NULL, &window->vk.cmd_pool);

}

static void
handle_surface_configure(void *data, struct xdg_surface *surface,
			 uint32_t serial)
{
	struct window *window = data;

	xdg_surface_ack_configure(surface, serial);

	window->wait_for_configure = false;
}

static const struct xdg_surface_listener xdg_surface_listener = {
	handle_surface_configure
};

static void
handle_toplevel_configure(void *data, struct xdg_toplevel *toplevel,
			  int32_t width, int32_t height,
			  struct wl_array *states)
{
	struct window *window = data;
	uint32_t *p;

	window->fullscreen = 0;
	window->maximized = 0;
	wl_array_for_each(p, states) {
		uint32_t state = *p;
		switch (state) {
		case XDG_TOPLEVEL_STATE_FULLSCREEN:
			window->fullscreen = 1;
			break;
		case XDG_TOPLEVEL_STATE_MAXIMIZED:
			window->maximized = 1;
			break;
		}
	}

	if (width > 0 && height > 0) {
		if (!window->fullscreen && !window->maximized) {
			window->window_size.width = width;
			window->window_size.height = height;
		}
		window->logical_size.width = width;
		window->logical_size.height = height;
	} else if (!window->fullscreen && !window->maximized) {
		window->logical_size = window->window_size;
	}

	window->needs_buffer_geometry_update = true;
}

static void
handle_toplevel_close(void *data, struct xdg_toplevel *xdg_toplevel)
{
	running = 0;
}

static const struct xdg_toplevel_listener xdg_toplevel_listener = {
	handle_toplevel_configure,
	handle_toplevel_close,
};

static void
add_window_output(struct window *window, struct wl_output *wl_output)
{
	struct output *output;
	struct output *output_found = NULL;
	struct window_output *window_output;

	wl_list_for_each(output, &window->display->output_list, link) {
		if (output->wl_output == wl_output) {
			output_found = output;
			break;
		}
	}

	if (!output_found)
		return;

	window_output = xmalloc(sizeof *window_output);
	window_output->output = output_found;

	wl_list_insert(window->window_output_list.prev, &window_output->link);
	window->needs_buffer_geometry_update = true;
}

static void
destroy_window_output(struct window *window, struct wl_output *wl_output)
{
	struct window_output *window_output;
	struct window_output *window_output_found = NULL;

	wl_list_for_each(window_output, &window->window_output_list, link) {
		if (window_output->output->wl_output == wl_output) {
			window_output_found = window_output;
			break;
		}
	}

	if (window_output_found) {
		wl_list_remove(&window_output_found->link);
		free(window_output_found);
		window->needs_buffer_geometry_update = true;
	}
}

static void
draw_triangle(struct window *window, struct window_buffer *b)
{
	vkWaitForFences(window->vk.device, 1, &b->fence, VK_TRUE, UINT64_MAX);
	vkResetFences(window->vk.device, 1, &b->fence);

	const VkCommandBufferBeginInfo command_buffer_begin_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = 0
	};
	vkBeginCommandBuffer(b->cmd_buffer, &command_buffer_begin_info);

	const VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 0.5f}}};
	const VkRenderPassBeginInfo render_pass_begin_info = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = window->vk.render_pass,
		.framebuffer = b->framebuffer,
		.renderArea.offset = {0, 0},
		.renderArea.extent = {window->buffer_size.width, window->buffer_size.height},
		.clearValueCount = 1,
		.pClearValues = &clear_color,
	};
	vkCmdBeginRenderPass(b->cmd_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindVertexBuffers(b->cmd_buffer, 0, 2,
			(VkBuffer[]) {
			window->vk.buffer,
			window->vk.buffer,
			},
			(VkDeviceSize[]) {
			window->vk.vertex_offset,
			window->vk.colors_offset,
			});

	vkCmdBindPipeline(b->cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, window->vk.pipeline);

	vkCmdBindDescriptorSets(b->cmd_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			window->vk.pipeline_layout,
			0, 1,
			&window->vk.descriptor_set, 0, NULL);

	const VkViewport viewport = {
		.x = 0,
		.y = 0,
		.width = window->buffer_size.width,
		.height = window->buffer_size.height,
		.minDepth = 0,
		.maxDepth = 1,
	};
	vkCmdSetViewport(b->cmd_buffer, 0, 1, &viewport);

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = { window->buffer_size.width, window->buffer_size.height, },
	};
	vkCmdSetScissor(b->cmd_buffer, 0, 1, &scissor);

	vkCmdDraw(b->cmd_buffer, 3, 1, 0, 0);

	vkCmdEndRenderPass(b->cmd_buffer);

	vkEndCommandBuffer(b->cmd_buffer);

	VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkSubmitInfo submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = &window->vk.image_semaphore;
	submit_info.pWaitDstStageMask = wait_stages;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &b->cmd_buffer;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &window->vk.render_semaphore;

	vkQueueSubmit(window->vk.queue, 1, &submit_info, b->fence);
	window->frames++;
}

static void
set_tearing(struct window *window, bool enable)
{
	if (!window->tear_control)
		return;

	if (enable) {
		wp_tearing_control_v1_set_presentation_hint(window->tear_control,
							    WP_TEARING_CONTROL_V1_PRESENTATION_HINT_ASYNC);
	} else {
		wp_tearing_control_v1_set_presentation_hint(window->tear_control,
							    WP_TEARING_CONTROL_V1_PRESENTATION_HINT_VSYNC);
	}
	window->tear_enabled = enable;
}

static void
surface_enter(void *data,
	      struct wl_surface *wl_surface, struct wl_output *wl_output)
{
	struct window *window = data;

	add_window_output(window, wl_output);
}

static void
surface_leave(void *data,
	      struct wl_surface *wl_surface, struct wl_output *wl_output)
{
	struct window *window = data;

	destroy_window_output(window, wl_output);
}

static const struct wl_surface_listener surface_listener = {
	surface_enter,
	surface_leave
};

static void fractional_scale_handle_preferred_scale(void *data,
						    struct wp_fractional_scale_v1 *info,
						    uint32_t wire_scale) {
	struct window *window = data;

	window->fractional_buffer_scale = wire_scale / 120.0;
	window->needs_buffer_geometry_update = true;
}

static const struct wp_fractional_scale_v1_listener fractional_scale_listener = {
	.preferred_scale = fractional_scale_handle_preferred_scale,
};

static void
create_surface(struct window *window)
{
	struct display *display = window->display;

	window->surface = wl_compositor_create_surface(display->compositor);
	wl_surface_add_listener(window->surface, &surface_listener, window);

	if (display->tearing_manager && window->tearing) {
		window->tear_control = wp_tearing_control_manager_v1_get_tearing_control(
			display->tearing_manager,
			window->surface);
		set_tearing(window, true);
	}

	window->xdg_surface = xdg_wm_base_get_xdg_surface(display->wm_base,
							  window->surface);
	xdg_surface_add_listener(window->xdg_surface,
				 &xdg_surface_listener, window);

	window->xdg_toplevel =
		xdg_surface_get_toplevel(window->xdg_surface);
	xdg_toplevel_add_listener(window->xdg_toplevel,
				  &xdg_toplevel_listener, window);

	xdg_toplevel_set_title(window->xdg_toplevel, "simple-vulkan");
	xdg_toplevel_set_app_id(window->xdg_toplevel,
			"org.freedesktop.weston.simple-vulkan");

	if (window->fullscreen)
		xdg_toplevel_set_fullscreen(window->xdg_toplevel, NULL);
	else if (window->maximized)
		xdg_toplevel_set_maximized(window->xdg_toplevel);

	if (display->viewporter && display->fractional_scale_manager) {
		window->viewport = wp_viewporter_get_viewport(display->viewporter,
							      window->surface);
		window->fractional_scale_obj =
			wp_fractional_scale_manager_v1_get_fractional_scale(display->fractional_scale_manager,
									    window->surface);
		wp_fractional_scale_v1_add_listener(window->fractional_scale_obj,
						    &fractional_scale_listener,
						    window);
	}

	window->wait_for_configure = true;
	wl_surface_commit(window->surface);
}

static void
destroy_surface(struct window *window)
{
	if (window->xdg_toplevel)
		xdg_toplevel_destroy(window->xdg_toplevel);
	if (window->xdg_surface)
		xdg_surface_destroy(window->xdg_surface);
	if (window->viewport)
		wp_viewport_destroy(window->viewport);
	if (window->fractional_scale_obj)
		wp_fractional_scale_v1_destroy(window->fractional_scale_obj);
	wl_surface_destroy(window->surface);
}


static void
redraw(struct window *window)
{
	float angle;
	struct weston_matrix rotation;
	static const uint32_t speed_div = 5, benchmark_interval = 5;
	struct timeval tv;
	VkResult result;
	uint32_t index;

	if (window->needs_buffer_geometry_update) {
		update_buffer_geometry(window);
		recreate_swapchain(window);
	}

	gettimeofday(&tv, NULL);
	uint32_t time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	if (window->frames == 0) {
		window->initial_frame_time = time;
		window->benchmark_time = time;
	}
	if (time - window->benchmark_time > (benchmark_interval * 1000)) {
		printf("%d frames in %d seconds: %f fps\n",
		       window->frames,
		       benchmark_interval,
		       (float) window->frames / benchmark_interval);
		window->benchmark_time = time;
		window->frames = 0;
		if (window->toggled_tearing)
			set_tearing(window, window->tear_enabled ^ true);
	}

	weston_matrix_init(&rotation);

	angle = ((time - window->initial_frame_time) / speed_div)
		% 360 * M_PI / 180.0;

	rotation.d[0] =   cos(angle);
	rotation.d[2] =   sin(angle);
	rotation.d[8] =  -sin(angle);
	rotation.d[10] =  cos(angle);
	/* Flip from OpenGL to Vulkan coordinates */
	rotation.d[5] *= -1.0;

	switch (window->buffer_transform) {
	default:
	case WL_OUTPUT_TRANSFORM_NORMAL:
	case WL_OUTPUT_TRANSFORM_FLIPPED:
		break;
	case WL_OUTPUT_TRANSFORM_90:
	case WL_OUTPUT_TRANSFORM_FLIPPED_90:
		weston_matrix_rotate_xy(&rotation, 0, 1);
		break;
	case WL_OUTPUT_TRANSFORM_180:
	case WL_OUTPUT_TRANSFORM_FLIPPED_180:
		weston_matrix_rotate_xy(&rotation, -1, 0);
		break;
	case WL_OUTPUT_TRANSFORM_270:
	case WL_OUTPUT_TRANSFORM_FLIPPED_270:
		weston_matrix_rotate_xy(&rotation, 0, -1);
		break;
	}

	memcpy(window->vk.map, &rotation.d, sizeof(rotation.d));

	result = vkAcquireNextImageKHR(window->vk.device, window->vk.swap_chain, UINT64_MAX,
			window->vk.image_semaphore, VK_NULL_HANDLE, &index);
	if (result == VK_SUBOPTIMAL_KHR) {
		recreate_swapchain(window);
		return;
	} else if (result != VK_SUCCESS) {
		assert(0);
	}
	assert(index <= MAX_NUM_IMAGES);

	draw_triangle(window, &window->vk.buffers[index]);

	VkPresentInfoKHR present_info = {};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &window->vk.render_semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &window->vk.swap_chain;
	present_info.pImageIndices = &index;
	present_info.pResults = NULL;

	result = vkQueuePresentKHR(window->vk.queue, &present_info);
	if (result != VK_SUCCESS)
		return;
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		recreate_swapchain(window);
		return;
	}
	else if (result != VK_SUCCESS) {
		assert(0);
	}

	vkQueueWaitIdle(window->vk.queue);
}

static void
pointer_handle_enter(void *data, struct wl_pointer *pointer,
		     uint32_t serial, struct wl_surface *surface,
		     wl_fixed_t sx, wl_fixed_t sy)
{
	struct display *display = data;
	struct wl_buffer *buffer;
	struct wl_cursor *cursor = display->default_cursor;
	struct wl_cursor_image *image;

	if (display->window->fullscreen)
		wl_pointer_set_cursor(pointer, serial, NULL, 0, 0);
	else if (cursor) {
		image = display->default_cursor->images[0];
		buffer = wl_cursor_image_get_buffer(image);
		if (!buffer)
			return;
		wl_pointer_set_cursor(pointer, serial,
				      display->cursor_surface,
				      image->hotspot_x,
				      image->hotspot_y);
		wl_surface_attach(display->cursor_surface, buffer, 0, 0);
		wl_surface_damage(display->cursor_surface, 0, 0,
				  image->width, image->height);
		wl_surface_commit(display->cursor_surface);
	}
}

static void
pointer_handle_leave(void *data, struct wl_pointer *pointer,
		     uint32_t serial, struct wl_surface *surface)
{
}

static void
pointer_handle_motion(void *data, struct wl_pointer *pointer,
		      uint32_t time, wl_fixed_t sx, wl_fixed_t sy)
{
}

static void
pointer_handle_button(void *data, struct wl_pointer *wl_pointer,
		      uint32_t serial, uint32_t time, uint32_t button,
		      uint32_t state)
{
	struct display *display = data;

	if (!display->window->xdg_toplevel)
		return;

	if (button == BTN_LEFT && state == WL_POINTER_BUTTON_STATE_PRESSED)
		xdg_toplevel_move(display->window->xdg_toplevel,
				  display->seat, serial);
}

static void
pointer_handle_axis(void *data, struct wl_pointer *wl_pointer,
		    uint32_t time, uint32_t axis, wl_fixed_t value)
{
}

static const struct wl_pointer_listener pointer_listener = {
	pointer_handle_enter,
	pointer_handle_leave,
	pointer_handle_motion,
	pointer_handle_button,
	pointer_handle_axis,
};

static void
touch_handle_down(void *data, struct wl_touch *wl_touch,
		  uint32_t serial, uint32_t time, struct wl_surface *surface,
		  int32_t id, wl_fixed_t x_w, wl_fixed_t y_w)
{
	struct display *d = (struct display *)data;

	if (!d->wm_base)
		return;

	xdg_toplevel_move(d->window->xdg_toplevel, d->seat, serial);
}

static void
touch_handle_up(void *data, struct wl_touch *wl_touch,
		uint32_t serial, uint32_t time, int32_t id)
{
}

static void
touch_handle_motion(void *data, struct wl_touch *wl_touch,
		    uint32_t time, int32_t id, wl_fixed_t x_w, wl_fixed_t y_w)
{
}

static void
touch_handle_frame(void *data, struct wl_touch *wl_touch)
{
}

static void
touch_handle_cancel(void *data, struct wl_touch *wl_touch)
{
}

static const struct wl_touch_listener touch_listener = {
	touch_handle_down,
	touch_handle_up,
	touch_handle_motion,
	touch_handle_frame,
	touch_handle_cancel,
};

static void
keyboard_handle_keymap(void *data, struct wl_keyboard *keyboard,
		       uint32_t format, int fd, uint32_t size)
{
	/* Just so we don’t leak the keymap fd */
	close(fd);
}

static void
keyboard_handle_enter(void *data, struct wl_keyboard *keyboard,
		      uint32_t serial, struct wl_surface *surface,
		      struct wl_array *keys)
{
}

static void
keyboard_handle_leave(void *data, struct wl_keyboard *keyboard,
		      uint32_t serial, struct wl_surface *surface)
{
}

static void
keyboard_handle_key(void *data, struct wl_keyboard *keyboard,
		    uint32_t serial, uint32_t time, uint32_t key,
		    uint32_t state)
{
	struct display *d = data;

	if (!d->wm_base)
		return;

	if (key == KEY_F11 && state) {
		if (d->window->fullscreen)
			xdg_toplevel_unset_fullscreen(d->window->xdg_toplevel);
		else
			xdg_toplevel_set_fullscreen(d->window->xdg_toplevel, NULL);
	} else if (key == KEY_ESC && state)
		running = 0;
}

static void
keyboard_handle_modifiers(void *data, struct wl_keyboard *keyboard,
			  uint32_t serial, uint32_t mods_depressed,
			  uint32_t mods_latched, uint32_t mods_locked,
			  uint32_t group)
{
}

static const struct wl_keyboard_listener keyboard_listener = {
	keyboard_handle_keymap,
	keyboard_handle_enter,
	keyboard_handle_leave,
	keyboard_handle_key,
	keyboard_handle_modifiers,
};

static void
seat_handle_capabilities(void *data, struct wl_seat *seat,
			 enum wl_seat_capability caps)
{
	struct display *d = data;

	if ((caps & WL_SEAT_CAPABILITY_POINTER) && !d->pointer) {
		d->pointer = wl_seat_get_pointer(seat);
		wl_pointer_add_listener(d->pointer, &pointer_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_POINTER) && d->pointer) {
		wl_pointer_destroy(d->pointer);
		d->pointer = NULL;
	}

	if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !d->keyboard) {
		d->keyboard = wl_seat_get_keyboard(seat);
		wl_keyboard_add_listener(d->keyboard, &keyboard_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_KEYBOARD) && d->keyboard) {
		wl_keyboard_destroy(d->keyboard);
		d->keyboard = NULL;
	}

	if ((caps & WL_SEAT_CAPABILITY_TOUCH) && !d->touch) {
		d->touch = wl_seat_get_touch(seat);
		wl_touch_set_user_data(d->touch, d);
		wl_touch_add_listener(d->touch, &touch_listener, d);
	} else if (!(caps & WL_SEAT_CAPABILITY_TOUCH) && d->touch) {
		wl_touch_destroy(d->touch);
		d->touch = NULL;
	}
}

static const struct wl_seat_listener seat_listener = {
	seat_handle_capabilities,
};

static void
xdg_wm_base_ping(void *data, struct xdg_wm_base *shell, uint32_t serial)
{
	xdg_wm_base_pong(shell, serial);
}

static const struct xdg_wm_base_listener wm_base_listener = {
	xdg_wm_base_ping,
};

static void
display_handle_geometry(void *data,
			struct wl_output *wl_output,
			int32_t x, int32_t y,
			int32_t physical_width,
			int32_t physical_height,
			int32_t subpixel,
			const char *make,
			const char *model,
			int32_t transform)
{
	struct output *output = data;

	output->transform = transform;
	output->display->window->needs_buffer_geometry_update = true;
}

static void
display_handle_mode(void *data,
		    struct wl_output *wl_output,
		    uint32_t flags,
		    int32_t width,
		    int32_t height,
		    int32_t refresh)
{
}

static void
display_handle_done(void *data,
		     struct wl_output *wl_output)
{
}

static void
display_handle_scale(void *data,
		     struct wl_output *wl_output,
		     int32_t scale)
{
	struct output *output = data;

	output->scale = scale;
	output->display->window->needs_buffer_geometry_update = true;
}

static const struct wl_output_listener output_listener = {
	display_handle_geometry,
	display_handle_mode,
	display_handle_done,
	display_handle_scale
};

static void
display_add_output(struct display *d, uint32_t name)
{
	struct output *output;

	output = xzalloc(sizeof *output);
	output->display = d;
	output->scale = 1;
	output->wl_output =
		wl_registry_bind(d->registry, name, &wl_output_interface, 2);
	output->name = name;
	wl_list_insert(d->output_list.prev, &output->link);

	wl_output_add_listener(output->wl_output, &output_listener, output);
}

static void
display_destroy_output(struct display *d, struct output *output)
{
	destroy_window_output(d->window, output->wl_output);
	wl_output_destroy(output->wl_output);
	wl_list_remove(&output->link);
	free(output);
}

static void
display_destroy_outputs(struct display *d)
{
	struct output *tmp;
	struct output *output;

	wl_list_for_each_safe(output, tmp, &d->output_list, link)
		display_destroy_output(d, output);
}

static void
registry_handle_global(void *data, struct wl_registry *registry,
		       uint32_t name, const char *interface, uint32_t version)
{
	struct display *d = data;

	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		d->compositor =
			wl_registry_bind(registry, name,
					 &wl_compositor_interface,
					 MIN(version, 4));
	} else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		d->wm_base = wl_registry_bind(registry, name,
					      &xdg_wm_base_interface, 1);
		xdg_wm_base_add_listener(d->wm_base, &wm_base_listener, d);
	} else if (strcmp(interface, wl_seat_interface.name) == 0) {
		d->seat = wl_registry_bind(registry, name,
					   &wl_seat_interface, 1);
		wl_seat_add_listener(d->seat, &seat_listener, d);
	} else if (strcmp(interface, wl_shm_interface.name) == 0) {
		d->shm = wl_registry_bind(registry, name,
					  &wl_shm_interface, 1);
		d->cursor_theme = wl_cursor_theme_load(NULL, 32, d->shm);
		if (!d->cursor_theme) {
			fprintf(stderr, "unable to load default theme\n");
			return;
		}
		d->default_cursor =
			wl_cursor_theme_get_cursor(d->cursor_theme, "left_ptr");
		if (!d->default_cursor) {
			fprintf(stderr, "unable to load default left pointer\n");
			// TODO: abort ?
		}
	} else if (strcmp(interface, wl_output_interface.name) == 0 && version >= 2) {
		display_add_output(d, name);
	} else if (strcmp(interface, wp_tearing_control_manager_v1_interface.name) == 0) {
		d->tearing_manager = wl_registry_bind(registry, name,
						      &wp_tearing_control_manager_v1_interface,
						      1);
	} else if (strcmp(interface, wp_viewporter_interface.name) == 0) {
		d->viewporter = wl_registry_bind(registry, name,
						 &wp_viewporter_interface,
						 1);
	} else if (strcmp(interface, wp_fractional_scale_manager_v1_interface.name) == 0) {
		d->fractional_scale_manager =
			wl_registry_bind(registry, name,
					 &wp_fractional_scale_manager_v1_interface,
					 1);
	}
}

static void
registry_handle_global_remove(void *data, struct wl_registry *registry,
			      uint32_t name)
{
	struct display *d = data;
	struct output *output;

	wl_list_for_each(output, &d->output_list, link) {
		if (output->name == name) {
			display_destroy_output(d, output);
			break;
		}
	}
}

static const struct wl_registry_listener registry_listener = {
	registry_handle_global,
	registry_handle_global_remove
};

static void
signal_int(int signum)
{
	running = 0;
}

static void
usage(int error_code)
{
	fprintf(stderr, "Usage: simple-vulkan [OPTIONS]\n\n"
		"  -d <us>\tBuffer swap delay in microseconds\n"
		"  -p <presentation mode>\tSet presentation mode\n"
		"     immediate = 0\n"
		"     mailbox = 1\n"
		"     fifo = 2 (default)\n"
		"     fifo_relaxed = 3\n"
		"  -f\tRun in fullscreen mode\n"
		"  -r\tUse fixed width/height ratio when run in fullscreen mode\n"
		"  -m\tRun in maximized mode\n"
		"  -o\tCreate an opaque surface\n"
		"  -t\tEnable tearing via the tearing_control protocol\n"
		"  -T\tEnable and disable tearing every 5 seconds\n"
		"  -h\tThis help text\n\n");

	exit(error_code);
}



int
main(int argc, char **argv)
{
	struct sigaction sigint;
	struct display display = {};
	struct window window = {};
	int i, ret = 0;

	window.display = &display;
	display.window = &window;
	window.buffer_size.width  = 250;
	window.buffer_size.height = 250;
	window.window_size = window.buffer_size;
	window.buffer_scale = 1;
	window.buffer_transform = WL_OUTPUT_TRANSFORM_NORMAL;
	window.needs_buffer_geometry_update = false;
	window.delay = 0;
	window.fullscreen_ratio = false;
	window.present_mode = VK_PRESENT_MODE_FIFO_KHR;

	wl_list_init(&display.output_list);
	wl_list_init(&window.window_output_list);

	for (i = 1; i < argc; i++) {
		if (strcmp("-d", argv[i]) == 0 && i+1 < argc)
			window.delay = atoi(argv[++i]);
		else if (strcmp("-p", argv[i]) == 0 && i+1 < argc) {
			window.present_mode = atoi(argv[++i]);
			assert(window.present_mode >= 0 && window.present_mode < 4);
		} else if (strcmp("-f", argv[i]) == 0)
			window.fullscreen = 1;
		else if (strcmp("-r", argv[i]) == 0)
			window.fullscreen_ratio = true;
		else if (strcmp("-m", argv[i]) == 0)
			window.maximized = 1;
		else if (strcmp("-o", argv[i]) == 0)
			window.opaque = 1;
		else if (strcmp("-t", argv[i]) == 0) {
			window.tearing = true;
		} else if (strcmp("-T", argv[i]) == 0) {
			window.tearing = true;
			window.toggled_tearing = true;
		}
		else if (strcmp("-h", argv[i]) == 0)
			usage(EXIT_SUCCESS);
		else
			usage(EXIT_FAILURE);
	}

	display.display = wl_display_connect(NULL);
	assert(display.display);

	display.registry = wl_display_get_registry(display.display);
	wl_registry_add_listener(display.registry,
				 &registry_listener, &display);

	wl_display_roundtrip(display.display);

	if (!display.wm_base) {
		fprintf(stderr, "xdg-shell support required. simple-vulkan exiting\n");
		goto out_no_xdg_shell;
	}

	create_surface(&window);

	/* we already have wait_for_configure set after create_surface() */
	while (running && ret != -1 && window.wait_for_configure) {
		ret = wl_display_dispatch(display.display);

		/* wait until xdg_surface::configure acks the new dimensions */
		if (window.wait_for_configure)
			continue;

		init_vulkan(&window);
	}

	create_swapchain(&window);

	display.cursor_surface =
		wl_compositor_create_surface(display.compositor);

	sigint.sa_handler = signal_int;
	sigemptyset(&sigint.sa_mask);
	sigint.sa_flags = SA_RESETHAND;
	sigaction(SIGINT, &sigint, NULL);

	while (running && ret != -1) {
		ret = wl_display_dispatch_pending(display.display);
		redraw(&window);
	}

	fprintf(stderr, "simple-vulkan exiting\n");

	destroy_surface(&window);
	destroy_swapchain(&window);

	wl_surface_destroy(display.cursor_surface);
out_no_xdg_shell:
	display_destroy_outputs(&display);

	if (display.cursor_theme)
		wl_cursor_theme_destroy(display.cursor_theme);

	if (display.shm)
		wl_shm_destroy(display.shm);

	if (display.pointer)
		wl_pointer_destroy(display.pointer);

	if (display.keyboard)
		wl_keyboard_destroy(display.keyboard);

	if (display.touch)
		wl_touch_destroy(display.touch);

	if (display.seat)
		wl_seat_destroy(display.seat);

	if (display.wm_base)
		xdg_wm_base_destroy(display.wm_base);

	if (display.compositor)
		wl_compositor_destroy(display.compositor);

	if (display.viewporter)
		wp_viewporter_destroy(display.viewporter);

	if (display.fractional_scale_manager)
		wp_fractional_scale_manager_v1_destroy(display.fractional_scale_manager);

	wl_registry_destroy(display.registry);
	wl_display_flush(display.display);
	wl_display_disconnect(display.display);

	return 0;
}
