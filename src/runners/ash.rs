//! ash runner implementation - Direct Vulkan API via ash

use crate::{
    error::{ChimeraError, Result},
    SortRunner,
};
use ash::{vk, Device, Entry, Instance};
use shared::{BitonicParams, WORKGROUP_SIZE};
use std::ffi::CString;

/// Vulkan-based runner for bitonic sort using raw Vulkan API via ash
pub struct AshRunner {
    _entry: Entry,
    instance: Instance,
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    device: Device,
    #[allow(dead_code)]
    queue_family_index: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device_name: String,
    // Cached pipeline resources
    pipeline: Option<vk::Pipeline>,
    pipeline_layout: Option<vk::PipelineLayout>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_pool: Option<vk::DescriptorPool>,
    shader_module: Option<vk::ShaderModule>,
}

impl AshRunner {
    /// Create a new Vulkan runner using raw Vulkan API via ash
    pub fn new() -> Result<Self> {
        unsafe {
            // Load Vulkan entry point
            let entry = Entry::load()
                .map_err(|e| ChimeraError::Other(format!("Failed to load Vulkan: {e:?}")))?;

            // Create instance
            let app_name = CString::new("Rust GPU Chimera Demo")
                .map_err(|e| ChimeraError::Other(e.to_string()))?;
            let engine_name =
                CString::new("rust-gpu-chimera").map_err(|e| ChimeraError::Other(e.to_string()))?;

            let app_info = vk::ApplicationInfo::default()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_2);

            // Enable portability extensions for MoltenVK on macOS
            #[allow(unused_mut)] // Only modified on macOS
            let mut extension_names = vec![];
            #[allow(unused_mut)] // Only modified on macOS
            let mut create_flags = vk::InstanceCreateFlags::empty();

            #[cfg(target_os = "macos")]
            {
                // Get available extensions
                let available_extensions = entry.enumerate_instance_extension_properties(None)?;
                let has_portability = available_extensions.iter().any(|ext| {
                    let name = std::ffi::CStr::from_ptr(ext.extension_name.as_ptr());
                    name.to_bytes() == b"VK_KHR_portability_enumeration"
                });

                if has_portability {
                    extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
                    create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
                }
            }

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_extension_names(&extension_names)
                    .flags(create_flags),
                None,
            )?;

            // Get physical device
            let physical_devices = instance.enumerate_physical_devices()?;
            let device_count = physical_devices.len();
            let physical_device = physical_devices
                .into_iter()
                .next()
                .ok_or(ChimeraError::NoVulkanDevice(device_count))?;

            // Get device properties
            let properties = instance.get_physical_device_properties(physical_device);
            let device_name = std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .to_string();

            let memory_properties = instance.get_physical_device_memory_properties(physical_device);

            // Find compute queue family
            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);
            let queue_family_index = queue_families
                .iter()
                .enumerate()
                .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(idx, _)| idx as u32)
                .ok_or(ChimeraError::NoComputeQueue)?;

            // Create logical device
            let queue_priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);

            let device_features = vk::PhysicalDeviceFeatures::default();

            // Enable device extensions
            #[allow(unused_mut)] // Only modified on macOS
            let mut device_extension_names = vec![];

            // Keep portability extension name alive for the duration of device creation
            #[cfg(target_os = "macos")]
            let _portability_subset_name = {
                // Get available device extensions
                let available_device_extensions =
                    instance.enumerate_device_extension_properties(physical_device)?;
                let has_portability_subset = available_device_extensions.iter().any(|ext| {
                    let name = std::ffi::CStr::from_ptr(ext.extension_name.as_ptr());
                    name.to_bytes() == b"VK_KHR_portability_subset"
                });

                if has_portability_subset {
                    let name = CString::new("VK_KHR_portability_subset").unwrap();
                    device_extension_names.push(name.as_ptr());
                    Some(name)
                } else {
                    None
                }
            };

            let device = instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&[queue_info])
                    .enabled_features(&device_features)
                    .enabled_extension_names(&device_extension_names),
                None,
            )?;

            // Get queue
            let queue = device.get_device_queue(queue_family_index, 0);

            // Create command pool
            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family_index),
                None,
            )?;

            let mut runner = Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue_family_index,
                queue,
                command_pool,
                memory_properties,
                device_name,
                pipeline: None,
                pipeline_layout: None,
                descriptor_set_layout: None,
                descriptor_pool: None,
                shader_module: None,
            };

            // Initialize the pipeline
            runner.create_pipeline()?;

            Ok(runner)
        }
    }

    fn create_pipeline(&mut self) -> Result<()> {
        unsafe {
            // Use the embedded kernel from the main crate
            let kernel_bytes = crate::BITONIC_SPIRV;
            let kernel_code = ash::util::read_spv(&mut std::io::Cursor::new(kernel_bytes))?;
            let shader_module = self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&kernel_code),
                None,
            )?;

            // Create descriptor set layout for 1 buffer
            let descriptor_set_layout = self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                ]),
                None,
            )?;

            // Create pipeline layout with push constants
            let pipeline_layout = self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[descriptor_set_layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .offset(0)
                        .size(std::mem::size_of::<BitonicParams>() as u32)]),
                None,
            )?;

            // Create compute pipeline
            let entry_point = option_env!("BITONIC_KERNEL_SPV_ENTRY").unwrap_or("bitonic_kernel");
            let entry_name =
                CString::new(entry_point).map_err(|e| ChimeraError::Other(e.to_string()))?;

            let pipeline = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(shader_module)
                                .name(&entry_name),
                        )
                        .layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0];

            // Create descriptor pool
            let descriptor_pool = self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1) // Only need 1 set at a time since we reset the pool
                    .pool_sizes(&[
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1), // Only need 1 descriptor per set
                    ])
                    .flags(vk::DescriptorPoolCreateFlags::empty()), // Allow resetting
                None,
            )?;

            // Store the created resources
            self.shader_module = Some(shader_module);
            self.descriptor_set_layout = Some(descriptor_set_layout);
            self.pipeline_layout = Some(pipeline_layout);
            self.pipeline = Some(pipeline);
            self.descriptor_pool = Some(descriptor_pool);

            Ok(())
        }
    }

    fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        (0..self.memory_properties.memory_type_count)
            .find(|&i| {
                (type_filter & (1 << i)) != 0
                    && self.memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
            })
            .ok_or_else(|| ChimeraError::Other("Failed to find suitable memory type".to_string()))
    }
}

impl SortRunner for AshRunner {
    fn backend_info(
        &self,
    ) -> (
        &'static str,
        Option<&'static str>,
        Option<String>,
        Option<String>,
    ) {
        ("ash", Some("Vulkan"), Some(self.device_name.clone()), None)
    }

    fn execute_kernel_pass(&self, data: &mut [u32], params: BitonicParams) -> Result<()> {
        self.run_bitonic_kernel_single_pass(data, params)
    }
}

impl AshRunner {
    fn run_bitonic_kernel_single_pass(
        &self,
        data: &mut [u32],
        params: BitonicParams,
    ) -> Result<()> {
        unsafe {
            let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
            let workgroup_size = WORKGROUP_SIZE;

            // Create data buffer
            let data_buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(buffer_size)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;

            // Allocate memory for buffer
            let data_mem_reqs = self.device.get_buffer_memory_requirements(data_buffer);

            let memory_type_index = self.find_memory_type(
                data_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let data_memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(data_mem_reqs.size)
                    .memory_type_index(memory_type_index),
                None,
            )?;

            self.device
                .bind_buffer_memory(data_buffer, data_memory, 0)?;

            // Copy input data
            let data_ptr =
                self.device
                    .map_memory(data_memory, 0, buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr as *mut u32, data.len());
            self.device.unmap_memory(data_memory);

            // Use cached pipeline resources
            let pipeline = self
                .pipeline
                .ok_or_else(|| ChimeraError::Other("Pipeline not initialized".to_string()))?;
            let pipeline_layout = self.pipeline_layout.ok_or_else(|| {
                ChimeraError::Other("Pipeline layout not initialized".to_string())
            })?;
            let descriptor_set_layout = self.descriptor_set_layout.ok_or_else(|| {
                ChimeraError::Other("Descriptor set layout not initialized".to_string())
            })?;
            let descriptor_pool = self.descriptor_pool.ok_or_else(|| {
                ChimeraError::Other("Descriptor pool not initialized".to_string())
            })?;

            // Reset descriptor pool to avoid fragmentation
            self.device
                .reset_descriptor_pool(descriptor_pool, vk::DescriptorPoolResetFlags::empty())?;

            // Allocate descriptor set
            let descriptor_set = self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[descriptor_set_layout]),
            )?[0];

            // Update descriptor set
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[vk::DescriptorBufferInfo::default()
                        .buffer(data_buffer)
                        .offset(0)
                        .range(buffer_size)])],
                &[],
            );

            // Create command buffer
            let command_buffer = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];

            // Record commands for this single pass
            self.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            // Push constants - use the params passed in
            let push_data = bytemuck::bytes_of(&params);
            self.device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_data,
            );

            // Dispatch workgroups
            let num_workgroups = params.num_elements.div_ceil(workgroup_size);
            self.device
                .cmd_dispatch(command_buffer, num_workgroups, 1, 1);

            self.device.end_command_buffer(command_buffer)?;

            // Submit and wait
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;

            // Read results
            let data_ptr =
                self.device
                    .map_memory(data_memory, 0, buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data_ptr as *const u32, data.as_mut_ptr(), data.len());
            self.device.unmap_memory(data_memory);

            // Cleanup (only temporary resources, not cached ones)
            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
            self.device.free_memory(data_memory, None);
            self.device.destroy_buffer(data_buffer, None);

            Ok(())
        }
    }
}

impl Drop for AshRunner {
    fn drop(&mut self) {
        unsafe {
            // Destroy cached pipeline resources
            if let Some(pipeline) = self.pipeline {
                self.device.destroy_pipeline(pipeline, None);
            }
            if let Some(pipeline_layout) = self.pipeline_layout {
                self.device.destroy_pipeline_layout(pipeline_layout, None);
            }
            if let Some(descriptor_set_layout) = self.descriptor_set_layout {
                self.device
                    .destroy_descriptor_set_layout(descriptor_set_layout, None);
            }
            if let Some(descriptor_pool) = self.descriptor_pool {
                self.device.destroy_descriptor_pool(descriptor_pool, None);
            }
            if let Some(shader_module) = self.shader_module {
                self.device.destroy_shader_module(shader_module, None);
            }

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AshRunner;
    use crate::{verify_sorted, SortRunner};
    use shared::SortOrder;

    #[test]
    fn test_bitonic_u32() {
        let runner = AshRunner::new().unwrap();
        let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
        assert_eq!(data, vec![0, 7, 13, 42, 128, 256, 511, 999]);
    }

    #[test]
    fn test_bitonic_i32() {
        let runner = AshRunner::new().unwrap();
        let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
        assert_eq!(data, vec![-999, -256, -42, -1, 0, 7, 13, 128]);
    }

    #[test]
    fn test_bitonic_f32() {
        let runner = AshRunner::new().unwrap();
        let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
    }

    #[test]
    fn test_bitonic_u32_descending() {
        let runner = AshRunner::new().unwrap();
        let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
        assert_eq!(data, vec![999, 511, 256, 128, 42, 13, 7, 0]);
    }

    #[test]
    fn test_bitonic_i32_descending() {
        let runner = AshRunner::new().unwrap();
        let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
        assert_eq!(data, vec![128, 13, 7, 0, -1, -42, -256, -999]);
    }

    #[test]
    fn test_bitonic_f32_descending() {
        let runner = AshRunner::new().unwrap();
        let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
    }
}
