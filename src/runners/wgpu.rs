//! wgpu runner implementation

use crate::{
    error::{ChimeraError, Result},
    SortRunner,
};
use shared::{BitonicParams, WORKGROUP_SIZE};
use wgpu::util::DeviceExt;

/// WebGPU-based runner for bitonic sort supporting multiple backends (Vulkan, Metal, DX12, etc.)
pub struct WgpuRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bitonic_pipeline: Option<wgpu::ComputePipeline>,
    bitonic_bind_group_layout: Option<wgpu::BindGroupLayout>,
    backend_name: String,
    adapter_name: String,
    driver_info: String,
}

impl WgpuRunner {
    /// Create a new wgpu runner, automatically detecting the best available backend
    pub async fn new() -> Result<Self> {
        // Initialize wgpu with appropriate backend
        #[cfg(all(target_os = "macos", feature = "vulkan"))]
        let backends = wgpu::Backends::VULKAN;

        #[cfg(all(target_os = "macos", not(feature = "vulkan")))]
        let backends = wgpu::Backends::METAL;

        #[cfg(not(target_os = "macos"))]
        let backends = wgpu::Backends::all();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|_| ChimeraError::NoAdapter)?;

        // Get adapter info to show which backend we're using
        let info = adapter.get_info();
        let backend_name = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DirectX 12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Unknown",
        }
        .to_string();
        let adapter_name = info.name.clone();
        let driver_info = info.driver.clone();

        // Check if the backend supports SPIRV passthrough
        let adapter_features = adapter.features();
        let required_features =
            if adapter_features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH) {
                wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SPIRV_SHADER_PASSTHROUGH
            } else {
                // wgpu will translate SPIRV to the native format (MSL for Metal, HLSL for DX12, etc.)
                wgpu::Features::PUSH_CONSTANTS
            };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GPU Device"),
                required_features,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 128,
                    ..Default::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::default(),
            })
            .await?;

        // Try to create Bitonic pipeline
        let (bitonic_pipeline, bitonic_bind_group_layout) = Self::create_bitonic_pipeline(&device);

        Ok(Self {
            device,
            queue,
            bitonic_pipeline,
            bitonic_bind_group_layout,
            backend_name,
            adapter_name,
            driver_info,
        })
    }

    fn create_bitonic_pipeline(
        device: &wgpu::Device,
    ) -> (Option<wgpu::ComputePipeline>, Option<wgpu::BindGroupLayout>) {
        // Use the embedded kernel from the main crate
        let kernel_bytes = crate::BITONIC_SPIRV;

        let spirv_data = wgpu::util::make_spirv(kernel_bytes);
        let shader_module = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Bitonic Kernel"),
                    source: spirv_data,
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };

        // Create bind group layout for Bitonic (1 buffer: data)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bitonic Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bitonic Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<BitonicParams>() as u32,
            }],
        });

        // Get the entry point name from the build script
        let entry_point = option_env!("BITONIC_KERNEL_SPV_ENTRY").unwrap_or("bitonic_kernel");

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bitonic Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        (Some(pipeline), Some(bind_group_layout))
    }

    async fn execute_kernel_pass_async(
        &self,
        data: &mut [u32],
        params: BitonicParams,
        pipeline: &wgpu::ComputePipeline,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<()> {
        let size = std::mem::size_of_val(data) as u64;

        // Calculate workgroups to cover all elements
        let workgroup_size = WORKGROUP_SIZE;
        let num_workgroups = params.num_elements.div_ceil(workgroup_size);

        // Create GPU buffer for in-place sorting
        let data_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bitonic Data Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bitonic Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bitonic Bind Group"),
            layout: bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: data_buffer.as_entire_binding(),
            }],
        });

        // Encode commands for this single pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!(
                    "Bitonic Encoder Stage {} Pass {}",
                    params.stage.as_u32(),
                    params.pass_of_stage.as_u32()
                )),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!(
                    "Bitonic Pass Stage {} Pass {}",
                    params.stage.as_u32(),
                    params.pass_of_stage.as_u32()
                )),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&params));

            // Dispatch with WORKGROUP_SIZE threads per workgroup (matching kernel)
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Submit this pass and wait for completion
        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait);

        // Final encoder for copying results
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Bitonic Copy Encoder"),
            });

        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(&data_buffer, 0, &staging_buffer, 0, size);

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::Wait);
        receiver
            .await
            .map_err(|e| ChimeraError::Other(format!("Channel error: {e:?}")))?
            .map_err(|e| ChimeraError::Other(format!("Buffer async error: {e:?}")))?;

        {
            let view = buffer_slice.get_mapped_range();
            data.copy_from_slice(bytemuck::cast_slice(&view));
        }

        staging_buffer.unmap();

        Ok(())
    }
}

impl SortRunner for WgpuRunner {
    fn backend_info(
        &self,
    ) -> (
        &'static str,
        Option<&'static str>,
        Option<String>,
        Option<String>,
    ) {
        (
            "wgpu",
            Some(self.backend_name.as_str()).map(|s| match s {
                "Vulkan" => "Vulkan",
                "Metal" => "Metal",
                "DirectX 12" => "DirectX 12",
                "OpenGL" => "OpenGL",
                "WebGPU" => "WebGPU",
                _ => "Unknown",
            }),
            Some(self.adapter_name.clone()),
            Some(self.driver_info.clone()),
        )
    }

    fn execute_kernel_pass(&self, data: &mut [u32], params: BitonicParams) -> Result<()> {
        if self.bitonic_pipeline.is_none() || self.bitonic_bind_group_layout.is_none() {
            return Err(ChimeraError::Other(
                "Bitonic pipeline not available".to_string(),
            ));
        }

        let pipeline = self.bitonic_pipeline.as_ref().unwrap();
        let bind_group_layout = self.bitonic_bind_group_layout.as_ref().unwrap();

        futures::executor::block_on(self.execute_kernel_pass_async(
            data,
            params,
            pipeline,
            bind_group_layout,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::WgpuRunner;
    use crate::{verify_sorted, SortRunner};
    use shared::SortOrder;

    #[test]
    fn test_bitonic_u32() {
        let runner = futures::executor::block_on(WgpuRunner::new()).unwrap();
        let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
        assert_eq!(data, vec![0, 7, 13, 42, 128, 256, 511, 999]);
    }

    #[test]
    fn test_bitonic_i32() {
        let runner = futures::executor::block_on(WgpuRunner::new()).unwrap();
        let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
        assert_eq!(data, vec![-999, -256, -42, -1, 0, 7, 13, 128]);
    }

    #[test]
    fn test_bitonic_f32() {
        let runner = futures::executor::block_on(WgpuRunner::new()).unwrap();
        let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
    }

    #[test]
    fn test_bitonic_u32_descending() {
        let runner = futures::executor::block_on(WgpuRunner::new()).unwrap();
        let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
        assert_eq!(data, vec![999, 511, 256, 128, 42, 13, 7, 0]);
    }

    #[test]
    fn test_bitonic_i32_descending() {
        let runner = futures::executor::block_on(WgpuRunner::new()).unwrap();
        let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
        assert_eq!(data, vec![128, 13, 7, 0, -1, -42, -256, -999]);
    }

    #[test]
    fn test_bitonic_f32_descending() {
        let runner = futures::executor::block_on(WgpuRunner::new()).unwrap();
        let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
    }
}
