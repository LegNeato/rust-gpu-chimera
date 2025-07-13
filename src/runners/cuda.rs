//! CUDA runner implementation

use crate::{error::Result, SortRunner};
use cust::prelude::*;
use shared::{BitonicParams, BLOCK_SIZE};
use std::sync::OnceLock;

// Global CUDA context to ensure single initialization
static CUDA_CONTEXT: OnceLock<Context> = OnceLock::new();

/// CUDA-based runner for bitonic sort using NVIDIA GPUs
pub struct CudaRunner {
    module: Module,
    stream: Stream,
    device_name: String,
}

impl CudaRunner {
    /// Create a new CUDA runner, initializing the CUDA context if needed
    pub fn new() -> Result<Self> {
        // Initialize CUDA context only once
        let _ctx = CUDA_CONTEXT.get_or_try_init(cust::quick_init)?;

        // Get device info
        let device = Device::get_device(0)?;
        let device_name = device.name()?;

        // Use the embedded PTX from the main crate
        let ptx_data = crate::BITONIC_PTX;
        let module = Module::from_ptx(ptx_data, &[])?;

        // Create stream
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self {
            module,
            stream,
            device_name,
        })
    }
}

impl SortRunner for CudaRunner {
    fn backend_info(
        &self,
    ) -> (
        &'static str,
        Option<&'static str>,
        Option<String>,
        Option<String>,
    ) {
        ("cust", Some("CUDA"), Some(self.device_name.clone()), None)
    }

    fn execute_kernel_pass(&self, data: &mut [u32], params: BitonicParams) -> Result<()> {
        // Get kernel function
        let kernel = self.module.get_function("bitonic_kernel")?;

        // Set up launch configuration
        let block_size = BLOCK_SIZE;
        let grid_size = params.num_elements.div_ceil(block_size);

        // Allocate device memory
        let device_data = DeviceBuffer::from_slice(data)?;

        // Launch kernel
        let stream = &self.stream;
        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    device_data.as_device_ptr(),
                    params
                )
            )?;
        }

        // Wait for completion
        self.stream.synchronize()?;

        // Copy back to host
        device_data.copy_to(data)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CudaRunner;
    use crate::{verify_sorted, SortRunner};
    use shared::SortOrder;

    #[test]
    fn test_bitonic_u32() {
        if let Ok(runner) = CudaRunner::new() {
            let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

            runner.sort(&mut data, SortOrder::Ascending).unwrap();
            assert!(verify_sorted(&data, SortOrder::Ascending));
            assert_eq!(data, vec![0, 7, 13, 42, 128, 256, 511, 999]);
        }
    }

    #[test]
    fn test_bitonic_i32() {
        if let Ok(runner) = CudaRunner::new() {
            let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

            runner.sort(&mut data, SortOrder::Ascending).unwrap();
            assert!(verify_sorted(&data, SortOrder::Ascending));
            assert_eq!(data, vec![-999, -256, -42, -1, 0, 7, 13, 128]);
        }
    }

    #[test]
    fn test_bitonic_f32() {
        if let Ok(runner) = CudaRunner::new() {
            let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

            runner.sort(&mut data, SortOrder::Ascending).unwrap();
            assert!(verify_sorted(&data, SortOrder::Ascending));
        }
    }

    #[test]
    fn test_bitonic_u32_descending() {
        if let Ok(runner) = CudaRunner::new() {
            let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

            runner.sort(&mut data, SortOrder::Descending).unwrap();
            assert!(verify_sorted(&data, SortOrder::Descending));
            assert_eq!(data, vec![999, 511, 256, 128, 42, 13, 7, 0]);
        }
    }

    #[test]
    fn test_bitonic_i32_descending() {
        if let Ok(runner) = CudaRunner::new() {
            let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

            runner.sort(&mut data, SortOrder::Descending).unwrap();
            assert!(verify_sorted(&data, SortOrder::Descending));
            assert_eq!(data, vec![128, 13, 7, 0, -1, -42, -256, -999]);
        }
    }

    #[test]
    fn test_bitonic_f32_descending() {
        if let Ok(runner) = CudaRunner::new() {
            let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

            runner.sort(&mut data, SortOrder::Descending).unwrap();
            assert!(verify_sorted(&data, SortOrder::Descending));
        }
    }
}
