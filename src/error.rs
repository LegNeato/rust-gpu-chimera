//! Error types for the library

use thiserror::Error;

/// Error types for the Rust GPU Chimera demo
#[derive(Error, Debug)]
pub enum ChimeraError {
    #[error("No suitable GPU adapter found")]
    NoAdapter,

    #[error("No suitable Vulkan device found among {0} devices")]
    NoVulkanDevice(usize),

    #[error("Buffer size overflow: {0} elements × {1} bytes per element")]
    BufferSizeOverflow(usize, usize),

    #[error("Mapped memory size ({mapped}) is smaller than expected ({expected})")]
    InsufficientMappedMemory { mapped: u64, expected: u64 },

    #[error("Failed to find kernel module: {0}")]
    KernelNotFound(String),

    #[error("Failed to find compute queue family")]
    NoComputeQueue,

    #[cfg(feature = "wgpu")]
    #[error("wgpu error: {0}")]
    Wgpu(#[from] wgpu::Error),

    #[cfg(feature = "wgpu")]
    #[error("wgpu request device error: {0}")]
    WgpuRequestDevice(#[from] wgpu::RequestDeviceError),

    #[cfg(feature = "ash")]
    #[error("Vulkan error: {0}")]
    Vulkan(String),

    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(#[from] cust::error::CudaError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<Box<dyn std::error::Error>> for ChimeraError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        ChimeraError::Other(err.to_string())
    }
}

#[cfg(feature = "ash")]
impl From<ash::vk::Result> for ChimeraError {
    fn from(err: ash::vk::Result) -> Self {
        ChimeraError::Vulkan(format!("Vulkan error: {err:?}"))
    }
}

/// Convenience type alias for Results with [`ChimeraError`]
pub type Result<T> = std::result::Result<T, ChimeraError>;
