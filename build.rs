//! Build script for compiling kernels to SPIR-V and CUDA PTX

fn main() {
    // Only build kernels when the appropriate features are enabled
    #[cfg(any(feature = "vulkan", feature = "wgpu"))]
    build_spirv_kernel();

    #[cfg(feature = "cuda")]
    {
        #[cfg(target_os = "macos")]
        panic!("CUDA is not supported on macOS. CUDA requires NVIDIA GPUs and is only available on Linux and Windows");

        #[cfg(not(target_os = "macos"))]
        build_cuda_kernel();
    }
}

#[cfg(any(feature = "vulkan", feature = "wgpu"))]
fn build_spirv_kernel() {
    use spirv_builder::SpirvBuilder;
    use std::path::PathBuf;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_path = PathBuf::from(manifest_dir).join("kernel");

    let result = SpirvBuilder::new(crate_path, "spirv-unknown-vulkan1.2")
        .print_metadata(spirv_builder::MetadataPrintout::Full)
        .build()
        .unwrap();

    // Export the kernel path for the runtime to use
    println!(
        "cargo:rustc-env=BITONIC_KERNEL_SPV_PATH={}",
        result.module.unwrap_single().display()
    );

    // Use the first entry point
    println!(
        "cargo:rustc-env=BITONIC_KERNEL_SPV_ENTRY={}",
        result.entry_points.first().unwrap()
    );
}

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
fn build_cuda_kernel() {
    use cuda_builder::CudaBuilder;
    use std::path::PathBuf;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(&out_dir);

    let ptx_path = out_path.join("kernel.ptx");

    println!("cargo:rerun-if-changed=shared/src/lib.rs");
    println!("cargo:rerun-if-changed=build.rs");

    CudaBuilder::new("kernel")
        .copy_to(&ptx_path)
        .build()
        .expect("Failed to build CUDA kernel");

    // Export the PTX path as an environment variable for embedding
    println!(
        "cargo:rustc-env=BITONIC_KERNEL_PTX_PATH={}",
        ptx_path.display()
    );
}
