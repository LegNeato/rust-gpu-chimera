//! Build script for compiling kernels to SPIR-V and CUDA PTX

fn main() {
    // Rebuild if kernel source changes
    println!("cargo:rerun-if-changed=kernel/src/lib.rs");
    println!("cargo:rerun-if-changed=shared/src/lib.rs");
    println!("cargo:rerun-if-changed=shared/src/bitonic.rs");
    println!("cargo:rerun-if-changed=build.rs");

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
        .multimodule(true)
        .build()
        .expect("Failed to build SPIR-V kernel");

    // Export the kernel path for the runtime to use
    match result.module {
        spirv_builder::ModuleResult::SingleModule(path) => {
            println!("cargo:rustc-env=KERNEL_SPV_PATH={}", path.display());
            // For single module, use the first entry point
            if let Some(entry) = result.entry_points.first() {
                println!("cargo:rustc-env=KERNEL_SPV_ENTRY={entry}");
            }
        }
        spirv_builder::ModuleResult::MultiModule(modules) => {
            println!("cargo:warning=Found {} kernel modules", modules.len());
            for (name, path) in &modules {
                println!("cargo:warning=  Module: {} -> {}", name, path.display());
            }

            // Export paths for kernels
            // For now, just use the first available kernel
            if let Some((name, path)) = modules.iter().next() {
                println!("cargo:rustc-env=KERNEL_SPV_PATH={}", path.display());
                println!("cargo:rustc-env=KERNEL_SPV_ENTRY={name}");
            }

            // For bitonic kernel, it's now in lib.rs
            if let Some(path) = modules.get("lib::bitonic_kernel") {
                println!("cargo:rustc-env=BITONIC_KERNEL_SPV_PATH={}", path.display());
                // The actual entry point name in the SPIR-V is what's in result.entry_points
                if let Some(entry) = result
                    .entry_points
                    .iter()
                    .find(|e| e.contains("bitonic_kernel"))
                {
                    println!("cargo:rustc-env=BITONIC_KERNEL_SPV_ENTRY={entry}");
                } else {
                    // Fallback to the module name
                    println!("cargo:rustc-env=BITONIC_KERNEL_SPV_ENTRY=bitonic_kernel");
                }
            } else if let Some(path) = modules.get("bitonic_kernel") {
                // Try without module prefix
                println!("cargo:rustc-env=BITONIC_KERNEL_SPV_PATH={}", path.display());
                if let Some(entry) = result
                    .entry_points
                    .iter()
                    .find(|e| e.contains("bitonic_kernel"))
                {
                    println!("cargo:rustc-env=BITONIC_KERNEL_SPV_ENTRY={entry}");
                } else {
                    println!("cargo:rustc-env=BITONIC_KERNEL_SPV_ENTRY=bitonic_kernel");
                }
            }
        }
    }
}

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
fn build_cuda_kernel() {
    use cuda_builder::CudaBuilder;
    use std::path::PathBuf;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(&out_dir);

    let ptx_path = out_path.join("kernel.ptx");

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
