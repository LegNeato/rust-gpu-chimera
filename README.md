# Rust GPU Chimera Demo

A demonstration of unified GPU compute across multiple platforms, showcasing how the
same Rust code can run on CUDA, Vulkan (via SPIR-V), Metal, DirectX, and CPU.

### Supported Configurations

| Platform     | Features      | Host   | Backend | Driver        | How it Works         | Status             |
| ------------ | ------------- | ------ | ------- | ------------- | -------------------- | ------------------ |
| **Linux**    | -             | CPU    | -       | -             | Rust → Native        | ✅ Working         |
| Linux        | `wgpu`        | [wgpu] | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working         |
| Linux        | `ash`         | [ash]  | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working         |
| Linux        | `cuda`        | [cust] | CUDA    | Native        | Rust → NVVM → PTX    | ✅ Working         |
| **macOS**    | -             | CPU    | -       | -             | Rust → Native        | ✅ Working         |
| macOS        | `wgpu`        | [wgpu] | Metal   | Metal         | Rust → SPIR-V → MSL  | ✅ Working         |
| macOS        | `wgpu,vulkan` | [wgpu] | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | ✅ Working         |
| macOS        | `wgpu,vulkan` | [wgpu] | Vulkan  | [SwiftShader] | Rust → SPIR-V        | ✅ Working         |
| macOS        | `ash`         | [ash]  | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | ✅ Working         |
| macOS        | `ash`         | [ash]  | Vulkan  | [SwiftShader] | Rust → SPIR-V        | ✅ Working         |
| macOS        | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^1] |
| **Windows**  | -             | CPU    | -       | -             | Rust → Native        | ✅ Working         |
| Windows      | `wgpu`        | [wgpu] | DX12    | Native        | Rust → SPIR-V → HLSL | ✅ Working         |
| Windows      | `wgpu,vulkan` | [wgpu] | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working         |
| Windows      | `wgpu,vulkan` | [wgpu] | Vulkan  | [SwiftShader] | Rust → SPIR-V        | ✅ Working         |
| Windows      | `ash`         | [ash]  | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working         |
| Windows      | `ash`         | [ash]  | Vulkan  | [SwiftShader] | Rust → SPIR-V        | ✅ Working         |
| Windows      | `cuda`        | [cust] | CUDA    | Native        | Rust → NVVM → PTX    | ✅ Working         |
| **Android**  | -             | CPU    | -       | -             | Rust → Native        | ✅ Working         |
| Android      | `wgpu`        | [wgpu] | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working         |
| Android      | `ash`         | [ash]  | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working         |
| Android      | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^2] |
| **iOS**      | -             | CPU    | -       | -             | Rust → Native        | ✅ Working         |
| iOS          | `wgpu`        | [wgpu] | Metal   | Metal         | Rust → SPIR-V → MSL  | 🔷 Should work     |
| iOS          | `wgpu,vulkan` | [wgpu] | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work     |
| iOS          | `ash`         | [ash]  | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work     |
| iOS          | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^1] |
| **tvOS**     | -             | CPU    | -       | -             | Rust → Native        | ✅ Working         |
| tvOS         | `wgpu`        | [wgpu] | Metal   | Metal         | Rust → SPIR-V → MSL  | 🔷 Should work     |
| tvOS         | `wgpu,vulkan` | [wgpu] | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work     |
| tvOS         | `ash`         | [ash]  | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work     |
| tvOS         | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^1] |
| **visionOS** | -             | CPU    | -       | -             | Rust → Native        | ✅ Working         |
| visionOS     | `wgpu`        | [wgpu] | Metal   | Metal         | Rust → SPIR-V → MSL  | 🔷 Should work     |
| visionOS     | `wgpu,vulkan` | [wgpu] | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work     |
| visionOS     | `ash`         | [ash]  | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work     |
| visionOS     | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^1] |

[^1]:
    CUDA is not supported on macOS/iOS/tvOS/visionOS (NVIDIA GPUs not available).  
    [ZLUDA](https://github.com/vosen/ZLUDA) could potentially enable CUDA on these
    platforms in the future.

[^2]:
    CUDA is not supported on Android.  
    [ZLUDA](https://github.com/vosen/ZLUDA) could potentially enable CUDA on Android in
    the future.

## Running the Demo

The demo runs a bitonic sort on various data types (u32, i32, f32) with different sizes
and configurations.

### Linux

```bash
# CPU execution
cargo run --release

# Vulkan via wgpu
cargo run --release --features wgpu

# Vulkan via ash
cargo run --release --features ash

# CUDA (NVIDIA GPU required)
cargo run --release --features cuda
```

### macOS

```bash
# CPU execution
cargo run --release

# Metal via wgpu (SPIR-V → MSL translation)
cargo run --release --features wgpu

# Vulkan via wgpu (requires MoltenVK)
cargo run --release --features wgpu,vulkan

# Vulkan via ash (requires MoltenVK)
cargo run --release --features ash
```

### Windows

```bash
# CPU execution
cargo run --release

# DirectX 12 via wgpu (SPIR-V → HLSL translation)
cargo run --release --features wgpu

# Vulkan via wgpu
cargo run --release --features wgpu,vulkan

# Vulkan via ash
cargo run --release --features ash

# CUDA (NVIDIA GPU required)
cargo run --release --features cuda
```

Instead of `cargo run` you can replace it with `cargo test` to run unit tests for the
same configuration.

## Project Structure

```
rust-gpu-chimera-demo/
├── kernel/           # Compute kernel logic and entrypoints
│   └── src/
│       └── lib.rs
├── shared/           # Code that runs on both the CPU and GPU
│   └── src/
│       └── lib.rs
├── src/
│   ├── runners/      # Code that runs on the CPU/host and interfaces with the GPU
│   │   ├── cpu.rs
│   │   ├── cuda.rs
│   │   ├── wgpu.rs
│   │   └── ash.rs
│   ├── lib.rs
│   └── main.rs       # Demo application binary
└── build.rs          # Kernel compilation orchestration
```

[wgpu]: https://github.com/gfx-rs/wgpu
[ash]: https://github.com/ash-rs/ash
[cust]: https://github.com/Rust-GPU/Rust-CUDA/tree/main/crates/cust
[MoltenVK]: https://github.com/KhronosGroup/MoltenVK
[SwiftShader]: https://github.com/google/swiftshader
