# Rust GPU Chimera Demo

A demonstration of the same Rust codebase running on multiple GPU backends (CUDA, Vulkan) and CPU, showcasing advanced Rust features in GPU compute shaders.

## Features

This project demonstrates:

- **Unified codebase**: Same Rust code compiles to CUDA PTX, SPIR-V (Vulkan), and native CPU
- **Advanced Rust features in shaders**:
  - Traits and trait bounds
  - Newtypes for type safety
  - Enums with pattern matching
  - Iterators and closures
  - Generic functions
  - `no_std` compatibility
- **Multiple backends**:
  - CPU execution with multithreading
  - CUDA via `rust-cuda`
  - Vulkan via `rust-gpu` with both `wgpu` and `ash` hosts
- **Shared dependencies**: Uses `glam` math library across all platforms

## Project Structure

```
├── shared/           # Shared types and physics logic (no_std)
├── kernel/           # GPU/CPU compute kernels
├── src/
│   ├── cpu_runner.rs    # CPU execution harness
│   ├── cuda_runner.rs   # CUDA host implementation
│   ├── wgpu_runner.rs   # Vulkan host using wgpu
│   ├── ash_runner.rs    # Vulkan host using ash
│   └── main.rs          # Demo application
└── build.rs          # Kernel compilation script
```

## Building and Running

### Supported Configurations

| Platform     | Features      | Host   | Backend | Driver        | How it Works         | Status          |
| ------------ | ------------- | ------ | ------- | ------------- | -------------------- | --------------- |
| **Linux**    | -             | CPU    | -       | -             | Rust → Native        | ✅ Working      |
| Linux        | `wgpu`        | [wgpu] | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working      |
| Linux        | `ash`         | [ash]  | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working      |
| Linux        | `cuda`        | [cust] | CUDA    | Native        | Rust → NVVM → PTX    | ✅ Working      |
| **macOS**    | -             | CPU    | -       | -             | Rust → Native        | ✅ Working      |
| macOS        | `wgpu`        | [wgpu] | Metal   | Metal         | Rust → SPIR-V → MSL  | ✅ Working      |
| macOS        | `wgpu,vulkan` | [wgpu] | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | ✅ Working      |
| macOS        | `wgpu,vulkan` | [wgpu] | Vulkan  | [SwiftShader] | Rust → SPIR-V        | ✅ Working      |
| macOS        | `ash`         | [ash]  | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | ✅ Working      |
| macOS        | `ash`         | [ash]  | Vulkan  | [SwiftShader] | Rust → SPIR-V        | ✅ Working      |
| macOS        | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^1] |
| **Windows**  | -             | CPU    | -       | -             | Rust → Native        | ✅ Working      |
| Windows      | `wgpu`        | [wgpu] | DX12    | Native        | Rust → SPIR-V → HLSL | ✅ Working      |
| Windows      | `wgpu,vulkan` | [wgpu] | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working      |
| Windows      | `wgpu,vulkan` | [wgpu] | Vulkan  | [SwiftShader] | Rust → SPIR-V        | ✅ Working      |
| Windows      | `ash`         | [ash]  | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working      |
| Windows      | `ash`         | [ash]  | Vulkan  | [SwiftShader] | Rust → SPIR-V        | ✅ Working      |
| Windows      | `cuda`        | [cust] | CUDA    | Native        | Rust → NVVM → PTX    | ✅ Working      |
| **Android**  | -             | CPU    | -       | -             | Rust → Native        | ✅ Working      |
| Android      | `wgpu`        | [wgpu] | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working      |
| Android      | `ash`         | [ash]  | Vulkan  | Native        | Rust → SPIR-V        | ✅ Working      |
| Android      | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^2] |
| **iOS**      | -             | CPU    | -       | -             | Rust → Native        | ✅ Working      |
| iOS          | `wgpu`        | [wgpu] | Metal   | Metal         | Rust → SPIR-V → MSL  | 🔷 Should work  |
| iOS          | `wgpu,vulkan` | [wgpu] | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work  |
| iOS          | `ash`         | [ash]  | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work  |
| iOS          | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^1] |
| **tvOS**     | -             | CPU    | -       | -             | Rust → Native        | ✅ Working      |
| tvOS         | `wgpu`        | [wgpu] | Metal   | Metal         | Rust → SPIR-V → MSL  | 🔷 Should work  |
| tvOS         | `wgpu,vulkan` | [wgpu] | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work  |
| tvOS         | `ash`         | [ash]  | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work  |
| tvOS         | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^1] |
| **visionOS** | -             | CPU    | -       | -             | Rust → Native        | ✅ Working      |
| visionOS     | `wgpu`        | [wgpu] | Metal   | Metal         | Rust → SPIR-V → MSL  | 🔷 Should work  |
| visionOS     | `wgpu,vulkan` | [wgpu] | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work  |
| visionOS     | `ash`         | [ash]  | Vulkan  | [MoltenVK]    | Rust → SPIR-V        | 🔷 Should work  |
| visionOS     | `cuda`        | [cust] | CUDA    | -             | -                    | ❌ Unavailable[^1] |

### Key Technical Details

- **SPIR-V Generation**: [rust-gpu](https://github.com/rust-gpu/rust-gpu) compiles Rust to SPIR-V for Vulkan
- **Metal Support**: [wgpu](https://github.com/gfx-rs/wgpu)'s [naga](https://github.com/gfx-rs/naga) library translates SPIR-V to MSL at runtime
- **CUDA Support**: [rust-cuda](https://github.com/Rust-GPU/Rust-CUDA) compiles Rust to NVVM → PTX for NVIDIA GPUs
- **Platform Guards**: Compile-time checks prevent unsupported configurations

[^1]: CUDA is not supported on macOS/iOS/tvOS/visionOS (NVIDIA GPUs not available).  
  [ZLUDA](https://github.com/vosen/ZLUDA) could potentially enable CUDA on these platforms in the future.

[^2]: CUDA is not supported on Android.  
  [ZLUDA](https://github.com/vosen/ZLUDA) could potentially enable CUDA on Android in the future.

### Related Projects

- [wgpu](https://github.com/gfx-rs/wgpu) - Rust graphics API that runs on Metal, Vulkan, DX12, and more
- [ash](https://github.com/ash-rs/ash) - Low-level Vulkan bindings for Rust
- [rust-gpu](https://github.com/rust-gpu/rust-gpu) - Rust to SPIR-V compiler
- [rust-cuda](https://github.com/Rust-GPU/Rust-CUDA) - Rust to CUDA compiler
- [MoltenVK](https://github.com/KhronosGroup/MoltenVK) - Vulkan-on-Metal translation layer
- [SwiftShader](https://github.com/google/swiftshader) - CPU-based Vulkan implementation
- [ZLUDA](https://github.com/vosen/ZLUDA) - CUDA-on-non-NVIDIA-GPUs implementation

[wgpu]: https://github.com/gfx-rs/wgpu
[ash]: https://github.com/ash-rs/ash
[cust]: https://github.com/Rust-GPU/Rust-CUDA/tree/main/crates/cust
[MoltenVK]: https://github.com/KhronosGroup/MoltenVK
[SwiftShader]: https://github.com/google/swiftshader

## Running Examples

### Linux

```bash
# CPU execution
cargo run --release

# Vulkan via wgpu (SPIR-V passthrough)
cargo run --release --features wgpu

# Vulkan via ash (direct API)
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

# Vulkan via wgpu (requires SwiftShader)
cargo run --release --features wgpu,vulkan
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

### Android

```bash
# Requires Android NDK and cargo-apk or similar tooling
# CPU execution
cargo apk run --release

# Vulkan via wgpu
cargo apk run --release --features wgpu

# Vulkan via ash
cargo apk run --release --features ash
```

### iOS/tvOS/visionOS

```bash
# Requires Xcode and cargo-mobile or similar tooling
# CPU execution
cargo mobile run --release

# Metal via wgpu (SPIR-V → MSL translation)
cargo mobile run --release --features wgpu
```

### Docker Testing (for macOS users)

```bash
# Test Linux backends using Mesa software renderer
cd container
docker build -f Dockerfile.linux-arm64-mesa -t rust-gpu-demo .
docker run -it --rm -v $(pwd)/..:/workspace rust-gpu-demo bash

# Inside container:
cd /workspace
cargo run --release --features ash
```

## Implementation Status

### Core Infrastructure

- ✅ Build system for rust-gpu (SPIR-V) and rust-cuda
- ✅ Feature-based backend selection
- ✅ Platform-specific compile-time validation
- ✅ Proper error handling with thiserror/anyhow

### Compute Backends

- ✅ **CPU**: Native Rust execution
- ✅ **wgpu**: High-level GPU API (Metal on macOS, Vulkan on Linux)
- ✅ **ash**: Low-level Vulkan API (Linux only)
- ✅ **CUDA**: NVIDIA GPU support (Linux only, builds but not runtime tested)

### Kernel Implementation

- ✅ Simple compute kernel (index \* 2 + 42)
- ✅ Shared kernel code between CPU/GPU
- ✅ Separate SPIR-V and CUDA kernel entry points
- ❌ Advanced Rust features in shaders (traits, iterators, etc.) - TODO

## Project Structure

The project demonstrates:

- Same Rust code running on CPU, CUDA, and Vulkan
- Conditional compilation for platform-specific code
- Shared memory layout using `#[repr(C)]` structs
- Build-time kernel compilation (SPIR-V and PTX)
  }

  ```

  ```

3. **Enums** with pattern matching in shaders:

   ```rust
   match params.force_type {
       ForceType::Electrostatic => { /* ... */ }
       ForceType::Gravity => { /* ... */ }
       ForceType::Both => { /* ... */ }
   }
   ```

4. **Iterators** in GPU code:

   ```rust
   particles.iter()
       .enumerate()
       .filter(|(i, _)| *i != particle_idx)
       .map(|(_, other)| calculator.calculate_force(current, other))
       .for_each(|force| accumulator.add(force));
   ```

5. **Generic functions** with trait bounds:
   ```rust
   fn calculate_all_forces<F: ForceCalculator>(
       particle_idx: u32,
       particles: &[ParticleData],
       calculator: &F,
       accumulator: &mut ForceAccumulator,
       params: &SimulationParams,
   )
   ```
