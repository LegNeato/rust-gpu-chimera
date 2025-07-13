//! Simple demo showing the same compute kernel running on CPU, CUDA, and Vulkan

use anyhow::Result;
use rust_gpu_chimera_demo::*;
use shared::{SortOrder, SortableKey};

fn print_header() {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║     🧬 Rust GPU Chimera Demo - Bitonic Sort 🦀    ║");
    println!("╚═══════════════════════════════════════════════════╝");
}

fn print_test_header(test_name: &str) {
    println!("\n┌─────────────────────────────────────────────────┐");
    println!("│ {test_name:<47} │");
    println!("└─────────────────────────────────────────────────┘");
}

fn log_backend_info(
    host: &str,
    backend: Option<&str>,
    adapter: Option<&str>,
    driver: Option<&str>,
) {
    println!("  Host: {host}");

    if let Some(b) = backend {
        println!("  Backend: {b}");
    }

    if let Some(a) = adapter {
        println!("  Adapter: {a}");
    }

    if let Some(d) = driver {
        if !d.is_empty() {
            println!("  Driver: {d}");
        }
    }
}

fn run_sort_test<T, R>(runner: &R, data: &mut [T], test_type: &str, order: SortOrder) -> Result<()>
where
    T: SortableKey + bytemuck::Pod + Send + Sync + std::fmt::Debug + PartialOrd + Clone,
    R: SortRunner,
{
    // Get and log backend info
    let (host, backend, adapter, driver) = runner.backend_info();
    log_backend_info(host, backend, adapter.as_deref(), driver.as_deref());

    let len = data.len();
    let original_first_10 = data[..10.min(len)].to_vec();
    let original_last_10 = if len > 10 {
        data[len - 10..].to_vec()
    } else {
        vec![]
    };

    runner.sort(data, order)?;

    // Verify sort
    let is_sorted = match order {
        SortOrder::Ascending => data.windows(2).all(|w| w[0] <= w[1]),
        SortOrder::Descending => data.windows(2).all(|w| w[0] >= w[1]),
    };

    // Display results
    println!("\n  Original (first 10): {original_first_10:?}");
    if !original_last_10.is_empty() {
        println!("  Original (last 10):  {original_last_10:?}");
    }
    println!("  Sorted (first 10):   {:?}", &data[..10.min(len)]);
    if len > 10 {
        println!("  Sorted (last 10):    {:?}", &data[len - 10..]);
    }

    if is_sorted {
        println!("\n  ✅ {test_type} sort ({order}): PASSED ({len} elements)");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "{} sort ({}) failed: array not properly sorted",
            test_type,
            order
        ))
    }
}

fn run_test_on_backend<T>(data: &mut [T], test_type: &str, order: SortOrder) -> Result<()>
where
    T: SortableKey + bytemuck::Pod + Send + Sync + std::fmt::Debug + PartialOrd + Clone,
{
    #[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "ash")))]
    {
        let runner = CpuRunner;
        run_sort_test(&runner, data, test_type, order)?;
    }

    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "ash"))]
    {
        let mut gpu_executed = false;

        #[cfg(feature = "cuda")]
        if !gpu_executed {
            if let Ok(runner) = CudaRunner::new() {
                run_sort_test(&runner, data, test_type, order)?;
                gpu_executed = true;
            } else if let Err(e) = CudaRunner::new() {
                eprintln!("  CUDA initialization failed: {e}");
            }
        }

        #[cfg(feature = "wgpu")]
        if !gpu_executed {
            if let Ok(runner) = futures::executor::block_on(WgpuRunner::new()) {
                run_sort_test(&runner, data, test_type, order)?;
                gpu_executed = true;
            } else if let Err(e) = futures::executor::block_on(WgpuRunner::new()) {
                eprintln!("  wgpu initialization failed: {e}");
            }
        }

        #[cfg(feature = "ash")]
        if !gpu_executed {
            if let Ok(runner) = AshRunner::new() {
                run_sort_test(&runner, data, test_type, order)?;
                gpu_executed = true;
            } else if let Err(e) = AshRunner::new() {
                eprintln!("  Vulkan initialization failed: {e}");
            }
        }

        if !gpu_executed {
            eprintln!("\n  ❌ No GPU backend available");
            return Err(anyhow::anyhow!("No GPU backend available"));
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    print_header();

    // Test 1: u32 with 1000 elements
    print_test_header("Test 1: Sorting 1000 u32 elements");
    let mut u32_data = vec![0u32; 1000];
    for (i, v) in u32_data.iter_mut().enumerate() {
        *v = ((i * 31337 + 42) % 1000) as u32;
    }
    run_test_on_backend(&mut u32_data, "u32", SortOrder::Ascending)?;

    // Test 2: u32 with special values
    print_test_header("Test 2: Sorting u32 with special values");
    let mut u32_special = vec![
        42u32,
        7,
        999,
        0,
        13,
        256,
        128,
        1,
        u32::MAX,
        u32::MIN,
        u32::MAX / 2,
        u32::MAX - 1,
        1000000,
        999999,
        100,
        50,
    ];
    run_test_on_backend(&mut u32_special, "u32 special", SortOrder::Ascending)?;

    // Test 3: i32 with 1000 elements
    print_test_header("Test 3: Sorting 1000 i32 elements");
    let mut i32_data = vec![0i32; 1000];
    for (i, v) in i32_data.iter_mut().enumerate() {
        *v = ((i as i32 * 31337 - 500000) % 2000) - 1000;
    }
    run_test_on_backend(&mut i32_data, "i32", SortOrder::Ascending)?;

    // Test 4: i32 with special values
    print_test_header("Test 4: Sorting i32 with special values");
    let mut i32_special = vec![
        -42i32,
        7,
        -999,
        0,
        13,
        -256,
        128,
        -1,
        i32::MAX,
        i32::MIN,
        i32::MAX / 2,
        i32::MIN / 2,
        -1000000,
        999999,
        -100,
        50,
    ];
    run_test_on_backend(&mut i32_special, "i32 special", SortOrder::Ascending)?;

    // Test 5: f32 with 1000 elements
    print_test_header("Test 5: Sorting 1000 f32 elements");
    let mut f32_data = vec![0.0f32; 1000];
    for (i, v) in f32_data.iter_mut().enumerate() {
        *v = ((i as f32 * std::f32::consts::PI) - 500.0) * 0.123;
    }
    run_test_on_backend(&mut f32_data, "f32", SortOrder::Ascending)?;

    // Test 6: f32 with special values
    print_test_header("Test 6: Sorting f32 with special values");
    let mut f32_special = vec![
        std::f32::consts::PI,
        -2.71,
        0.0,
        -0.0,
        1.41,
        -99.9,
        42.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-10,
        -1e10,
        0.1,
    ];
    run_test_on_backend(&mut f32_special, "f32 special", SortOrder::Ascending)?;

    // Test 7: u32 descending
    print_test_header("Test 7: Sorting u32 descending");
    let u32_desc = vec![42u32, 7, 999, 0, 13, 256, 128, 511, 1, 64];
    run_test_on_backend(&mut u32_desc.clone(), "u32", SortOrder::Descending)?;

    // Test 8: i32 descending with negatives
    print_test_header("Test 8: Sorting i32 descending with negatives");
    let i32_desc = vec![-42i32, 7, -999, 0, 13, -256, 128, -1, 100, -100];
    run_test_on_backend(&mut i32_desc.clone(), "i32", SortOrder::Descending)?;

    // Test 9: f32 descending with special values
    print_test_header("Test 9: Sorting f32 descending with special values");
    let f32_desc = vec![
        std::f32::consts::PI,
        -2.71,
        0.0,
        -0.0,
        1.41,
        -99.9,
        42.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
    ];
    run_test_on_backend(&mut f32_desc.clone(), "f32", SortOrder::Descending)?;

    println!("\n═══════════════════════════════════════════════════");
    println!("All tests completed successfully! 🎉");
    println!("═══════════════════════════════════════════════════");

    Ok(())
}
