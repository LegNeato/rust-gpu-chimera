//! CPU execution for compute kernels

use crate::{error::Result, SortRunner};
use kernel::bitonic_sort_step;
use shared::{BitonicParams, ThreadId};

/// CPU-based runner for bitonic sort using native Rust code
pub struct CpuRunner;

impl SortRunner for CpuRunner {
    fn backend_info(
        &self,
    ) -> (
        &'static str,
        Option<&'static str>,
        Option<String>,
        Option<String>,
    ) {
        ("cpu", Some("Native"), None, None)
    }

    fn execute_kernel_pass(&self, data: &mut [u32], params: BitonicParams) -> Result<()> {
        // Process all threads (on CPU, we simulate parallel execution)
        for thread_idx in 0..params.num_elements {
            let thread_id = ThreadId::new(thread_idx);
            bitonic_sort_step(
                thread_id,
                data,
                params.stage,
                params.pass_of_stage,
                params.num_elements,
                params.sort_order,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CpuRunner;
    use crate::{verify_sorted, SortRunner};
    use shared::SortOrder;

    #[test]
    fn test_bitonic_u32() {
        let runner = CpuRunner;
        let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
        assert_eq!(data, vec![0, 7, 13, 42, 128, 256, 511, 999]);
    }

    #[test]
    fn test_bitonic_i32() {
        let runner = CpuRunner;
        let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
        assert_eq!(data, vec![-999, -256, -42, -1, 0, 7, 13, 128]);
    }

    #[test]
    fn test_bitonic_f32() {
        let runner = CpuRunner;
        let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

        runner.sort(&mut data, SortOrder::Ascending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Ascending));
    }

    #[test]
    fn test_bitonic_u32_descending() {
        let runner = CpuRunner;
        let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
        assert_eq!(data, vec![999, 511, 256, 128, 42, 13, 7, 0]);
    }

    #[test]
    fn test_bitonic_i32_descending() {
        let runner = CpuRunner;
        let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
        assert_eq!(data, vec![128, 13, 7, 0, -1, -42, -256, -999]);
    }

    #[test]
    fn test_bitonic_f32_descending() {
        let runner = CpuRunner;
        let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

        runner.sort(&mut data, SortOrder::Descending).unwrap();
        assert!(verify_sorted(&data, SortOrder::Descending));
    }
}
