//! Compute kernel for bitonic sort.
//!
//! This demonstrates the same Rust code running on CUDA, Vulkan (SPIR-V), Metal, HLSL,
//! and CPU.

#![cfg_attr(target_arch = "spirv", no_std)]
#![cfg_attr(target_os = "cuda", no_std)]

#[cfg(any(target_arch = "spirv", target_os = "cuda"))]
use shared::BitonicParams;
use shared::{Pass, SortOrder, Stage, ThreadId};

#[cfg(target_arch = "spirv")]
use spirv_std::{glam::UVec3, spirv};

#[cfg(target_os = "cuda")]
use cuda_std::{kernel, thread};

/// Newtype wrapper for comparison distance
#[derive(Copy, Clone, Debug)]
pub struct ComparisonDistance(u32);

impl ComparisonDistance {
    #[inline]
    fn from_stage_pass(stage: Stage, pass: Pass) -> Self {
        Self(1u32 << (stage.as_u32() - pass.as_u32()))
    }

    #[inline]
    fn find_partner(&self, thread_id: ThreadId) -> ThreadId {
        ThreadId::new(thread_id.as_u32() ^ self.0)
    }
}

/// Represents a comparison pair in the bitonic network
#[derive(Copy, Clone, Debug)]
pub struct ComparisonPair {
    lower: ThreadId,
    upper: ThreadId,
}

impl ComparisonPair {
    #[inline]
    fn try_new(thread_id: ThreadId, partner: ThreadId) -> (bool, Self) {
        let is_valid = partner.as_u32() > thread_id.as_u32();
        let pair = Self {
            lower: thread_id,
            upper: partner,
        };
        (is_valid, pair)
    }

    #[inline]
    fn is_in_bounds(&self, num_elements: u32) -> bool {
        self.upper.as_u32() < num_elements
    }
}

/// Encapsulates the bitonic sort direction logic
#[derive(Copy, Clone, Debug)]
pub struct BitonicDirection {
    block_ascending: bool,
}

impl BitonicDirection {
    #[inline]
    fn from_position(thread_id: ThreadId, stage: Stage, global_order: SortOrder) -> Self {
        let block_size = 2u32 << stage.as_u32();
        let block_index = thread_id.as_u32() / block_size;
        let block_ascending = block_index % 2 == 0;

        Self {
            block_ascending: match global_order {
                SortOrder::Ascending => block_ascending,
                SortOrder::Descending => !block_ascending,
            },
        }
    }

    #[inline]
    fn should_swap<T: PartialOrd>(&self, val_i: T, val_j: T) -> bool {
        if self.block_ascending {
            val_i > val_j
        } else {
            val_i < val_j
        }
    }
}

/// Generic comparison and swap operation
#[inline]
fn compare_and_swap<T>(data: &mut [T], pair: ComparisonPair, direction: BitonicDirection)
where
    T: Copy + PartialOrd,
{
    let i = pair.lower.as_usize();
    let j = pair.upper.as_usize();

    let val_i = data[i];
    let val_j = data[j];

    if direction.should_swap(val_i, val_j) {
        data[i] = val_j;
        data[j] = val_i;
    }
}

/// Common bitonic sort logic that works on both CUDA and Vulkan
#[inline]
pub fn bitonic_sort_step(
    thread_id: ThreadId,
    data: &mut [u32],
    stage: Stage,
    pass: Pass,
    num_elements: u32,
    sort_order: SortOrder,
) {
    // Early exit for out-of-bounds threads
    if thread_id.as_u32() >= num_elements {
        return;
    }

    // Calculate comparison distance for this pass
    let distance = ComparisonDistance::from_stage_pass(stage, pass);

    // Find comparison partner
    let partner = distance.find_partner(thread_id);

    // Create comparison pair if valid
    let (is_valid, pair) = ComparisonPair::try_new(thread_id, partner);
    if is_valid && pair.is_in_bounds(num_elements) {
        // Determine sort direction for this comparison
        let direction = BitonicDirection::from_position(thread_id, stage, sort_order);

        // Perform the comparison and swap
        compare_and_swap(data, pair, direction);
    }
}

/// GPU entry point for Vulkan/SPIR-V
#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(256)))]
pub fn bitonic_kernel(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
    #[spirv(push_constant)] params: &BitonicParams,
) {
    let thread_id = ThreadId::new(gid.x);

    bitonic_sort_step(
        thread_id,
        data,
        params.stage,
        params.pass_of_stage,
        params.num_elements,
        params.sort_order,
    );
}

/// GPU entry point for CUDA
#[cfg(target_os = "cuda")]
#[kernel]
pub unsafe fn bitonic_kernel(data: *mut u32, params: BitonicParams) {
    let thread_id =
        ThreadId::new(thread::thread_idx_x() + thread::block_idx_x() * thread::block_dim_x());

    // Create a slice from the raw pointer
    // Safety: The caller must ensure the pointer is valid for num_elements
    let data_slice = core::slice::from_raw_parts_mut(data, params.num_elements as usize);

    bitonic_sort_step(
        thread_id,
        data_slice,
        params.stage,
        params.pass_of_stage,
        params.num_elements,
        params.sort_order,
    );
}
