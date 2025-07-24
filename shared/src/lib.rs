//! Shared types for the compute demos

#![no_std]

use bytemuck::{Pod, Zeroable};
use core::fmt::{self, Display};

/// Workgroup size for compute shaders
/// IMPORTANT: This must be kept in sync with the literal value in kernel/src/lib.rs
pub const WORKGROUP_SIZE: u32 = 256;

/// CUDA-specific alias for WORKGROUP_SIZE (CUDA uses "block" terminology)
#[cfg(feature = "cuda")]
pub const BLOCK_SIZE: u32 = WORKGROUP_SIZE;

/// The constant used in the computation (index * 2 + COMPUTE_CONSTANT)
pub const COMPUTE_CONSTANT: u32 = 42;

/// Newtype wrapper for thread IDs to ensure type safety
#[derive(Copy, Clone, Debug)]
pub struct ThreadId(u32);

impl ThreadId {
    #[inline]
    #[cfg(not(any(target_arch = "spirv", target_os = "cuda")))]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    #[cfg(any(target_arch = "spirv", target_os = "cuda"))]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }

    #[inline]
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

/// Push constants shared between CPU and GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConstants {
    pub num_elements: u32,
}

// Bitonic sort implementation
// A comparison-based sorting algorithm well-suited for parallel execution on GPUs

/// Newtype wrapper for bitonic sort stage
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(transparent)]
pub struct Stage(pub u32);

impl Stage {
    #[inline]
    pub fn new(stage: u32) -> Self {
        Self(stage)
    }

    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// Newtype wrapper for pass within a stage
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(transparent)]
pub struct Pass(pub u32);

impl Pass {
    #[inline]
    pub fn new(pass: u32) -> Self {
        Self(pass)
    }

    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// Sort order
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u32)]
pub enum SortOrder {
    Ascending = 0,
    Descending = 1,
}

impl Display for SortOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SortOrder::Ascending => write!(f, "ascending"),
            SortOrder::Descending => write!(f, "descending"),
        }
    }
}

// Implement DeviceCopy for CUDA support
#[cfg(feature = "cuda")]
use cust::memory::DeviceCopy;

#[cfg(feature = "cuda")]
unsafe impl DeviceCopy for Stage {}

#[cfg(feature = "cuda")]
unsafe impl DeviceCopy for Pass {}

#[cfg(feature = "cuda")]
unsafe impl DeviceCopy for SortOrder {}

#[cfg(feature = "cuda")]
unsafe impl DeviceCopy for BitonicParams {}

impl TryFrom<u32> for SortOrder {
    type Error = &'static str;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SortOrder::Ascending),
            1 => Ok(SortOrder::Descending),
            _ => Err("Invalid SortOrder value"),
        }
    }
}

impl From<SortOrder> for u32 {
    fn from(order: SortOrder) -> u32 {
        match order {
            SortOrder::Ascending => 0,
            SortOrder::Descending => 1,
        }
    }
}

/// Trait for types that can be sorted using bitonic sort
pub trait SortableKey: Copy + Pod + Zeroable + PartialOrd {
    /// Convert to sortable unsigned representation for GPU operations
    fn to_sortable_u32(&self) -> u32;

    /// Convert back from sortable representation
    fn from_sortable_u32(val: u32) -> Self;

    /// Compare two values for sorting
    fn should_swap(&self, other: &Self, order: SortOrder) -> bool {
        match order {
            SortOrder::Ascending => self > other,
            SortOrder::Descending => self < other,
        }
    }

    /// Get the maximum value for this type (used for padding)
    fn max_value() -> Self;

    /// Get the minimum value for this type (used for padding)
    fn min_value() -> Self;
}

// Implement SortableKey for u32
impl SortableKey for u32 {
    #[inline]
    fn to_sortable_u32(&self) -> u32 {
        *self
    }

    #[inline]
    fn from_sortable_u32(val: u32) -> Self {
        val
    }

    #[inline]
    fn max_value() -> Self {
        u32::MAX
    }

    #[inline]
    fn min_value() -> Self {
        u32::MIN
    }
}

// Implement SortableKey for i32
impl SortableKey for i32 {
    #[inline]
    fn to_sortable_u32(&self) -> u32 {
        // Flip sign bit to make negative numbers sort correctly
        (*self as u32) ^ (1 << 31)
    }

    #[inline]
    fn from_sortable_u32(val: u32) -> Self {
        (val ^ (1 << 31)) as i32
    }

    #[inline]
    fn max_value() -> Self {
        i32::MAX
    }

    #[inline]
    fn min_value() -> Self {
        i32::MIN
    }
}

// Implement SortableKey for f32
impl SortableKey for f32 {
    #[inline]
    fn to_sortable_u32(&self) -> u32 {
        let bits = self.to_bits();
        // If negative, flip all bits; if positive, flip just sign bit
        if bits & (1 << 31) != 0 {
            !bits
        } else {
            bits | (1 << 31)
        }
    }

    #[inline]
    fn from_sortable_u32(val: u32) -> Self {
        let bits = if val & (1 << 31) != 0 {
            val & !(1 << 31)
        } else {
            !val
        };
        f32::from_bits(bits)
    }

    #[inline]
    fn max_value() -> Self {
        f32::INFINITY
    }

    #[inline]
    fn min_value() -> Self {
        f32::NEG_INFINITY
    }
}

/// Parameters for GPU bitonic sorting
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct BitonicParams {
    pub num_elements: u32,
    pub stage: Stage,        // Current stage (for multi-dispatch approach)
    pub pass_of_stage: Pass, // Current pass within stage
    pub sort_order: u32,     // Sort order as u32 (0 = Ascending, 1 = Descending)
}

/// Direction for bitonic compare operations
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CompareDirection {
    Up,   // Sort in ascending order
    Down, // Sort in descending order
}

impl CompareDirection {
    #[inline]
    pub fn from_bool(ascending: bool) -> Self {
        if ascending {
            CompareDirection::Up
        } else {
            CompareDirection::Down
        }
    }

    #[inline]
    pub fn is_ascending(&self) -> bool {
        matches!(self, CompareDirection::Up)
    }
}

/// Helper to check if array size is power of 2
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Get the next power of 2 greater than or equal to n
#[inline]
pub fn next_power_of_two(n: usize) -> usize {
    if is_power_of_two(n) {
        n
    } else {
        1 << (64 - (n - 1).leading_zeros())
    }
}
