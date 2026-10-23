//! Bounds-checked loads and stores for the SIMD kernels.
//!
//! The load and store intrinsics take raw pointers, so a kernel that calls
//! them directly is only as safe as its index arithmetic. These wrappers take
//! a slice and an index instead and slice out the whole vector first, so an
//! out-of-range index panics rather than touching memory outside the slice.
//! Kernels built on them need no `unsafe` of their own beyond the call into a
//! `#[target_feature]` function. The slicing is cheap, and the compiler drops
//! it altogether when the loop bound already proves the vector is in range.

/// Plain-Rust building blocks for kernels that let the compiler vectorize.
///
/// Compiled inside a `#[target_feature]` function, code written with these
/// gets the same vector instructions the hand-written intrinsics used, while
/// keeping their exact arithmetic, so results stay bit-identical.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod lanes {
    #[cfg(not(feature = "std"))]
    use crate::compat::Math;

    /// `a * b + c` with a single rounding, as `_mm256_fmadd_ps` computes it.
    #[inline(always)]
    pub(crate) fn fmadd(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// Loads `s[i..i + 8]`.
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn load8(s: &[f32], i: usize) -> __m256 {
        let v = &s[i..i + 8];
        // SAFETY: `v` is 8 readable f32s; loadu has no alignment requirement.
        unsafe { _mm256_loadu_ps(v.as_ptr()) }
    }

    /// Stores `v` to `s[i..i + 8]`.
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn store8(s: &mut [f32], i: usize, v: __m256) {
        let d = &mut s[i..i + 8];
        // SAFETY: `d` is 8 writable f32s; storeu has no alignment requirement.
        unsafe { _mm256_storeu_ps(d.as_mut_ptr(), v) }
    }

    /// Loads `s[i..i + 4]`.
    #[inline]
    #[target_feature(enable = "sse")]
    pub(crate) fn load4(s: &[f32], i: usize) -> __m128 {
        let v = &s[i..i + 4];
        // SAFETY: `v` is 4 readable f32s; loadu has no alignment requirement.
        unsafe { _mm_loadu_ps(v.as_ptr()) }
    }

    /// Stores `v` to `s[i..i + 4]`.
    #[inline]
    #[target_feature(enable = "sse")]
    pub(crate) fn store4(s: &mut [f32], i: usize, v: __m128) {
        let d = &mut s[i..i + 4];
        // SAFETY: `d` is 4 writable f32s; storeu has no alignment requirement.
        unsafe { _mm_storeu_ps(d.as_mut_ptr(), v) }
    }

    /// Loads `s[i..i + 8]` as eight i32 lanes.
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn load8_i32(s: &[i32], i: usize) -> __m256i {
        let v = &s[i..i + 8];
        // SAFETY: `v` is 32 readable bytes; loadu has no alignment requirement.
        unsafe { _mm256_loadu_si256(v.as_ptr().cast()) }
    }

    /// Stores eight i32 lanes to `s[i..i + 8]`.
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) fn store8_i32(s: &mut [i32], i: usize, v: __m256i) {
        let d = &mut s[i..i + 8];
        // SAFETY: `d` is 32 writable bytes; storeu has no alignment requirement.
        unsafe { _mm256_storeu_si256(d.as_mut_ptr().cast(), v) }
    }

    /// Stores four i32 lanes to `s[i..i + 4]`.
    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn store4_i32(s: &mut [i32], i: usize, v: __m128i) {
        let d = &mut s[i..i + 4];
        // SAFETY: `d` is 16 writable bytes; storeu has no alignment requirement.
        unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), v) }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use core::arch::aarch64::*;

    /// Loads `s[i..i + 4]`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn ld4(s: &[f32], i: usize) -> float32x4_t {
        let v = &s[i..i + 4];
        // SAFETY: `v` is 4 readable f32s; vld1q has no alignment requirement
        // beyond that of f32.
        unsafe { vld1q_f32(v.as_ptr()) }
    }

    /// Stores `v` to `s[i..i + 4]`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn st4(s: &mut [f32], i: usize, v: float32x4_t) {
        let d = &mut s[i..i + 4];
        // SAFETY: `d` is 4 writable f32s.
        unsafe { vst1q_f32(d.as_mut_ptr(), v) }
    }

    /// Loads `s[i..i + 2]`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn ld2(s: &[f32], i: usize) -> float32x2_t {
        let v = &s[i..i + 2];
        // SAFETY: `v` is 2 readable f32s.
        unsafe { vld1_f32(v.as_ptr()) }
    }

    /// Loads `s[i..i + 8]` de-interleaved: even elements in `.0`, odd in `.1`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn ld2x4(s: &[f32], i: usize) -> float32x4x2_t {
        let v = &s[i..i + 8];
        // SAFETY: `v` is 8 readable f32s, exactly what vld2q reads.
        unsafe { vld2q_f32(v.as_ptr()) }
    }

    /// Stores `v` to `s[i..i + 8]` interleaved: `.0` to even, `.1` to odd.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn st2x4(s: &mut [f32], i: usize, v: float32x4x2_t) {
        let d = &mut s[i..i + 8];
        // SAFETY: `d` is 8 writable f32s, exactly what vst2q writes.
        unsafe { vst2q_f32(d.as_mut_ptr(), v) }
    }

    /// Loads `s[i..i + 4]` as four i32 lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn ld4_s32(s: &[i32], i: usize) -> int32x4_t {
        let v = &s[i..i + 4];
        // SAFETY: `v` is 4 readable i32s.
        unsafe { vld1q_s32(v.as_ptr()) }
    }

    /// Stores four i32 lanes to `s[i..i + 4]`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn st4_s32(s: &mut [i32], i: usize, v: int32x4_t) {
        let d = &mut s[i..i + 4];
        // SAFETY: `d` is 4 writable i32s.
        unsafe { vst1q_s32(d.as_mut_ptr(), v) }
    }
}
