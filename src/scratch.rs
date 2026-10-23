//! Stack scratch buffers that only pay for the part they use.
//!
//! Several hot paths keep a buffer sized for the largest band or transform
//! on the stack but only use its first `n` elements. Zeroing the whole
//! buffer on every call would cost more than the work done on a short band,
//! so the buffer starts uninitialised and [`zeroed_prefix`] clears and hands
//! out just the prefix. All of the `unsafe` this needs lives here, where it
//! can be checked once.

use core::mem::MaybeUninit;

/// Zeroes `buf[..n]` and returns it as an initialised slice.
///
/// Panics if `n > buf.len()`, so a caller cannot get a slice that reaches
/// past the buffer.
#[inline(always)]
pub(crate) fn zeroed_prefix<T: Copy + Default>(buf: &mut [MaybeUninit<T>], n: usize) -> &mut [T] {
    let buf = &mut buf[..n];
    for v in buf.iter_mut() {
        v.write(T::default());
    }
    // SAFETY: every element of `buf` was written just above, and
    // `MaybeUninit<T>` has the same size, alignment and layout as `T`.
    unsafe { &mut *(buf as *mut [MaybeUninit<T>] as *mut [T]) }
}

/// An uninitialised buffer of `N` elements, to be handed to [`zeroed_prefix`].
#[inline(always)]
pub(crate) const fn uninit<T: Copy, const N: usize>() -> [MaybeUninit<T>; N] {
    [MaybeUninit::uninit(); N]
}
