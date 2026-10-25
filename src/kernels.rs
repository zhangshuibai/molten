// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
//! Runtime-compiled Metal kernels and Rust dispatch wrappers.

pub mod attention;
pub mod elementwise;
pub mod gemm;
pub mod lfm;
pub mod norm;
pub mod quant;
pub mod steel;

pub fn u32_bytes(v: usize) -> [u8; 4] {
    (v as u32).to_ne_bytes()
}
