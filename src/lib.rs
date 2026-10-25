// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
//! Molten: a Metal (Apple Silicon) inference engine for LFM2.5-24B-A2B, built
//! on the `objc2` bindings.
//!
//! Kernels are compiled from MSL source at runtime, so building or running
//! this crate needs no offline `.metal` -> `.metallib` step.

pub mod chat_template;
#[cfg(target_os = "macos")]
pub mod cli;
pub mod json;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod lfm;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod parse;
#[cfg(target_os = "macos")]
pub mod protocol;
pub mod safetensors;
#[cfg(target_os = "macos")]
pub mod server;
#[cfg(target_os = "macos")]
pub mod session;
#[cfg(target_os = "macos")]
pub mod tensor;
pub mod text;
#[cfg(target_os = "macos")]
pub mod weights;
