// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
//! Quantized linear weights in MLX affine form, as the checkpoint loader in
//! `lfm.rs` hands them to the kernels.

use crate::tensor::Tensor;

/// One linear layer's weights in MLX affine form: `codes` packs `32 / bits`
/// elements per u32 along the input dim (low element first), dequantized per
/// `group_size` input elements as `w = scale * q + bias`. `bits` is 4 for the
/// linear projections and 6 for the token embedding.
pub struct QuantWeights {
    /// U32 `[out, in * bits / 32]`.
    pub codes: Tensor,
    /// BF16 `[out, in/group_size]`.
    pub scales: Tensor,
    /// BF16 `[out, in/group_size]`.
    pub biases: Tensor,
    pub group_size: usize,
    pub bits: usize,
}

impl QuantWeights {
    pub fn out_features(&self) -> usize {
        self.codes.shape()[0]
    }

    pub fn in_features(&self) -> usize {
        self.codes.shape()[1] * 32 / self.bits
    }
}
