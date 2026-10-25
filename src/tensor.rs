// Modified from Lily (Perplexity AI, Apache-2.0): see VENDORED.md for the changes to this file.
//! Device tensors: a Metal buffer plus shape/dtype. Contiguous row-major only.

use anyhow::{Result, anyhow, ensure};
use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use crate::metal::{Buffer, MetalContext};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    BF16,
    F32,
    U32,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::BF16 => 2,
            DType::F32 | DType::U32 => 4,
        }
    }
}

/// Cloning hands out another handle to the SAME storage -- `buf` is refcounted -- which
/// is the relationship [`Tensor::view`] already creates. It is not a copy of the data.
#[derive(Clone)]
pub struct Tensor {
    buf: Buffer,
    shape: Vec<usize>,
    dtype: DType,
    /// Byte offset of element 0 within `buf` — nonzero for views produced by
    /// [`Self::view`].
    offset: usize,
}

impl Tensor {
    pub fn zeros(ctx: &MetalContext, shape: &[usize], dtype: DType) -> Result<Self> {
        let numel: usize = shape.iter().product();
        // `new_buffer` zero-fills (Metal does not guarantee fresh contents).
        let buf = ctx.new_buffer(numel * dtype.size())?;
        Ok(Self { buf, shape: shape.to_vec(), dtype, offset: 0 })
    }

    /// Wraps an existing buffer (e.g. one filled directly from a checkpoint
    /// read) as a tensor.
    pub fn from_buffer(buf: Buffer, shape: &[usize], dtype: DType) -> Result<Self> {
        let numel: usize = shape.iter().product();
        ensure!(
            buf.length() >= numel * dtype.size(),
            "buffer holds {} bytes, shape {shape:?} of {dtype:?} needs {}",
            buf.length(),
            numel * dtype.size(),
        );
        Ok(Self { buf, shape: shape.to_vec(), dtype, offset: 0 })
    }

    /// A differently shaped view of the same storage (same dtype, same
    /// element offset).
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        ensure!(
            numel == self.numel(),
            "reshape {shape:?} has {numel} elements, tensor has {}",
            self.numel()
        );
        Ok(Tensor {
            buf: self.buf.clone(),
            shape: shape.to_vec(),
            dtype: self.dtype,
            offset: self.offset,
        })
    }

    /// A contiguous sub-range of this tensor starting `start` elements in,
    /// sharing the underlying buffer (fused weights/activations hand out
    /// per-segment views this way). Metal requires 4-byte-aligned buffer
    /// offsets, so bf16 views must start at an even element.
    pub fn view(&self, start: usize, shape: &[usize]) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        ensure!(
            start + numel <= self.numel(),
            "view [{start}, {start}+{numel}) exceeds tensor numel {}",
            self.numel()
        );
        let offset = self.offset + start * self.dtype.size();
        ensure!(offset.is_multiple_of(4), "view byte offset {offset} not 4-aligned");
        Ok(Tensor {
            buf: self.buf.clone(),
            shape: shape.to_vec(),
            dtype: self.dtype,
            offset,
        })
    }

    /// Raw payload bytes, for hashing without widening to f32. Same idle-GPU
    /// contract as [`Self::to_f32`].
    pub fn raw_bytes(&self) -> &[u8] {
        self.contents()
    }

    /// The tensor's payload size in bytes.
    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    /// Zeroes the tensor's contents. Same idle-GPU contract as
    /// [`Self::write_bytes`].
    pub fn zero_fill(&self) {
        // SAFETY: shared-storage buffer holding at least offset + numel*size
        // bytes (checked at construction); the GPU is idle on this buffer per
        // the doc contract.
        unsafe {
            core::ptr::write_bytes(
                self.buf.contents().as_ptr().cast::<u8>().add(self.offset),
                0,
                self.byte_len(),
            );
        }
    }

    /// Overwrites the tensor's contents from host memory. Callers must
    /// ensure no committed-but-unfinished GPU pass touches this buffer
    /// (prefill uploads between commit_wait-synchronized chunks).
    pub fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        ensure!(
            bytes.len() == self.numel() * self.dtype.size(),
            "byte length {} does not match shape {:?} of {:?}",
            bytes.len(),
            self.shape,
            self.dtype,
        );
        // SAFETY: shared-storage buffer holding at least offset + numel*size
        // bytes (checked at construction); the GPU is idle on this buffer per
        // the doc contract.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.buf.contents().as_ptr().cast::<u8>().add(self.offset),
                bytes.len(),
            );
        }
        Ok(())
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Buffer and byte offset, the pair every kernel binding needs.
    pub fn binding(&self) -> (&ProtocolObject<dyn MTLBuffer>, usize) {
        (&self.buf, self.offset)
    }

    fn contents(&self) -> &[u8] {
        // SAFETY: shared-storage buffer holding at least offset + numel*size
        // bytes (checked at construction); the GPU is idle when hosts read
        // (callers commit_wait first).
        unsafe {
            core::slice::from_raw_parts(
                self.buf.contents().as_ptr().cast::<u8>().add(self.offset),
                self.numel() * self.dtype.size(),
            )
        }
    }

    /// Reads the tensor back to host memory as f32 (converting from bf16 when
    /// needed).
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        match self.dtype {
            DType::F32 => Ok(bytemuck::cast_slice(self.contents()).to_vec()),
            DType::BF16 => {
                let vals: &[bf16] = bytemuck::cast_slice(self.contents());
                Ok(vals.iter().map(|v| v.to_f32()).collect())
            }
            DType::U32 => Err(anyhow!("to_f32 on U32 tensor")),
        }
    }

    pub fn to_u32(&self) -> Result<Vec<u32>> {
        ensure!(self.dtype == DType::U32, "to_u32 on {:?} tensor", self.dtype);
        Ok(bytemuck::cast_slice(self.contents()).to_vec())
    }
}
