//! Attention computation — RoPE, GQA, causal masking, GPU dispatch.
//!
//! Submodules:
//! - `rope`: Rotary Position Embeddings (full and partial rotation)
//! - `gqa`: Grouped-Query Attention with BLAS-fused dot products
//! - `block`: CPU attention block (norm → proj → RoPE → GQA → O → residual)
//! - `gpu`: GPU-accelerated attention, KV-capture, Q4 projection

pub mod block;
pub mod decode;
pub mod deepseek2_mla;
pub mod gguf_manifest;
pub mod gpu;
pub mod gqa;
pub mod rope;

use ndarray::Array2;

/// Per-head attention weights for the last token position.
pub struct AttentionWeights {
    /// Per-head attention distribution for the last sequence position.
    /// `heads[h][j]` = attention weight from last token to position j.
    pub heads: Vec<Vec<f32>>,
}

/// Per-head attention weights for every query position.
///
/// `heads[h][i][j]` = attention weight from query position `i` to source
/// position `j`. Rows are padded to the full sequence length; causal-future
/// entries are zero.
pub struct AttentionAllWeights {
    pub heads: Vec<Vec<Vec<f32>>>,
}

/// Shared KV pair: post-RoPE K and post-V-norm V from a source layer.
pub type SharedKV = (Array2<f32>, Array2<f32>);

// ── Re-exports: preserve `crate::attention::*` paths ──

pub use block::{
    run_attention_block, run_attention_block_replace_head_residual_delta,
    run_attention_block_replace_pre_o_head, run_attention_block_shared,
    run_attention_block_shared_with_pre_o, run_attention_block_subtract_pre_o_heads,
    run_attention_block_with_kv_out, run_attention_block_with_pre_o,
    run_attention_block_with_pre_o_and_all_attention_weights,
    run_attention_block_with_pre_o_and_reduced_qk_attention_weights,
    run_attention_block_zero_pre_o_heads,
};
pub use decode::{
    gqa_attention_decode_step, run_attention_block_decode_step,
    run_attention_block_decode_step_backend, KvCache,
};
pub use deepseek2_mla::{deepseek2_mla_single_token_cpu, Deepseek2MlaShape};
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use deepseek2_mla::{
    load_deepseek2_cuda_resident_2d_attention, Deepseek2CudaResident2dAttention,
};
pub use gguf_manifest::{
    load_deepseek2_attention_layer_tensors, load_deepseek2_attention_manifest_plan,
    Deepseek2AttentionLayerPlan, Deepseek2AttentionLayerTensors, Deepseek2AttentionManifestPlan,
    GgufAttentionTensorRef,
};
pub use gpu::{
    q4_attention_proj, run_attention_block_gpu, run_attention_with_kv,
    run_attention_with_kv_backend,
};
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use gpu::{run_attention_block_cuda_resident, CudaAttentionResidency};
pub use gqa::{gqa_attention, gqa_attention_with_all_weights, gqa_attention_with_weights};
pub use rope::{apply_rope, apply_rope_partial, apply_rope_partial_at};
