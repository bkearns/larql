//! Minimal CPU reference for Kimi/DeepSeek2 MLA attention from GGUF manifest tensors.
//!
//! This is a correctness seam for manifest-backed decode: it consumes exactly one
//! dequantized attention layer and a single hidden-state token. It intentionally
//! does not touch MoE expert tensors.

use larql_models::loading::gguf::LoadedGgufTensor;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use ndarray::ArrayView2;

use super::gguf_manifest::Deepseek2AttentionLayerTensors;
use crate::error::InferenceError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deepseek2MlaShape {
    pub hidden: usize,
    pub q_rank: usize,
    pub kv_rank: usize,
    pub heads: usize,
    pub qk_nope: usize,
    pub qk_rope: usize,
    pub v_dim: usize,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub struct Deepseek2CudaResident2dAttention {
    pub layer: usize,
    pub q_a: larql_compute::cuda::CudaResidentF32Matrix,
    pub q_b: larql_compute::cuda::CudaResidentF32Matrix,
    pub kv_a_mqa: larql_compute::cuda::CudaResidentF32Matrix,
    pub output: larql_compute::cuda::CudaResidentF32Matrix,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
impl Deepseek2CudaResident2dAttention {
    pub fn resident_matrix_count(&self) -> usize {
        4
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub fn load_deepseek2_cuda_resident_2d_attention(
    cuda: &larql_compute::cuda::CudaBackend,
    layer: &Deepseek2AttentionLayerTensors,
) -> Result<Deepseek2CudaResident2dAttention, InferenceError> {
    let prefix = format!("blk.{}.attn", layer.layer);
    Ok(Deepseek2CudaResident2dAttention {
        layer: layer.layer,
        q_a: resident_2d(cuda, required(layer, &format!("{prefix}_q_a.weight"))?)?,
        q_b: resident_2d(cuda, required(layer, &format!("{prefix}_q_b.weight"))?)?,
        kv_a_mqa: resident_2d(cuda, required(layer, &format!("{prefix}_kv_a_mqa.weight"))?)?,
        output: resident_2d(cuda, required(layer, &format!("{prefix}_output.weight"))?)?,
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn resident_2d(
    cuda: &larql_compute::cuda::CudaBackend,
    tensor: &LoadedGgufTensor,
) -> Result<larql_compute::cuda::CudaResidentF32Matrix, InferenceError> {
    expect_rank(tensor, 2)?;
    let cols = tensor.dims[0] as usize;
    let rows = tensor.dims[1] as usize;
    let view = ArrayView2::from_shape((rows, cols), &tensor.values).map_err(|err| {
        InferenceError::Parse(format!(
            "{} cannot be viewed as row-major [{rows},{cols}]: {err}",
            tensor.name
        ))
    })?;
    cuda.resident_f32_matrix(view).ok_or_else(|| {
        InferenceError::Parse(format!(
            "failed to stage {} as CUDA-resident f32 matrix",
            tensor.name
        ))
    })
}

pub fn deepseek2_mla_single_token_cpu(
    layer: &Deepseek2AttentionLayerTensors,
    hidden: &[f32],
) -> Result<Vec<f32>, InferenceError> {
    let prefix = format!("blk.{}.attn", layer.layer);
    let q_a = required(layer, &format!("{prefix}_q_a.weight"))?;
    let q_a_norm = required(layer, &format!("{prefix}_q_a_norm.weight"))?;
    let q_b = required(layer, &format!("{prefix}_q_b.weight"))?;
    let kv_a = required(layer, &format!("{prefix}_kv_a_mqa.weight"))?;
    let kv_a_norm = required(layer, &format!("{prefix}_kv_a_norm.weight"))?;
    let k_b = required(layer, &format!("{prefix}_k_b.weight"))?;
    let v_b = required(layer, &format!("{prefix}_v_b.weight"))?;
    let output = required(layer, &format!("{prefix}_output.weight"))?;
    let attn_norm = layer.get(&format!("{prefix}_norm.weight"));

    let shape = infer_shape(q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, output)?;
    if hidden.len() != shape.hidden {
        return Err(InferenceError::Parse(format!(
            "hidden length {} does not match DeepSeek2 MLA hidden {}",
            hidden.len(),
            shape.hidden
        )));
    }

    let normed_hidden = if let Some(norm) = attn_norm {
        rms_norm(hidden, &norm.values)?
    } else {
        hidden.to_vec()
    };
    let q_latent = matvec_2d(q_a, &normed_hidden)?;
    let q_latent = rms_norm(&q_latent, &q_a_norm.values)?;
    let q = matvec_2d(q_b, &q_latent)?;

    let kv = matvec_2d(kv_a, &normed_hidden)?;
    let kv_latent = rms_norm(&kv[..shape.kv_rank], &kv_a_norm.values)?;
    let k_rope = &kv[shape.kv_rank..shape.kv_rank + shape.qk_rope];
    let k_nope = matvec_3d_heads(k_b, &kv_latent)?;
    let v = matvec_3d_heads(v_b, &kv_latent)?;

    // Single-token causal attention has a length-1 softmax per head, so the
    // attention probability is 1.0. Still compute a score to validate q/k
    // geometry and catch dimension drift.
    for head in 0..shape.heads {
        let q_base = head * (shape.qk_nope + shape.qk_rope);
        let k_base = head * shape.qk_nope;
        let _score_nope: f32 = q[q_base..q_base + shape.qk_nope]
            .iter()
            .zip(&k_nope[k_base..k_base + shape.qk_nope])
            .map(|(a, b)| a * b)
            .sum();
        let _score_rope: f32 = q[q_base + shape.qk_nope..q_base + shape.qk_nope + shape.qk_rope]
            .iter()
            .zip(k_rope.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    matvec_2d(output, &v)
}

fn required<'a>(
    layer: &'a Deepseek2AttentionLayerTensors,
    name: &str,
) -> Result<&'a LoadedGgufTensor, InferenceError> {
    layer
        .get(name)
        .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
}

fn infer_shape(
    q_a: &LoadedGgufTensor,
    q_a_norm: &LoadedGgufTensor,
    q_b: &LoadedGgufTensor,
    kv_a: &LoadedGgufTensor,
    kv_a_norm: &LoadedGgufTensor,
    k_b: &LoadedGgufTensor,
    v_b: &LoadedGgufTensor,
    output: &LoadedGgufTensor,
) -> Result<Deepseek2MlaShape, InferenceError> {
    expect_rank(q_a, 2)?;
    expect_rank(q_a_norm, 1)?;
    expect_rank(q_b, 2)?;
    expect_rank(kv_a, 2)?;
    expect_rank(kv_a_norm, 1)?;
    expect_rank(k_b, 3)?;
    expect_rank(v_b, 3)?;
    expect_rank(output, 2)?;

    let hidden = q_a.dims[0] as usize;
    let q_rank = q_a.dims[1] as usize;
    let kv_rank = kv_a_norm.dims[0] as usize;
    let heads = k_b.dims[2] as usize;
    let qk_nope = if k_b.dims[1] as usize == kv_rank {
        k_b.dims[0] as usize
    } else {
        k_b.dims[1] as usize
    };
    let kv_total = kv_a.dims[1] as usize;
    let qk_rope = kv_total
        .checked_sub(kv_rank)
        .ok_or_else(|| InferenceError::Parse("kv_a output smaller than kv rank".to_string()))?;
    let q_width = q_b.dims[1] as usize;
    if q_width != heads * (qk_nope + qk_rope) {
        return Err(InferenceError::Parse(format!(
            "q_b output width {q_width} != heads {heads} × (qk_nope {qk_nope} + qk_rope {qk_rope})"
        )));
    }
    let v_dim = if v_b.dims[0] as usize == kv_rank {
        v_b.dims[1] as usize
    } else {
        v_b.dims[0] as usize
    };
    if q_a_norm.values.len() != q_rank || q_b.dims[0] as usize != q_rank {
        return Err(InferenceError::Parse(
            "q-rank tensor dimensions disagree".to_string(),
        ));
    }
    if kv_a.dims[0] as usize != hidden
        || !(k_b.dims[0] as usize == kv_rank || k_b.dims[1] as usize == kv_rank)
        || !(v_b.dims[0] as usize == kv_rank || v_b.dims[1] as usize == kv_rank)
    {
        return Err(InferenceError::Parse(
            "kv-rank tensor dimensions disagree".to_string(),
        ));
    }
    if output.dims[0] as usize != heads * v_dim || output.dims[1] as usize != hidden {
        return Err(InferenceError::Parse(
            "output projection dimensions disagree".to_string(),
        ));
    }

    Ok(Deepseek2MlaShape {
        hidden,
        q_rank,
        kv_rank,
        heads,
        qk_nope,
        qk_rope,
        v_dim,
    })
}

fn matvec_2d(tensor: &LoadedGgufTensor, x: &[f32]) -> Result<Vec<f32>, InferenceError> {
    expect_rank(tensor, 2)?;
    let cols = tensor.dims[0] as usize;
    let rows = tensor.dims[1] as usize;
    if x.len() != cols {
        return Err(InferenceError::Parse(format!(
            "{} expects input width {cols}, got {}",
            tensor.name,
            x.len()
        )));
    }
    let mut out = vec![0.0f32; rows];
    for (row, out_cell) in out.iter_mut().enumerate() {
        let base = row * cols;
        *out_cell = tensor.values[base..base + cols]
            .iter()
            .zip(x)
            .map(|(w, v)| w * v)
            .sum();
    }
    Ok(out)
}

fn matvec_3d_heads(tensor: &LoadedGgufTensor, x: &[f32]) -> Result<Vec<f32>, InferenceError> {
    expect_rank(tensor, 3)?;
    let d0 = tensor.dims[0] as usize;
    let d1 = tensor.dims[1] as usize;
    let heads = tensor.dims[2] as usize;
    if x.len() == d0 {
        let cols = d0;
        let rows = d1;
        let mut out = vec![0.0f32; heads * rows];
        for head in 0..heads {
            for row in 0..rows {
                let base = (head * rows + row) * cols;
                out[head * rows + row] = tensor.values[base..base + cols]
                    .iter()
                    .zip(x)
                    .map(|(w, v)| w * v)
                    .sum();
            }
        }
        Ok(out)
    } else if x.len() == d1 {
        let rows = d0;
        let cols = d1;
        let mut out = vec![0.0f32; heads * rows];
        for head in 0..heads {
            for row in 0..rows {
                let mut sum = 0.0f32;
                for col in 0..cols {
                    let idx = (head * cols + col) * rows + row;
                    sum += tensor.values[idx] * x[col];
                }
                out[head * rows + row] = sum;
            }
        }
        Ok(out)
    } else {
        Err(InferenceError::Parse(format!(
            "{} expects input width {} or {}, got {}",
            tensor.name,
            d0,
            d1,
            x.len()
        )))
    }
}

fn rms_norm(x: &[f32], weight: &[f32]) -> Result<Vec<f32>, InferenceError> {
    if x.len() != weight.len() {
        return Err(InferenceError::Parse(format!(
            "RMS norm width mismatch: x={} weight={}",
            x.len(),
            weight.len()
        )));
    }
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let scale = (mean_square + 1e-6).sqrt().recip();
    Ok(x.iter()
        .zip(weight)
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

fn expect_rank(tensor: &LoadedGgufTensor, rank: usize) -> Result<(), InferenceError> {
    if tensor.dims.len() == rank {
        Ok(())
    } else {
        Err(InferenceError::Parse(format!(
            "{} rank {} != expected {rank}",
            tensor.name,
            tensor.dims.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn tensor(name: &str, dims: Vec<u64>, values: Vec<f32>) -> LoadedGgufTensor {
        LoadedGgufTensor {
            name: name.to_string(),
            dims,
            tensor_type: larql_models::quant::ggml::TYPE_F32,
            values,
        }
    }

    #[test]
    fn deepseek2_mla_single_token_cpu_runs_tiny_layer() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "blk.0.attn_norm.weight".to_string(),
            tensor("blk.0.attn_norm.weight", vec![2], vec![1.0, 1.0]),
        );
        tensors.insert(
            "blk.0.attn_q_a.weight".to_string(),
            tensor("blk.0.attn_q_a.weight", vec![2, 1], vec![1.0, 0.0]),
        );
        tensors.insert(
            "blk.0.attn_q_a_norm.weight".to_string(),
            tensor("blk.0.attn_q_a_norm.weight", vec![1], vec![1.0]),
        );
        tensors.insert(
            "blk.0.attn_q_b.weight".to_string(),
            tensor("blk.0.attn_q_b.weight", vec![1, 2], vec![1.0, 1.0]),
        );
        tensors.insert(
            "blk.0.attn_kv_a_mqa.weight".to_string(),
            tensor(
                "blk.0.attn_kv_a_mqa.weight",
                vec![2, 2],
                vec![1.0, 0.0, 0.0, 1.0],
            ),
        );
        tensors.insert(
            "blk.0.attn_kv_a_norm.weight".to_string(),
            tensor("blk.0.attn_kv_a_norm.weight", vec![1], vec![1.0]),
        );
        tensors.insert(
            "blk.0.attn_k_b.weight".to_string(),
            tensor("blk.0.attn_k_b.weight", vec![1, 1, 1], vec![1.0]),
        );
        tensors.insert(
            "blk.0.attn_v_b.weight".to_string(),
            tensor("blk.0.attn_v_b.weight", vec![1, 1, 1], vec![2.0]),
        );
        tensors.insert(
            "blk.0.attn_output.weight".to_string(),
            tensor("blk.0.attn_output.weight", vec![1, 2], vec![3.0, 4.0]),
        );
        let layer = Deepseek2AttentionLayerTensors { layer: 0, tensors };

        let out = deepseek2_mla_single_token_cpu(&layer, &[2.0, 0.0]).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|value| value.is_finite()));
        assert!(out[0] > 0.0);
        assert!(out[1] > out[0]);
    }
}
