//! Compact GGUF attention-manifest planning for split-residency Kimi/DeepSeek2.
//!
//! This module intentionally plans dense attention/MLA tensors separately from
//! mmap-backed MoE expert tensors. It does not materialize tensor data yet; the
//! next loader stage can consume the plan to decide what may be staged on CUDA.

use std::collections::BTreeMap;
use std::path::Path;

use larql_vindex::format::filenames::GGUF_ATTENTION_MANIFEST_JSON;
use serde::Deserialize;

use crate::error::InferenceError;

#[derive(Clone, Debug, Deserialize)]
struct RawGgufAttentionManifest {
    version: u32,
    architecture: String,
    split_count: usize,
    residency: String,
    tensors: Vec<GgufAttentionTensorRef>,
}

/// Byte-range reference to one attention/MLA tensor inside the source GGUF shards.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct GgufAttentionTensorRef {
    pub tensor: String,
    pub source_file: String,
    pub shard_idx: usize,
    pub tensor_type: u32,
    pub dims: Vec<usize>,
    pub tensor_offset: u64,
    pub data_offset: u64,
}

/// Planned dense-attention residency for one DeepSeek2/Kimi layer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Deepseek2AttentionLayerPlan {
    pub layer: usize,
    pub q_a: Option<GgufAttentionTensorRef>,
    pub q_a_norm: Option<GgufAttentionTensorRef>,
    pub q_b: Option<GgufAttentionTensorRef>,
    pub kv_a_mqa: Option<GgufAttentionTensorRef>,
    pub kv_a_norm: Option<GgufAttentionTensorRef>,
    pub k_b: Option<GgufAttentionTensorRef>,
    pub v_b: Option<GgufAttentionTensorRef>,
    pub output: Option<GgufAttentionTensorRef>,
    pub other_attention_tensors: Vec<GgufAttentionTensorRef>,
}

impl Deepseek2AttentionLayerPlan {
    /// True when the layer has the full MLA tensor family needed by Kimi/DeepSeek2.
    pub fn is_mla_complete(&self) -> bool {
        self.q_a.is_some()
            && self.q_a_norm.is_some()
            && self.q_b.is_some()
            && self.kv_a_mqa.is_some()
            && self.kv_a_norm.is_some()
            && self.k_b.is_some()
            && self.v_b.is_some()
            && self.output.is_some()
    }

    pub fn tensors_for_cuda_residency(&self) -> Vec<&GgufAttentionTensorRef> {
        let mut tensors = Vec::new();
        for tensor in [
            self.q_a.as_ref(),
            self.q_a_norm.as_ref(),
            self.q_b.as_ref(),
            self.kv_a_mqa.as_ref(),
            self.kv_a_norm.as_ref(),
            self.k_b.as_ref(),
            self.v_b.as_ref(),
            self.output.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            tensors.push(tensor);
        }
        tensors.extend(self.other_attention_tensors.iter());
        tensors
    }
}

/// Split-residency plan derived from `gguf_attention_manifest.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Deepseek2AttentionManifestPlan {
    pub version: u32,
    pub architecture: String,
    pub split_count: usize,
    pub residency: String,
    pub layers: Vec<Deepseek2AttentionLayerPlan>,
}

impl Deepseek2AttentionManifestPlan {
    pub fn complete_mla_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.is_mla_complete())
            .count()
    }

    /// Only returns tensors classified as attention/MLA tensors. Expert/FFN tensors are
    /// deliberately excluded from this planning seam.
    pub fn tensors_for_cuda_residency(&self) -> Vec<&GgufAttentionTensorRef> {
        self.layers
            .iter()
            .flat_map(|layer| layer.tensors_for_cuda_residency())
            .collect()
    }
}

pub fn load_deepseek2_attention_manifest_plan(
    vindex_dir: &Path,
) -> Result<Deepseek2AttentionManifestPlan, InferenceError> {
    let manifest_path = vindex_dir.join(GGUF_ATTENTION_MANIFEST_JSON);
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let raw: RawGgufAttentionManifest = serde_json::from_str(&manifest_text)
        .map_err(|err| InferenceError::Parse(err.to_string()))?;
    plan_deepseek2_attention_manifest(raw)
}

fn plan_deepseek2_attention_manifest(
    raw: RawGgufAttentionManifest,
) -> Result<Deepseek2AttentionManifestPlan, InferenceError> {
    if raw.architecture != "deepseek2" {
        return Err(InferenceError::Parse(format!(
            "unsupported GGUF attention architecture: {}",
            raw.architecture
        )));
    }

    let mut by_layer: BTreeMap<usize, Deepseek2AttentionLayerPlan> = BTreeMap::new();
    for tensor in raw.tensors {
        let Some((layer, role)) = parse_deepseek2_attention_tensor_role(&tensor.tensor) else {
            continue;
        };
        let plan = by_layer
            .entry(layer)
            .or_insert_with(|| Deepseek2AttentionLayerPlan {
                layer,
                ..Default::default()
            });
        match role {
            "attn_q_a.weight" => plan.q_a = Some(tensor),
            "attn_q_a_norm.weight" => plan.q_a_norm = Some(tensor),
            "attn_q_b.weight" => plan.q_b = Some(tensor),
            "attn_kv_a_mqa.weight" => plan.kv_a_mqa = Some(tensor),
            "attn_kv_a_norm.weight" => plan.kv_a_norm = Some(tensor),
            "attn_k_b.weight" => plan.k_b = Some(tensor),
            "attn_v_b.weight" => plan.v_b = Some(tensor),
            "attn_output.weight" => plan.output = Some(tensor),
            _ => plan.other_attention_tensors.push(tensor),
        }
    }

    Ok(Deepseek2AttentionManifestPlan {
        version: raw.version,
        architecture: raw.architecture,
        split_count: raw.split_count,
        residency: raw.residency,
        layers: by_layer.into_values().collect(),
    })
}

fn parse_deepseek2_attention_tensor_role(name: &str) -> Option<(usize, &str)> {
    let suffix = name.strip_prefix("blk.")?;
    let (layer_text, role) = suffix.split_once('.')?;
    if !role.starts_with("attn_") {
        return None;
    }
    let layer = layer_text.parse().ok()?;
    Some((layer, role))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn deepseek2_attention_manifest_plan_groups_mla_tensors_without_experts() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = r#"{
          "version": 1,
          "architecture": "deepseek2",
          "split_count": 13,
          "residency": "split_residency_cuda_attention_mmap_experts",
          "tensors": [
            {"tensor":"blk.0.attn_q_a.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":8,"dims":[7168,1536],"tensor_offset":10,"data_offset":20},
            {"tensor":"blk.0.attn_q_a_norm.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":0,"dims":[1536],"tensor_offset":11,"data_offset":21},
            {"tensor":"blk.0.attn_q_b.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":8,"dims":[128,1536,64],"tensor_offset":12,"data_offset":22},
            {"tensor":"blk.0.attn_kv_a_mqa.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":8,"dims":[7168,576],"tensor_offset":13,"data_offset":23},
            {"tensor":"blk.0.attn_kv_a_norm.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":0,"dims":[512],"tensor_offset":14,"data_offset":24},
            {"tensor":"blk.0.attn_k_b.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":8,"dims":[128,512,64],"tensor_offset":15,"data_offset":25},
            {"tensor":"blk.0.attn_v_b.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":8,"dims":[128,512,64],"tensor_offset":16,"data_offset":26},
            {"tensor":"blk.0.attn_output.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":8,"dims":[7168,8192],"tensor_offset":17,"data_offset":27},
            {"tensor":"blk.0.ffn_gate_exps.weight","source_file":"shard0.gguf","shard_idx":0,"tensor_type":8,"dims":[7168,2048,384],"tensor_offset":18,"data_offset":28}
          ]
        }"#;
        fs::write(dir.path().join("gguf_attention_manifest.json"), manifest).unwrap();

        let plan = load_deepseek2_attention_manifest_plan(dir.path()).unwrap();
        assert_eq!(plan.architecture, "deepseek2");
        assert_eq!(
            plan.residency,
            "split_residency_cuda_attention_mmap_experts"
        );
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.complete_mla_layers(), 1);
        let layer = &plan.layers[0];
        assert_eq!(layer.layer, 0);
        assert!(layer.is_mla_complete());
        assert_eq!(layer.q_a.as_ref().unwrap().tensor, "blk.0.attn_q_a.weight");
        assert_eq!(
            layer.kv_a_mqa.as_ref().unwrap().tensor,
            "blk.0.attn_kv_a_mqa.weight"
        );
        assert_eq!(
            layer.output.as_ref().unwrap().tensor,
            "blk.0.attn_output.weight"
        );
        assert!(plan
            .tensors_for_cuda_residency()
            .iter()
            .all(|tensor| tensor.tensor.contains("attn_") && !tensor.tensor.contains("ffn_")));
    }
}
