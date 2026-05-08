//! Compact GGUF attention-manifest planning for split-residency Kimi/DeepSeek2.
//!
//! This module intentionally plans dense attention/MLA tensors separately from
//! mmap-backed MoE expert tensors. It does not materialize tensor data yet; the
//! next loader stage can consume the plan to decide what may be staged on CUDA.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use larql_models::loading::gguf::{GgufFile, LoadedGgufTensor};
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

    pub fn layer(&self, layer: usize) -> Option<&Deepseek2AttentionLayerPlan> {
        self.layers.iter().find(|plan| plan.layer == layer)
    }
}

/// Dequantized tensor payloads for one DeepSeek2/Kimi MLA attention layer.
#[derive(Debug)]
pub struct Deepseek2AttentionLayerTensors {
    pub layer: usize,
    pub tensors: HashMap<String, LoadedGgufTensor>,
}

impl Deepseek2AttentionLayerTensors {
    pub fn get(&self, tensor_name: &str) -> Option<&LoadedGgufTensor> {
        self.tensors.get(tensor_name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

pub fn load_deepseek2_attention_layer_tensors(
    vindex_dir: &Path,
    layer: usize,
) -> Result<Deepseek2AttentionLayerTensors, InferenceError> {
    let plan = load_deepseek2_attention_manifest_plan(vindex_dir)?;
    let layer_plan = plan.layer(layer).ok_or_else(|| {
        InferenceError::MissingTensor(format!("DeepSeek2 attention layer {layer}"))
    })?;
    load_deepseek2_attention_layer_tensors_from_plan(vindex_dir, layer_plan)
}

fn load_deepseek2_attention_layer_tensors_from_plan(
    vindex_dir: &Path,
    layer_plan: &Deepseek2AttentionLayerPlan,
) -> Result<Deepseek2AttentionLayerTensors, InferenceError> {
    if !layer_plan.is_mla_complete() {
        return Err(InferenceError::MissingTensor(format!(
            "DeepSeek2 layer {} complete MLA attention tensor set",
            layer_plan.layer
        )));
    }

    let mut by_source: BTreeMap<PathBuf, Vec<&GgufAttentionTensorRef>> = BTreeMap::new();
    for tensor_ref in layer_plan.tensors_for_cuda_residency() {
        let source = PathBuf::from(&tensor_ref.source_file);
        let source = if source.is_absolute() {
            source
        } else {
            vindex_dir.join(source)
        };
        by_source.entry(source).or_default().push(tensor_ref);
    }

    let mut tensors = HashMap::new();
    for (source, refs) in by_source {
        let gguf = GgufFile::open(&source)?;
        for tensor_ref in refs {
            let loaded = gguf.load_tensor_data_by_name(&tensor_ref.tensor)?;
            if loaded
                .dims
                .iter()
                .map(|dim| *dim as usize)
                .collect::<Vec<_>>()
                != tensor_ref.dims
            {
                return Err(InferenceError::Parse(format!(
                    "GGUF tensor {} dims {:?} do not match manifest dims {:?}",
                    tensor_ref.tensor, loaded.dims, tensor_ref.dims
                )));
            }
            tensors.insert(tensor_ref.tensor.clone(), loaded);
        }
    }

    Ok(Deepseek2AttentionLayerTensors {
        layer: layer_plan.layer,
        tensors,
    })
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

    #[test]
    fn deepseek2_attention_manifest_loader_reads_one_layer_tensor_payloads() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let shard_path = dir.path().join("tiny-attn.gguf");
        let tensor_names = [
            "blk.0.attn_q_a.weight",
            "blk.0.attn_q_a_norm.weight",
            "blk.0.attn_q_b.weight",
            "blk.0.attn_kv_a_mqa.weight",
            "blk.0.attn_kv_a_norm.weight",
            "blk.0.attn_k_b.weight",
            "blk.0.attn_v_b.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate_exps.weight",
        ];
        let mut file = std::fs::File::create(&shard_path).unwrap();
        file.write_all(&0x46554747u32.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&(tensor_names.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();

        let mut offsets = Vec::new();
        let mut running_offset = 0u64;
        for name in tensor_names {
            file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
            file.write_all(name.as_bytes()).unwrap();
            file.write_all(&2u32.to_le_bytes()).unwrap();
            file.write_all(&2u64.to_le_bytes()).unwrap();
            file.write_all(&2u64.to_le_bytes()).unwrap();
            file.write_all(&larql_models::quant::ggml::TYPE_F32.to_le_bytes())
                .unwrap();
            file.write_all(&running_offset.to_le_bytes()).unwrap();
            offsets.push(running_offset);
            running_offset += 16;
        }

        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        for idx in 0..tensor_names.len() {
            for v in 0..4u32 {
                file.write_all(&((idx as f32) + (v as f32 / 10.0)).to_le_bytes())
                    .unwrap();
            }
        }
        file.flush().unwrap();

        let manifest_entries = tensor_names
            .iter()
            .zip(offsets.iter())
            .map(|(name, offset)| {
                format!(
                    r#"{{"tensor":"{name}","source_file":"{}","shard_idx":0,"tensor_type":0,"dims":[2,2],"tensor_offset":{offset},"data_offset":0}}"#,
                    shard_path.display()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let manifest = format!(
            r#"{{"version":1,"architecture":"deepseek2","split_count":1,"residency":"split_residency_cuda_attention_mmap_experts","tensors":[{manifest_entries}]}}"#
        );
        fs::write(dir.path().join("gguf_attention_manifest.json"), manifest).unwrap();

        let loaded = load_deepseek2_attention_layer_tensors(dir.path(), 0).unwrap();
        assert_eq!(loaded.layer, 0);
        assert_eq!(loaded.len(), 8);
        assert!(loaded.get("blk.0.ffn_gate_exps.weight").is_none());
        let q_a = loaded.get("blk.0.attn_q_a.weight").unwrap();
        assert_eq!(q_a.dims, vec![2, 2]);
        assert_eq!(q_a.values, vec![0.0, 0.1, 0.2, 0.3]);
        let output = loaded.get("blk.0.attn_output.weight").unwrap();
        assert_eq!(output.values[0], 7.0);
    }
}
