use std::path::PathBuf;

use larql_inference::attention::load_deepseek2_attention_manifest_plan;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vindex_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: deepseek2_attention_manifest_plan <vindex-dir>")?;
    let plan = load_deepseek2_attention_manifest_plan(&vindex_dir)?;
    println!(
        "status=ok architecture={} residency={} layers={} complete_mla_layers={} cuda_residency_tensors={}",
        plan.architecture,
        plan.residency,
        plan.layers.len(),
        plan.complete_mla_layers(),
        plan.tensors_for_cuda_residency().len()
    );
    if let Some(first) = plan.layers.first() {
        println!(
            "first_layer={} complete_mla={} q_a={} kv_a_mqa={} output={}",
            first.layer,
            first.is_mla_complete(),
            first
                .q_a
                .as_ref()
                .map(|tensor| tensor.tensor.as_str())
                .unwrap_or("<missing>"),
            first
                .kv_a_mqa
                .as_ref()
                .map(|tensor| tensor.tensor.as_str())
                .unwrap_or("<missing>"),
            first
                .output
                .as_ref()
                .map(|tensor| tensor.tensor.as_str())
                .unwrap_or("<missing>")
        );
    }
    Ok(())
}
