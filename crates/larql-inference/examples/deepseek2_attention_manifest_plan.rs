use std::path::PathBuf;

#[cfg(all(feature = "cuda", target_os = "linux"))]
use larql_inference::attention::load_deepseek2_cuda_resident_2d_attention;
use larql_inference::attention::{
    deepseek2_mla_single_token_cpu, load_deepseek2_attention_layer_tensors,
    load_deepseek2_attention_manifest_plan,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let vindex_dir = args.next().map(PathBuf::from).ok_or(
        "usage: deepseek2_attention_manifest_plan <vindex-dir> [--load-layer N] [--mla-smoke]",
    )?;
    let mut load_layer = None;
    let mut mla_smoke = false;
    let mut cuda_resident_2d = false;
    while let Some(arg) = args.next() {
        if arg == "--load-layer" {
            let layer = args
                .next()
                .ok_or("--load-layer requires a layer index")?
                .to_string_lossy()
                .parse::<usize>()?;
            load_layer = Some(layer);
        } else if arg == "--mla-smoke" {
            mla_smoke = true;
        } else if arg == "--cuda-resident-2d" {
            cuda_resident_2d = true;
        }
    }

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

    if let Some(layer) = load_layer {
        let loaded = load_deepseek2_attention_layer_tensors(&vindex_dir, layer)?;
        let total_values: usize = loaded
            .tensors
            .values()
            .map(|tensor| tensor.values.len())
            .sum();
        let total_bytes = total_values * std::mem::size_of::<f32>();
        println!(
            "loaded_layer={} tensors={} f32_values={} f32_bytes={} contains_output={}",
            loaded.layer,
            loaded.len(),
            total_values,
            total_bytes,
            loaded
                .get(&format!("blk.{}.attn_output.weight", loaded.layer))
                .is_some()
        );
        if mla_smoke {
            let q_a = loaded
                .get(&format!("blk.{}.attn_q_a.weight", loaded.layer))
                .ok_or("loaded layer missing q_a")?;
            let hidden = q_a.dims[0] as usize;
            let mut input = vec![0.0f32; hidden];
            if let Some(first) = input.first_mut() {
                *first = 1.0;
            }
            let started = std::time::Instant::now();
            let out = deepseek2_mla_single_token_cpu(&loaded, &input)?;
            let elapsed = started.elapsed();
            let max_abs = out.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
            println!(
                "mla_smoke=ok layer={} input_hidden={} output_hidden={} elapsed_ms={:.3} output_max_abs={:.6}",
                loaded.layer,
                hidden,
                out.len(),
                elapsed.as_secs_f64() * 1000.0,
                max_abs
            );
        }
        if cuda_resident_2d {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            {
                let cuda =
                    larql_compute::cuda::CudaBackend::new().ok_or("CUDA backend unavailable")?;
                let started = std::time::Instant::now();
                let resident = load_deepseek2_cuda_resident_2d_attention(&cuda, &loaded)?;
                let elapsed = started.elapsed();
                println!(
                    "cuda_resident_2d=ok layer={} matrices={} elapsed_ms={:.3}",
                    resident.layer,
                    resident.resident_matrix_count(),
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            {
                return Err("--cuda-resident-2d requires --features cuda on Linux".into());
            }
        }
    }
    Ok(())
}
