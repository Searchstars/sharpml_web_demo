use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use sharp_mcu_core::{
    DEFAULT_ARIA2_CONNECTIONS, DEFAULT_MODEL_REVISION, GenerateConfig, GenerateRequest,
    InferenceBackend, InferenceConfig, ModelDownloadBackend, default_coreml_cache_dir,
    default_model_cache_dir, default_model_file, default_model_repo, generate_package,
};

fn default_layer_count() -> usize {
    GenerateConfig::default().layer_count
}

#[derive(Parser)]
#[command(name = "sharp-mcu")]
#[command(about = "Rust SHARP preprocessing and MCU slice packaging")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

fn default_cli_inference_backend() -> InferenceBackendArg {
    if cfg!(target_vendor = "apple") {
        InferenceBackendArg::CoremlNative
    } else {
        InferenceBackendArg::Cpu
    }
}

#[derive(Subcommand)]
enum Command {
    Generate {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        hf_endpoint: Option<String>,
        #[arg(long, value_enum, default_value_t = DownloadBackendArg::HfHub)]
        download_backend: DownloadBackendArg,
        #[arg(long, default_value_t = DEFAULT_ARIA2_CONNECTIONS)]
        aria2_connections: usize,
        #[arg(long, value_enum, default_value_t = default_cli_inference_backend())]
        inference_backend: InferenceBackendArg,
        #[arg(long)]
        ort_intra_threads: Option<usize>,
        #[arg(long)]
        ort_inter_threads: Option<usize>,
        #[arg(long, default_value_t = 0)]
        gpu_device_id: i32,
        #[arg(long)]
        cuda_root_dir: Option<PathBuf>,
        #[arg(long)]
        cudnn_root_dir: Option<PathBuf>,
        #[arg(long, default_value_os_t = default_coreml_cache_dir())]
        coreml_cache_dir: PathBuf,
        #[arg(long, default_value_os_t = default_model_cache_dir())]
        cache_dir: PathBuf,
        #[arg(long, default_value_t = default_model_repo())]
        repo: String,
        #[arg(long, default_value = DEFAULT_MODEL_REVISION)]
        revision: String,
        #[arg(long, alias = "onnx-file", default_value_t = default_model_file())]
        model_file: String,
        #[arg(long, alias = "onnx-data-file")]
        model_data_file: Option<String>,
        #[arg(long, alias = "layer-count", default_value_t = default_layer_count())]
        layers: usize,
        #[arg(long, default_value_t = 2560)]
        preview_max_edge: u32,
        #[arg(long, default_value_t = 2560 * 1440)]
        preview_target_pixels: u64,
        #[arg(long, default_value_t = 1.0)]
        disparity_factor: f32,
        #[arg(long, default_value_t = 1.0)]
        depth_scale: f32,
        #[arg(long, default_value_t = true)]
        emit_debug_ply: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DownloadBackendArg {
    HfHub,
    Aria2,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InferenceBackendArg {
    Auto,
    Cpu,
    Cuda,
    Migraphx,
    CoremlNative,
    Coreml,
    CoremlNeuralEngine,
    CoremlGpu,
    CoremlCpu,
}

impl From<DownloadBackendArg> for ModelDownloadBackend {
    fn from(value: DownloadBackendArg) -> Self {
        match value {
            DownloadBackendArg::HfHub => ModelDownloadBackend::HfHub,
            DownloadBackendArg::Aria2 => ModelDownloadBackend::Aria2,
        }
    }
}

impl From<InferenceBackendArg> for InferenceBackend {
    fn from(value: InferenceBackendArg) -> Self {
        match value {
            InferenceBackendArg::Auto => InferenceBackend::Auto,
            InferenceBackendArg::Cpu => InferenceBackend::Cpu,
            InferenceBackendArg::Cuda => InferenceBackend::Cuda,
            InferenceBackendArg::Migraphx => InferenceBackend::Migraphx,
            InferenceBackendArg::CoremlNative => InferenceBackend::CoreMlNative,
            InferenceBackendArg::Coreml => InferenceBackend::CoreMl,
            InferenceBackendArg::CoremlNeuralEngine => InferenceBackend::CoreMlNeuralEngine,
            InferenceBackendArg::CoremlGpu => InferenceBackend::CoreMlGpu,
            InferenceBackendArg::CoremlCpu => InferenceBackend::CoreMlCpu,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Generate {
            input,
            output_dir,
            model,
            hf_endpoint,
            download_backend,
            aria2_connections,
            inference_backend,
            ort_intra_threads,
            ort_inter_threads,
            gpu_device_id,
            cuda_root_dir,
            cudnn_root_dir,
            coreml_cache_dir,
            cache_dir,
            repo,
            revision,
            model_file,
            model_data_file,
            layers,
            preview_max_edge,
            preview_target_pixels,
            disparity_factor,
            depth_scale,
            emit_debug_ply,
        } => {
            let config = GenerateConfig {
                layer_count: layers,
                preview_max_edge,
                preview_target_pixels,
                disparity_factor,
                depth_scale,
                emit_debug_ply,
                ..GenerateConfig::default()
            };
            let request = GenerateRequest {
                input_path: input,
                output_dir,
                model_path: model,
                model_endpoint: hf_endpoint
                    .or_else(|| std::env::var("HF_ENDPOINT").ok())
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty()),
                download_backend: download_backend.into(),
                aria2_connections,
                inference: InferenceConfig {
                    backend: inference_backend.into(),
                    intra_threads: ort_intra_threads,
                    inter_threads: ort_inter_threads,
                    gpu_device_id,
                    cuda_root_dir,
                    cudnn_root_dir,
                    coreml_cache_dir: Some(coreml_cache_dir),
                },
                model_cache_dir: Some(cache_dir),
                model_repo: repo,
                model_revision: revision,
                model_file,
                model_data_file,
                config,
            };
            let report = generate_package(&request)?;
            println!("manifest: {}", report.manifest_path.display());
            if let Some(path) = report.debug_ply_path {
                println!("debug_ply: {}", path.display());
            }
            println!(
                "package: {}x{} · {} layers",
                report.manifest.package.width,
                report.manifest.package.height,
                report.manifest.package.layer_count
            );
        }
    }
    Ok(())
}
