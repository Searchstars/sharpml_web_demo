use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, anyhow, bail};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use image::{DynamicImage, GenericImageView, RgbaImage, imageops::FilterType};
use ndarray::{Array1, Array2, Array4, Axis, Ix3};
use onnx_protobuf::{
    AttributeProto, GraphProto, ModelProto, SparseTensorProto, StringStringEntryProto, TensorProto,
};
use ort::{
    ep::{
        CPU, CUDA, CoreML, ExecutionProviderDispatch, MIGraphX,
        coreml::{ComputeUnits, SpecializationStrategy},
        cuda::ConvAlgorithmSearch,
    },
    logging::LogLevel,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use protobuf::Message;

use crate::coreml_native::{DEFAULT_APPLE_MODEL_FILE, DEFAULT_APPLE_MODEL_REPO};

pub const MODEL_INPUT_WIDTH: u32 = 1536;
pub const MODEL_INPUT_HEIGHT: u32 = 1536;
pub const DEFAULT_ONNX_MODEL_REPO: &str = "pearsonkyle/Sharp-onnx";
pub const DEFAULT_MODEL_REVISION: &str = "main";
pub const DEFAULT_ONNX_MODEL_FILE: &str = "sharp_fp16.onnx";
pub const DEFAULT_MODEL_CACHE_DIR: &str = "caches/hf-hub";
pub const DEFAULT_COREML_CACHE_DIR: &str = "caches/coreml";
pub const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";
pub const DEFAULT_ARIA2_CONNECTIONS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDownloadBackend {
    HfHub,
    Aria2,
}

impl ModelDownloadBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HfHub => "hf-hub",
            Self::Aria2 => "aria2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceBackend {
    Auto,
    Cpu,
    Cuda,
    Migraphx,
    CoreMlNative,
    CoreMl,
    CoreMlNeuralEngine,
    CoreMlGpu,
    CoreMlCpu,
}

impl InferenceBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Migraphx => "migraphx",
            Self::CoreMlNative => "coreml-native",
            Self::CoreMl => "coreml",
            Self::CoreMlNeuralEngine => "coreml-neural-engine",
            Self::CoreMlGpu => "coreml-gpu",
            Self::CoreMlCpu => "coreml-cpu",
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub backend: InferenceBackend,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub gpu_device_id: i32,
    pub cuda_root_dir: Option<PathBuf>,
    pub cudnn_root_dir: Option<PathBuf>,
    pub coreml_cache_dir: Option<PathBuf>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: default_inference_backend(),
            intra_threads: None,
            inter_threads: None,
            gpu_device_id: 0,
            cuda_root_dir: None,
            cudnn_root_dir: None,
            coreml_cache_dir: Some(default_coreml_cache_dir()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Onnx,
    CoreMlPackage,
}

impl ModelFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::CoreMlPackage => "coreml-package",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub input_path: PathBuf,
    pub original_rgba: RgbaImage,
    pub input_tensor: Array4<f32>,
    pub original_width: u32,
    pub original_height: u32,
    pub focal_length_px: f32,
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub format: ModelFormat,
    pub repo: String,
    pub revision: String,
    pub endpoint: Option<String>,
    pub download_backend: ModelDownloadBackend,
    pub cache_dir: Option<PathBuf>,
    pub model_path: PathBuf,
    pub model_data_path: Option<PathBuf>,
}

pub fn default_model_repo() -> String {
    if cfg!(target_vendor = "apple") {
        DEFAULT_APPLE_MODEL_REPO.to_string()
    } else {
        DEFAULT_ONNX_MODEL_REPO.to_string()
    }
}

pub fn default_model_file() -> String {
    if cfg!(target_vendor = "apple") {
        DEFAULT_APPLE_MODEL_FILE.to_string()
    } else {
        DEFAULT_ONNX_MODEL_FILE.to_string()
    }
}

pub fn default_model_cache_dir() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or(crate_dir.as_path())
        .join(DEFAULT_MODEL_CACHE_DIR)
}

pub fn default_coreml_cache_dir() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or(crate_dir.as_path())
        .join(DEFAULT_COREML_CACHE_DIR)
}

pub fn default_inference_backend() -> InferenceBackend {
    if cfg!(target_vendor = "apple") {
        InferenceBackend::CoreMlNative
    } else {
        InferenceBackend::Cpu
    }
}

#[derive(Debug, Clone)]
pub struct SharpOutputs {
    pub mean_vectors: Array2<f32>,
    pub singular_values: Array2<f32>,
    pub quaternions: Array2<f32>,
    pub colors: Array2<f32>,
    pub opacities: Array1<f32>,
    pub output_names: Vec<String>,
}

pub struct SharpOnnx {
    session: Session,
}

pub fn prepare_image(path: &Path) -> Result<PreparedImage> {
    let image =
        image::open(path).with_context(|| format!("failed to open image {}", path.display()))?;
    let original_rgba = image.to_rgba8();
    let (original_width, original_height) = image.dimensions();
    let focal_length_px = original_width as f32;
    let input_tensor = image_to_tensor(&image)?;
    Ok(PreparedImage {
        input_path: path.to_path_buf(),
        original_rgba,
        input_tensor,
        original_width,
        original_height,
        focal_length_px,
    })
}

pub fn resolve_model(
    explicit_model: Option<&Path>,
    endpoint: Option<&str>,
    cache_dir: Option<&Path>,
    download_backend: ModelDownloadBackend,
    aria2_connections: usize,
    repo: &str,
    revision: &str,
    model_file: &str,
    model_data_file: Option<&str>,
) -> Result<ResolvedModel> {
    let endpoint = resolve_hf_endpoint(endpoint);
    if let Some(path) = explicit_model {
        if !path.exists() {
            bail!("model path {} does not exist", path.display());
        }
        let format = detect_model_format(path)?;
        let data_path = match format {
            ModelFormat::Onnx => resolve_local_model_data_path(path, model_data_file),
            ModelFormat::CoreMlPackage => None,
        };
        return Ok(ResolvedModel {
            format,
            repo: repo.to_string(),
            revision: revision.to_string(),
            endpoint,
            download_backend,
            cache_dir: cache_dir.map(Path::to_path_buf),
            model_path: path.to_path_buf(),
            model_data_path: data_path,
        });
    }

    let format = detect_model_format(Path::new(model_file))?;
    let cache_dir = cache_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(default_model_cache_dir);
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("failed to create model cache dir {}", cache_dir.display()))?;
    let (model_path, model_data_path) = match (format, download_backend) {
        (ModelFormat::Onnx, ModelDownloadBackend::HfHub) => download_onnx_via_hf_hub(
            endpoint.as_deref(),
            &cache_dir,
            repo,
            revision,
            model_file,
            model_data_file,
        )?,
        (ModelFormat::Onnx, ModelDownloadBackend::Aria2) => {
            let endpoint = endpoint
                .clone()
                .unwrap_or_else(|| "https://huggingface.co".to_string());
            download_onnx_via_aria2(
                &endpoint,
                &cache_dir,
                repo,
                revision,
                model_file,
                model_data_file,
                aria2_connections,
            )?
        }
        (ModelFormat::CoreMlPackage, ModelDownloadBackend::HfHub) => (
            download_coreml_package_via_hf_hub(
                endpoint.as_deref(),
                &cache_dir,
                repo,
                revision,
                model_file,
            )?,
            None,
        ),
        (ModelFormat::CoreMlPackage, ModelDownloadBackend::Aria2) => {
            let endpoint = endpoint
                .clone()
                .unwrap_or_else(|| "https://huggingface.co".to_string());
            (
                download_coreml_package_via_aria2(
                    &endpoint,
                    &cache_dir,
                    repo,
                    revision,
                    model_file,
                    aria2_connections,
                )?,
                None,
            )
        }
    };
    Ok(ResolvedModel {
        format,
        repo: repo.to_string(),
        revision: revision.to_string(),
        endpoint,
        download_backend,
        cache_dir: Some(cache_dir),
        model_path,
        model_data_path,
    })
}

fn detect_model_format(path: &Path) -> Result<ModelFormat> {
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "onnx" => Ok(ModelFormat::Onnx),
        "mlpackage" | "mlmodel" | "mlmodelc" => Ok(ModelFormat::CoreMlPackage),
        _ => bail!(
            "unsupported model file {} (expected .onnx, .mlpackage, .mlmodel, or .mlmodelc)",
            path.display()
        ),
    }
}

fn resolve_local_model_data_path(model_path: &Path, explicit: Option<&str>) -> Option<PathBuf> {
    if let Some(explicit) = explicit {
        let explicit_path = PathBuf::from(explicit);
        return Some(if explicit_path.is_absolute() {
            explicit_path
        } else {
            model_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(explicit_path)
        });
    }

    let sibling = model_path.with_file_name(format!(
        "{}.data",
        model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.onnx")
    ));
    sibling.exists().then_some(sibling)
}

fn resolve_hf_endpoint(explicit: Option<&str>) -> Option<String> {
    explicit
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            std::env::var(HF_ENDPOINT_ENV)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

fn download_onnx_via_hf_hub(
    endpoint: Option<&str>,
    cache_dir: &Path,
    repo: &str,
    revision: &str,
    model_file: &str,
    model_data_file: Option<&str>,
) -> Result<(PathBuf, Option<PathBuf>)> {
    let api = build_hf_api(endpoint, cache_dir)?;
    let hf_repo = Repo::with_revision(repo.to_string(), RepoType::Model, revision.to_string());
    let model_api = api.repo(hf_repo);
    let model_path = model_api
        .get(model_file)
        .with_context(|| format!("failed to download {repo}/{model_file}"))?;
    let model_data_path = model_data_file
        .map(|file| {
            model_api
                .get(file)
                .with_context(|| format!("failed to download {repo}/{file}"))
        })
        .transpose()?;
    Ok((model_path, model_data_path))
}

fn download_onnx_via_aria2(
    endpoint: &str,
    cache_dir: &Path,
    repo: &str,
    revision: &str,
    model_file: &str,
    model_data_file: Option<&str>,
    connections: usize,
) -> Result<(PathBuf, Option<PathBuf>)> {
    let repo_dir = cache_dir.join("aria2").join(repo).join(revision);
    fs::create_dir_all(&repo_dir)
        .with_context(|| format!("failed to create aria2 cache dir {}", repo_dir.display()))?;

    let model_path = repo_dir.join(model_file);
    let model_data_path = model_data_file.map(|file| repo_dir.join(file));

    if !model_path.exists() {
        let url = repo_file_url(endpoint, repo, revision, model_file);
        download_file_via_aria2(&url, &model_path, connections)?;
    }
    if let (Some(file), Some(path)) = (model_data_file, model_data_path.as_ref()) {
        if !path.exists() {
            let url = repo_file_url(endpoint, repo, revision, file);
            download_file_via_aria2(&url, path, connections)?;
        }
    }

    Ok((model_path, model_data_path))
}

fn download_coreml_package_via_hf_hub(
    endpoint: Option<&str>,
    cache_dir: &Path,
    repo: &str,
    revision: &str,
    package_dir: &str,
) -> Result<PathBuf> {
    let api = build_hf_api(endpoint, cache_dir)?;
    let hf_repo = Repo::with_revision(repo.to_string(), RepoType::Model, revision.to_string());
    let model_api = api.repo(hf_repo);
    let package_files = repo_package_files(&model_api.info()?, package_dir)?;
    let mut first_local = None;
    for remote_path in &package_files {
        let local_path = model_api
            .get(remote_path)
            .with_context(|| format!("failed to download {repo}/{remote_path}"))?;
        if first_local.is_none() {
            first_local = Some((local_path, remote_path.clone()));
        }
    }
    let (local_path, remote_path) =
        first_local.context("coreml package download resolved no files")?;
    local_root_for_remote_subpath(&local_path, &remote_path, package_dir)
}

fn download_coreml_package_via_aria2(
    endpoint: &str,
    cache_dir: &Path,
    repo: &str,
    revision: &str,
    package_dir: &str,
    connections: usize,
) -> Result<PathBuf> {
    let repo_dir = cache_dir.join("aria2").join(repo).join(revision);
    fs::create_dir_all(&repo_dir)
        .with_context(|| format!("failed to create aria2 cache dir {}", repo_dir.display()))?;
    let files = repo_package_files(
        &repo_info(endpoint, &repo_dir, repo, revision)?,
        package_dir,
    )?;
    for remote_path in &files {
        let destination = repo_dir.join(remote_path);
        if destination.exists() {
            continue;
        }
        let url = repo_file_url(endpoint, repo, revision, remote_path);
        download_file_via_aria2(&url, &destination, connections)?;
    }
    let package_root = repo_dir.join(package_dir);
    if !package_root.exists() {
        bail!(
            "downloaded CoreML package files for {repo}/{package_dir}, but {} was not created",
            package_root.display()
        );
    }
    Ok(package_root)
}

fn build_hf_api(endpoint: Option<&str>, cache_dir: &Path) -> Result<hf_hub::api::sync::Api> {
    ApiBuilder::new()
        .with_progress(true)
        .with_cache_dir(cache_dir.to_path_buf())
        .with_endpoint(endpoint.unwrap_or("https://huggingface.co").to_string())
        .build()
        .context("failed to initialize Hugging Face sync API")
}

fn repo_info(
    endpoint: &str,
    cache_dir: &Path,
    repo: &str,
    revision: &str,
) -> Result<hf_hub::api::RepoInfo> {
    let api = build_hf_api(Some(endpoint), cache_dir)?;
    let hf_repo = Repo::with_revision(repo.to_string(), RepoType::Model, revision.to_string());
    api.repo(hf_repo)
        .info()
        .with_context(|| format!("failed to query repo info for {repo}@{revision}"))
}

fn repo_package_files(repo_info: &hf_hub::api::RepoInfo, package_dir: &str) -> Result<Vec<String>> {
    let prefix = format!("{}/", package_dir.trim_end_matches('/'));
    let mut files = repo_info
        .siblings
        .iter()
        .filter_map(|sibling| {
            let name = sibling.rfilename.as_str();
            (name == package_dir || name.starts_with(&prefix)).then(|| name.to_string())
        })
        .collect::<Vec<_>>();
    files.sort();
    files.dedup();
    if files.is_empty() {
        bail!("repo did not contain any files under {}", package_dir);
    }
    Ok(files)
}

fn local_root_for_remote_subpath(
    local_path: &Path,
    remote_path: &str,
    root_remote_path: &str,
) -> Result<PathBuf> {
    let relative = Path::new(remote_path)
        .strip_prefix(root_remote_path)
        .with_context(|| format!("{remote_path} was not under expected root {root_remote_path}"))?;
    let mut root = local_path.to_path_buf();
    for _ in relative.components() {
        root = root
            .parent()
            .context("downloaded package path had no parent while resolving root")?
            .to_path_buf();
    }
    Ok(root)
}

fn download_file_via_aria2(url: &str, destination: &Path, connections: usize) -> Result<()> {
    let program = "aria2c";
    let output_dir = destination
        .parent()
        .context("aria2 destination had no parent directory")?;
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create aria2 output dir {}", output_dir.display()))?;
    let output_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .context("aria2 destination had invalid filename")?;
    let connections = connections.max(1);
    let status = Command::new(program)
        .arg("-c")
        .arg("--file-allocation=none")
        .arg("--auto-file-renaming=false")
        .arg("--dir")
        .arg(output_dir)
        .arg("--out")
        .arg(output_name)
        .arg("-x")
        .arg(connections.to_string())
        .arg("-s")
        .arg(connections.to_string())
        .arg(url)
        .status()
        .with_context(|| {
            format!(
                "failed to spawn {program}; install aria2 first if you want multi-connection downloads"
            )
        })?;
    if !status.success() {
        bail!(
            "{program} failed while downloading {url} into {}",
            output_dir.display()
        );
    }
    Ok(())
}

fn repo_file_url(endpoint: &str, repo: &str, revision: &str, filename: &str) -> String {
    let endpoint = endpoint.trim_end_matches('/');
    format!("{endpoint}/{repo}/resolve/{revision}/{filename}")
}

fn model_uses_external_tensor_data(model_path: &Path) -> Result<bool> {
    let bytes = fs::read(model_path)
        .with_context(|| format!("failed to read model {}", model_path.display()))?;
    let model = ModelProto::parse_from_bytes(&bytes)
        .with_context(|| format!("failed to parse ONNX model {}", model_path.display()))?;
    let mut locations = BTreeSet::<String>::new();
    if let Some(graph) = model.graph.as_ref() {
        collect_external_locations_from_graph(graph, &mut locations);
    }
    Ok(!locations.is_empty())
}

fn collect_external_locations_from_graph(graph: &GraphProto, out: &mut BTreeSet<String>) {
    for tensor in &graph.initializer {
        collect_external_location_from_tensor(tensor, out);
    }
    for sparse in &graph.sparse_initializer {
        collect_external_locations_from_sparse_tensor(sparse, out);
    }
    for node in &graph.node {
        for attr in &node.attribute {
            collect_external_locations_from_attribute(attr, out);
        }
    }
}

fn collect_external_locations_from_attribute(attr: &AttributeProto, out: &mut BTreeSet<String>) {
    if let Some(tensor) = attr.t.as_ref() {
        collect_external_location_from_tensor(tensor, out);
    }
    if let Some(graph) = attr.g.as_ref() {
        collect_external_locations_from_graph(graph, out);
    }
    if let Some(sparse) = attr.sparse_tensor.as_ref() {
        collect_external_locations_from_sparse_tensor(sparse, out);
    }
    for tensor in &attr.tensors {
        collect_external_location_from_tensor(tensor, out);
    }
    for graph in &attr.graphs {
        collect_external_locations_from_graph(graph, out);
    }
    for sparse in &attr.sparse_tensors {
        collect_external_locations_from_sparse_tensor(sparse, out);
    }
}

fn collect_external_locations_from_sparse_tensor(
    sparse: &SparseTensorProto,
    out: &mut BTreeSet<String>,
) {
    if let Some(values) = sparse.values.as_ref() {
        collect_external_location_from_tensor(values, out);
    }
    if let Some(indices) = sparse.indices.as_ref() {
        collect_external_location_from_tensor(indices, out);
    }
}

fn collect_external_location_from_tensor(tensor: &TensorProto, out: &mut BTreeSet<String>) {
    if let Some(location) = external_data_location(&tensor.external_data) {
        out.insert(location);
    }
}

fn external_data_location(entries: &[StringStringEntryProto]) -> Option<String> {
    entries
        .iter()
        .find(|entry| entry.key == "location")
        .map(|entry| entry.value.clone())
}

impl SharpOnnx {
    pub fn new(model: &ResolvedModel, config: &InferenceConfig) -> Result<Self> {
        if model.format != ModelFormat::Onnx {
            bail!(
                "SharpOnnx expected an ONNX model, got {} at {}",
                model.format.as_str(),
                model.model_path.display()
            );
        }
        if matches!(config.backend, InferenceBackend::CoreMlNative) {
            bail!("coreml-native backend requires a .mlpackage/.mlmodel asset, not ONNX");
        }

        let uses_external_data = model_uses_external_tensor_data(&model.model_path)?;
        if uses_external_data {
            match config.backend {
                InferenceBackend::Auto => {
                    if cfg!(target_vendor = "apple") {
                        eprintln!(
                            "warning: skipping CoreML for {} because the model uses external tensor data (.onnx.data); falling back to CPU to avoid extreme memory use during CoreML compilation",
                            model.model_path.display()
                        );
                    }
                }
                InferenceBackend::CoreMl
                | InferenceBackend::CoreMlNeuralEngine
                | InferenceBackend::CoreMlGpu
                | InferenceBackend::CoreMlCpu => {
                    bail!(
                        "CoreML inference is currently disabled for ONNX models that use external tensor data (.onnx.data), including {}. \
The required ONNX Runtime workaround can balloon RAM usage into tens of GiB during CoreML subgraph compilation. \
Use --inference-backend cpu for now, or switch to a single-file/native CoreML model.",
                        model.model_path.display()
                    );
                }
                InferenceBackend::Cpu
                | InferenceBackend::Cuda
                | InferenceBackend::Migraphx
                | InferenceBackend::CoreMlNative => {}
            }
        }

        maybe_preload_cuda_dependencies(config)?;

        let mut builder = Session::builder()
            .map_err(|err| anyhow!(err.to_string()))?
            .with_log_level(LogLevel::Error)
            .map_err(|err| anyhow!(err.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|err| anyhow!(err.to_string()))?;
        if let Some(threads) = config.intra_threads {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|err| anyhow!(err.to_string()))?;
        }
        if let Some(threads) = config.inter_threads {
            builder = builder
                .with_inter_threads(threads)
                .map_err(|err| anyhow!(err.to_string()))?;
        }
        let execution_providers = execution_providers_for_config(config, !uses_external_data);
        builder = builder
            .with_execution_providers(execution_providers)
            .map_err(|err| anyhow!(err.to_string()))?;
        let session = builder
            .commit_from_file(&model.model_path)
            .map_err(|err| anyhow!(err.to_string()))
            .with_context(|| format!("failed to load ONNX model {}", model.model_path.display()))?;
        Ok(Self { session })
    }

    pub fn infer(&mut self, image: &PreparedImage, disparity_factor: f32) -> Result<SharpOutputs> {
        let disparity = ndarray::arr1(&[disparity_factor]);
        let output_names = self
            .session
            .outputs()
            .iter()
            .map(|out| out.name().to_string())
            .collect::<Vec<_>>();
        let outputs = self
            .session
            .run(ort::inputs![
                "image" => TensorRef::from_array_view(image.input_tensor.view())?,
                "disparity_factor" => TensorRef::from_array_view(disparity.view())?
            ])
            .map_err(|err| anyhow!(err.to_string()))?;

        if outputs.len() == 1 {
            let concat = outputs[0]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix3>()
                .context("expected concat output to have rank 3")?;
            let batch = concat.index_axis(Axis(0), 0);
            if batch.len_of(Axis(1)) < 14 {
                bail!(
                    "concat output has {} channels, expected at least 14",
                    batch.len_of(Axis(1))
                );
            }
            return Ok(SharpOutputs {
                mean_vectors: batch.slice(ndarray::s![.., 0..3]).to_owned(),
                singular_values: batch.slice(ndarray::s![.., 3..6]).to_owned(),
                quaternions: batch.slice(ndarray::s![.., 6..10]).to_owned(),
                colors: batch.slice(ndarray::s![.., 10..13]).to_owned(),
                opacities: batch.slice(ndarray::s![.., 13]).to_owned(),
                output_names,
            });
        }

        if outputs.len() == 5 {
            let mean_vectors = outputs[0]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix3>()?
                .index_axis(Axis(0), 0)
                .to_owned();
            let singular_values = outputs[1]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix3>()?
                .index_axis(Axis(0), 0)
                .to_owned();
            let quaternions = outputs[2]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix3>()?
                .index_axis(Axis(0), 0)
                .to_owned();
            let colors = outputs[3]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix3>()?
                .index_axis(Axis(0), 0)
                .to_owned();
            let opacities = outputs[4]
                .try_extract_array::<f32>()?
                .into_dimensionality::<ndarray::Ix2>()?
                .index_axis(Axis(0), 0)
                .to_owned();
            return Ok(SharpOutputs {
                mean_vectors,
                singular_values,
                quaternions,
                colors,
                opacities,
                output_names,
            });
        }

        Err(anyhow!(
            "unsupported ONNX output layout: got {} tensors ({output_names:?})",
            outputs.len()
        ))
    }
}

fn execution_providers_for_config(
    config: &InferenceConfig,
    allow_coreml: bool,
) -> Vec<ExecutionProviderDispatch> {
    let cpu = CPU::default().with_arena_allocator(true).build();
    match config.backend {
        InferenceBackend::Cpu => vec![cpu.error_on_failure()],
        InferenceBackend::Cuda => vec![
            build_cuda_provider(config).error_on_failure(),
            cpu.error_on_failure(),
        ],
        InferenceBackend::Migraphx => vec![
            build_migraphx_provider(config).error_on_failure(),
            cpu.error_on_failure(),
        ],
        InferenceBackend::CoreMlNative => vec![cpu.error_on_failure()],
        InferenceBackend::Auto => {
            if cfg!(target_vendor = "apple") && allow_coreml {
                vec![
                    build_coreml_provider(ComputeUnits::CPUAndNeuralEngine, config).fail_silently(),
                    cpu.error_on_failure(),
                ]
            } else {
                vec![cpu.error_on_failure()]
            }
        }
        InferenceBackend::CoreMl => vec![
            build_coreml_provider(ComputeUnits::All, config).error_on_failure(),
            cpu.error_on_failure(),
        ],
        InferenceBackend::CoreMlNeuralEngine => vec![
            build_coreml_provider(ComputeUnits::CPUAndNeuralEngine, config).error_on_failure(),
            cpu.error_on_failure(),
        ],
        InferenceBackend::CoreMlGpu => vec![
            build_coreml_provider(ComputeUnits::CPUAndGPU, config).error_on_failure(),
            cpu.error_on_failure(),
        ],
        InferenceBackend::CoreMlCpu => vec![
            build_coreml_provider(ComputeUnits::CPUOnly, config).error_on_failure(),
            cpu.error_on_failure(),
        ],
    }
}

fn build_coreml_provider(
    compute_units: ComputeUnits,
    config: &InferenceConfig,
) -> ExecutionProviderDispatch {
    let mut provider = CoreML::default()
        .with_compute_units(compute_units)
        .with_static_input_shapes(true)
        .with_specialization_strategy(SpecializationStrategy::Default);
    if let Some(cache_dir) = &config.coreml_cache_dir {
        provider = provider.with_model_cache_dir(cache_dir.display().to_string());
    }
    provider.build()
}

fn build_cuda_provider(config: &InferenceConfig) -> ExecutionProviderDispatch {
    CUDA::default()
        .with_device_id(config.gpu_device_id)
        // Heuristic search keeps first-run setup cheaper than exhaustive cuDNN benchmarking.
        .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
        .build()
}

fn build_migraphx_provider(config: &InferenceConfig) -> ExecutionProviderDispatch {
    MIGraphX::default()
        .with_device_id(config.gpu_device_id)
        .build()
}

fn maybe_preload_cuda_dependencies(config: &InferenceConfig) -> Result<()> {
    if !matches!(config.backend, InferenceBackend::Cuda) {
        return Ok(());
    }
    ort::ep::cuda::preload_dylibs(
        config.cuda_root_dir.as_deref(),
        config.cudnn_root_dir.as_deref(),
    )
    .map_err(|err| anyhow!(err.to_string()))
    .context("failed to preload CUDA/cuDNN dynamic libraries")
}

fn image_to_tensor(image: &DynamicImage) -> Result<Array4<f32>> {
    let resized = image.resize_exact(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let mut tensor = Array4::<f32>::zeros((
        1,
        3,
        MODEL_INPUT_HEIGHT as usize,
        MODEL_INPUT_WIDTH as usize,
    ));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        let xi = x as usize;
        let yi = y as usize;
        tensor[[0, 0, yi, xi]] = r as f32 / 255.0;
        tensor[[0, 1, yi, xi]] = g as f32 / 255.0;
        tensor[[0, 2, yi, xi]] = b as f32 / 255.0;
    }
    Ok(tensor)
}
