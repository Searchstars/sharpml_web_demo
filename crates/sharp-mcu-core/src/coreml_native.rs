use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array1, Array2, Ix2, Ix3};
use serde::Deserialize;

use crate::{PreparedImage, SharpOutputs};

pub const DEFAULT_APPLE_MODEL_REPO: &str = "pearsonkyle/Sharp-coreml";
pub const DEFAULT_APPLE_MODEL_FILE: &str = "sharp.mlpackage";
pub const DEFAULT_COREML_NATIVE_CACHE_DIR: &str = "caches/coreml-native";

#[derive(Debug, Deserialize)]
struct OutputManifest {
    tensors: Vec<TensorMeta>,
}

#[derive(Debug, Deserialize)]
struct TensorMeta {
    name: String,
    file: String,
    shape: Vec<usize>,
}

pub struct SharpCoreMlNative {
    model_path: PathBuf,
    compiled_cache_dir: PathBuf,
    runner_path: PathBuf,
}

impl SharpCoreMlNative {
    pub fn new(model_path: &Path, coreml_cache_dir: Option<&Path>) -> Result<Self> {
        if !cfg!(target_vendor = "apple") {
            bail!("native CoreML backend is only supported on Apple host platforms");
        }
        let cache_root = coreml_cache_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(default_coreml_native_cache_dir);
        let compiled_cache_dir = cache_root.join("compiled-models");
        let runner_bin_dir = cache_root.join("runner-bin");
        fs::create_dir_all(&compiled_cache_dir).with_context(|| {
            format!(
                "failed to create CoreML cache dir {}",
                compiled_cache_dir.display()
            )
        })?;
        fs::create_dir_all(&runner_bin_dir).with_context(|| {
            format!(
                "failed to create CoreML runner cache dir {}",
                runner_bin_dir.display()
            )
        })?;
        let runner_path = compile_runner_if_needed(&runner_bin_dir)?;
        Ok(Self {
            model_path: model_path.to_path_buf(),
            compiled_cache_dir,
            runner_path,
        })
    }

    pub fn infer(&self, image: &PreparedImage, disparity_factor: f32) -> Result<SharpOutputs> {
        let temp_root = std::env::temp_dir().join(format!(
            "sharp-coreml-native-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&temp_root)
            .with_context(|| format!("failed to create temp dir {}", temp_root.display()))?;

        let status = Command::new(&self.runner_path)
            .arg(&self.model_path)
            .arg(&image.input_path)
            .arg(&temp_root)
            .arg(disparity_factor.to_string())
            .arg(&self.compiled_cache_dir)
            .status()
            .with_context(|| {
                format!(
                    "failed to spawn CoreML helper {}",
                    self.runner_path.display()
                )
            })?;
        if !status.success() {
            bail!(
                "CoreML helper failed for model {}",
                self.model_path.display()
            );
        }

        let result = load_outputs(&temp_root);
        let _ = fs::remove_dir_all(&temp_root);
        result
    }
}

pub fn default_coreml_native_cache_dir() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or(crate_dir.as_path())
        .join(DEFAULT_COREML_NATIVE_CACHE_DIR)
}

fn compile_runner_if_needed(runner_bin_dir: &Path) -> Result<PathBuf> {
    let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("apple")
        .join("sharp_coreml_runner.swift");
    let runner_path = runner_bin_dir.join("sharp_coreml_runner");

    let needs_compile = match (fs::metadata(&source_path), fs::metadata(&runner_path)) {
        (Ok(source), Ok(bin)) => {
            let source_time = source.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let bin_time = bin.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            bin_time < source_time
        }
        (Ok(_), Err(_)) => true,
        (Err(err), _) => {
            return Err(err).with_context(|| format!("failed to stat {}", source_path.display()));
        }
    };

    if !needs_compile {
        return Ok(runner_path);
    }

    let status = Command::new("swiftc")
        .arg("-O")
        .arg("-o")
        .arg(&runner_path)
        .arg(&source_path)
        .arg("-framework")
        .arg("CoreML")
        .arg("-framework")
        .arg("CoreImage")
        .arg("-framework")
        .arg("AppKit")
        .status()
        .with_context(|| "failed to invoke swiftc for CoreML helper")?;
    if !status.success() {
        bail!("swiftc failed while compiling CoreML helper");
    }

    Ok(runner_path)
}

fn load_outputs(output_dir: &Path) -> Result<SharpOutputs> {
    let manifest_path = output_dir.join("outputs.json");
    let manifest_bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "failed to read CoreML output manifest {}",
            manifest_path.display()
        )
    })?;
    let manifest: OutputManifest = serde_json::from_slice(&manifest_bytes).with_context(|| {
        format!(
            "failed to parse CoreML output manifest {}",
            manifest_path.display()
        )
    })?;

    let mut mean_vectors = None;
    let mut singular_values = None;
    let mut quaternions = None;
    let mut colors = None;
    let mut opacities = None;
    let mut output_names = Vec::new();

    for tensor in &manifest.tensors {
        output_names.push(tensor.name.clone());
        let path = output_dir.join(&tensor.file);
        let values = read_f32_file(&path)?;
        match tensor.name.as_str() {
            "mean_vectors_3d_positions" => {
                mean_vectors = Some(array2_from_tensor(&values, &tensor.shape)?);
            }
            "singular_values_scales" => {
                singular_values = Some(array2_from_tensor(&values, &tensor.shape)?);
            }
            "quaternions_rotations" => {
                quaternions = Some(array2_from_tensor(&values, &tensor.shape)?);
            }
            "colors_rgb_linear" => {
                colors = Some(array2_from_tensor(&values, &tensor.shape)?);
            }
            "opacities_alpha_channel" => {
                opacities = Some(array1_from_tensor(&values, &tensor.shape)?);
            }
            _ => {}
        }
    }

    Ok(SharpOutputs {
        mean_vectors: mean_vectors.context("CoreML outputs missing mean_vectors_3d_positions")?,
        singular_values: singular_values
            .context("CoreML outputs missing singular_values_scales")?,
        quaternions: quaternions.context("CoreML outputs missing quaternions_rotations")?,
        colors: colors.context("CoreML outputs missing colors_rgb_linear")?,
        opacities: opacities.context("CoreML outputs missing opacities_alpha_channel")?,
        output_names,
    })
}

fn read_f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read tensor file {}", path.display()))?;
    if bytes.len() % 4 != 0 {
        bail!(
            "tensor file {} had invalid byte length {}",
            path.display(),
            bytes.len()
        );
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn array2_from_tensor(values: &[f32], shape: &[usize]) -> Result<Array2<f32>> {
    match shape {
        [1, rows, cols] => ndarray::Array::from_shape_vec((*rows, *cols), values.to_vec())
            .map_err(|err| anyhow!(err.to_string())),
        [rows, cols] => ndarray::Array::from_shape_vec((*rows, *cols), values.to_vec())
            .map_err(|err| anyhow!(err.to_string())),
        _ => {
            let arr = ndarray::Array::from_shape_vec(shape.to_vec(), values.to_vec())
                .map_err(|err| anyhow!(err.to_string()))?;
            let arr = arr
                .into_dimensionality::<Ix3>()
                .map_err(|err| anyhow!(err.to_string()))?;
            Ok(arr.index_axis(ndarray::Axis(0), 0).to_owned())
        }
    }
}

fn array1_from_tensor(values: &[f32], shape: &[usize]) -> Result<Array1<f32>> {
    match shape {
        [1, len] => ndarray::Array::from_shape_vec(*len, values.to_vec())
            .map_err(|err| anyhow!(err.to_string())),
        [len] => ndarray::Array::from_shape_vec(*len, values.to_vec())
            .map_err(|err| anyhow!(err.to_string())),
        _ => {
            let arr = ndarray::Array::from_shape_vec(shape.to_vec(), values.to_vec())
                .map_err(|err| anyhow!(err.to_string()))?;
            let arr = arr
                .into_dimensionality::<Ix2>()
                .map_err(|err| anyhow!(err.to_string()))?;
            Ok(arr.index_axis(ndarray::Axis(0), 0).to_owned())
        }
    }
}
