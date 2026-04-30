use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::motion::{LayerMotion, MotionTuning};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    pub version: u32,
    pub generated_at_unix_sec: u64,
    pub source: SourceImageInfo,
    pub model: ModelManifest,
    pub package: PackageInfo,
    pub slicing: SlicingManifest,
    pub motion: MotionManifest,
    pub layers: Vec<LayerManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceImageInfo {
    pub input_path: String,
    pub original_width: u32,
    pub original_height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub repo: String,
    pub revision: String,
    pub endpoint: Option<String>,
    pub format: String,
    pub download_backend: String,
    pub inference_backend: String,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub gpu_device_id: i32,
    pub coreml_cache_dir: Option<String>,
    pub cache_dir: Option<String>,
    pub model_path: String,
    pub model_data_path: Option<String>,
    pub disparity_factor: f32,
    pub input_width: u32,
    pub input_height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub width: u32,
    pub height: u32,
    pub layer_count: usize,
    pub backdrop_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlicingManifest {
    pub preview_max_edge: u32,
    pub preview_target_pixels: u64,
    pub gamma: f32,
    pub depth_fill_passes: usize,
    pub blur_backdrop_scale: f32,
    pub projected_coverage: f32,
    pub sample_step: usize,
    pub depth_min: f32,
    pub depth_max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionManifest {
    pub tilt_default_deg: f32,
    pub parallax_default: f32,
    pub tuning: MotionTuning,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerManifest {
    pub name: String,
    pub file: String,
    pub is_backdrop: bool,
    pub band: Option<usize>,
    pub near_weight: f32,
    pub pixels: u32,
    pub draw_order: u32,
    pub motion: LayerMotion,
}

impl PackageManifest {
    pub fn layer_paths<'a>(
        &'a self,
        base_dir: &'a Path,
    ) -> impl Iterator<Item = std::path::PathBuf> + 'a {
        self.layers.iter().map(|layer| base_dir.join(&layer.file))
    }
}
