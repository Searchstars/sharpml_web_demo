use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use image::{
    Rgba, RgbaImage,
    imageops::{self, FilterType},
};

use crate::{
    InferenceBackend, InferenceConfig, LayerManifest, ModelDownloadBackend, ModelFormat,
    ModelManifest, MotionManifest, MotionTuning, PackageInfo, PackageManifest, PreparedImage,
    SharpCoreMlNative, SharpOnnx, SharpOutputs, SlicingManifest, SourceImageInfo,
    layer_motion_for_weight, prepare_image, resolve_model, write_debug_ply,
};

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub layer_count: usize,
    pub preview_max_edge: u32,
    pub preview_target_pixels: u64,
    pub disparity_factor: f32,
    pub depth_scale: f32,
    pub debug_ply_decimation: f32,
    pub depth_fill_passes: usize,
    pub gamma: f32,
    pub blur_backdrop_scale: f32,
    pub tilt_default_deg: f32,
    pub parallax_default: f32,
    pub emit_debug_ply: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            // 16 bands (was 10): smaller per-band depth slices means less
            // visible seam between adjacent layers when bipolar parallax
            // pulls them apart. Combined with the wider feather in
            // layer_alpha_for_depth, the band geometry blends into a
            // continuous gradient rather than discrete steps.
            layer_count: 16,
            preview_max_edge: 2560,
            preview_target_pixels: 2560 * 1440,
            disparity_factor: 1.0,
            depth_scale: 1.0,
            debug_ply_decimation: 1.0,
            depth_fill_passes: 6,
            gamma: 1.55,
            blur_backdrop_scale: 0.006,
            tilt_default_deg: 12.0,
            parallax_default: 0.10,
            emit_debug_ply: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub input_path: PathBuf,
    pub output_dir: PathBuf,
    pub model_path: Option<PathBuf>,
    pub model_endpoint: Option<String>,
    pub download_backend: ModelDownloadBackend,
    pub aria2_connections: usize,
    pub inference: InferenceConfig,
    pub model_cache_dir: Option<PathBuf>,
    pub model_repo: String,
    pub model_revision: String,
    pub model_file: String,
    pub model_data_file: Option<String>,
    pub config: GenerateConfig,
}

#[derive(Debug, Clone)]
pub struct GenerateReport {
    pub manifest_path: PathBuf,
    pub debug_ply_path: Option<PathBuf>,
    pub manifest: PackageManifest,
}

pub fn generate_package(request: &GenerateRequest) -> Result<GenerateReport> {
    if request.config.layer_count == 0 || request.config.layer_count > 255 {
        bail!("layer_count must be within 1..=255");
    }

    fs::create_dir_all(&request.output_dir).with_context(|| {
        format!(
            "failed to create output dir {}",
            request.output_dir.display()
        )
    })?;

    let prepared = prepare_image(&request.input_path)?;
    let resolved_model = resolve_model(
        request.model_path.as_deref(),
        request.model_endpoint.as_deref(),
        request.model_cache_dir.as_deref(),
        request.download_backend,
        request.aria2_connections,
        &request.model_repo,
        &request.model_revision,
        &request.model_file,
        request.model_data_file.as_deref(),
    )?;
    let (outputs, effective_backend) = run_model_inference(
        &resolved_model,
        &request.inference,
        &prepared,
        request.config.disparity_factor,
    )?;

    let preview_size = fit_size(
        prepared.original_width,
        prepared.original_height,
        request.config.preview_max_edge,
        request.config.preview_max_edge,
        request.config.preview_target_pixels,
    );
    let source_preview = imageops::resize(
        &prepared.original_rgba,
        preview_size.width,
        preview_size.height,
        FilterType::Triangle,
    );

    let mut projected = project_depth_map_from_outputs(
        &outputs,
        &prepared,
        preview_size.width,
        preview_size.height,
    )?;
    let (depth_min, depth_max) = fill_depth_holes(
        &mut projected.depth,
        preview_size.width,
        preview_size.height,
        request.config.depth_fill_passes,
    )?;
    median_filter_depth(
        &mut projected.depth,
        preview_size.width,
        preview_size.height,
    );

    let boundaries = compute_depth_boundaries(
        &projected.depth,
        request.config.layer_count,
        request.config.gamma,
    );
    let total_pixels = (preview_size.width as usize) * (preview_size.height as usize);
    let mut band_map = vec![0u8; total_pixels];
    let mut band_pixels = vec![0u32; request.config.layer_count];
    for (idx, depth) in projected.depth.iter().copied().enumerate() {
        let band = depth_band_for_value(depth, &boundaries);
        band_map[idx] = band as u8;
    }
    // Snapshot the raw (un-smoothed) band map for backdrop inpainting. The
    // backdrop seeds need crisp foreground/background separation; the smoothed
    // band map blurs depth boundaries, contaminating JFA seeds with silhouette
    // pixels and reproducing foreground colors in the backdrop.
    let raw_band_map = band_map.clone();
    smooth_band_map(
        &mut band_map,
        preview_size.width as usize,
        preview_size.height as usize,
        request.config.layer_count,
    );
    for &band in &band_map {
        band_pixels[band as usize] += 1;
    }

    let backdrop = build_inpainted_backdrop(
        &source_preview,
        &raw_band_map,
        preview_size.width as usize,
        preview_size.height as usize,
        request.config.layer_count,
        request.config.blur_backdrop_scale,
    );
    let backdrop_path = request.output_dir.join("backdrop.png");
    backdrop.save(&backdrop_path)?;

    let tuning = MotionTuning::default();
    let mut layers = Vec::with_capacity(request.config.layer_count + 1);
    let backdrop_motion = layer_motion_for_weight(true, 0.0, &tuning);
    layers.push(LayerManifest {
        name: "backdrop".to_string(),
        file: "backdrop.png".to_string(),
        is_backdrop: true,
        band: None,
        near_weight: 0.0,
        pixels: total_pixels as u32,
        draw_order: 0,
        motion: backdrop_motion,
    });

    let total_depth_span = (boundaries[boundaries.len() - 1] - boundaries[0]).max(1e-5);
    let denom = request.config.layer_count.saturating_sub(1).max(1) as f32;
    for (draw_idx, band) in (0..request.config.layer_count).rev().enumerate() {
        let near_weight = 1.0 - band as f32 / denom;
        let rgba = build_layer_rgba(
            &source_preview,
            &projected.depth,
            band,
            &boundaries,
            total_depth_span,
        );
        let file_name = format!("layer_{band:02}.png");
        let path = request.output_dir.join(&file_name);
        rgba.save(&path)?;
        layers.push(LayerManifest {
            name: format!("layer-{band:02}"),
            file: file_name,
            is_backdrop: false,
            band: Some(band),
            near_weight,
            pixels: band_pixels[band],
            draw_order: (draw_idx + 1) as u32,
            motion: layer_motion_for_weight(false, near_weight, &tuning),
        });
    }

    let manifest = PackageManifest {
        version: 1,
        generated_at_unix_sec: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        source: SourceImageInfo {
            input_path: request.input_path.display().to_string(),
            original_width: prepared.original_width,
            original_height: prepared.original_height,
        },
        model: ModelManifest {
            repo: resolved_model.repo.clone(),
            revision: resolved_model.revision.clone(),
            endpoint: resolved_model.endpoint.clone(),
            format: resolved_model.format.as_str().to_string(),
            download_backend: resolved_model.download_backend.as_str().to_string(),
            inference_backend: effective_backend.as_str().to_string(),
            intra_threads: request.inference.intra_threads,
            inter_threads: request.inference.inter_threads,
            gpu_device_id: request.inference.gpu_device_id,
            coreml_cache_dir: request
                .inference
                .coreml_cache_dir
                .as_ref()
                .map(|path| path.display().to_string()),
            cache_dir: resolved_model
                .cache_dir
                .as_ref()
                .map(|path| path.display().to_string()),
            model_path: resolved_model.model_path.display().to_string(),
            model_data_path: resolved_model
                .model_data_path
                .as_ref()
                .map(|path| path.display().to_string()),
            disparity_factor: request.config.disparity_factor,
            input_width: crate::onnx::MODEL_INPUT_WIDTH,
            input_height: crate::onnx::MODEL_INPUT_HEIGHT,
        },
        package: PackageInfo {
            width: preview_size.width,
            height: preview_size.height,
            layer_count: request.config.layer_count,
            backdrop_file: "backdrop.png".to_string(),
        },
        slicing: SlicingManifest {
            preview_max_edge: request.config.preview_max_edge,
            preview_target_pixels: request.config.preview_target_pixels,
            gamma: request.config.gamma,
            depth_fill_passes: request.config.depth_fill_passes,
            blur_backdrop_scale: request.config.blur_backdrop_scale,
            projected_coverage: projected.coverage,
            sample_step: projected.sample_step,
            depth_min,
            depth_max,
        },
        motion: MotionManifest {
            tilt_default_deg: request.config.tilt_default_deg,
            parallax_default: request.config.parallax_default,
            tuning,
        },
        layers,
    };

    let manifest_path = request.output_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("failed to write {}", manifest_path.display()))?;

    let debug_ply_path = if request.config.emit_debug_ply {
        let path = request.output_dir.join("debug.ply");
        write_debug_ply(
            &outputs,
            &path,
            prepared.focal_length_px,
            (prepared.original_width, prepared.original_height),
            request.config.debug_ply_decimation,
            request.config.depth_scale,
        )?;
        Some(path)
    } else {
        None
    };

    Ok(GenerateReport {
        manifest_path,
        debug_ply_path,
        manifest,
    })
}

fn run_model_inference(
    resolved_model: &crate::ResolvedModel,
    inference: &InferenceConfig,
    prepared: &PreparedImage,
    disparity_factor: f32,
) -> Result<(SharpOutputs, InferenceBackend)> {
    match resolved_model.format {
        ModelFormat::Onnx => {
            if matches!(inference.backend, InferenceBackend::CoreMlNative) {
                bail!("coreml-native backend requires a .mlpackage/.mlmodel asset");
            }
            let mut model = SharpOnnx::new(resolved_model, inference)?;
            Ok((model.infer(prepared, disparity_factor)?, inference.backend))
        }
        ModelFormat::CoreMlPackage => {
            if !cfg!(target_vendor = "apple") {
                bail!("native CoreML models are only supported on Apple host platforms");
            }
            match inference.backend {
                InferenceBackend::Auto | InferenceBackend::CoreMlNative => {
                    let model = SharpCoreMlNative::new(
                        &resolved_model.model_path,
                        inference.coreml_cache_dir.as_deref(),
                    )?;
                    Ok((
                        model.infer(prepared, disparity_factor)?,
                        InferenceBackend::CoreMlNative,
                    ))
                }
                InferenceBackend::Cpu
                | InferenceBackend::Cuda
                | InferenceBackend::Migraphx
                | InferenceBackend::CoreMl
                | InferenceBackend::CoreMlNeuralEngine
                | InferenceBackend::CoreMlGpu
                | InferenceBackend::CoreMlCpu => {
                    bail!(
                        "model {} is a native CoreML asset; use --inference-backend coreml-native or auto",
                        resolved_model.model_path.display()
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FitSize {
    width: u32,
    height: u32,
}

#[derive(Debug, Clone)]
struct ProjectedDepth {
    depth: Vec<f32>,
    coverage: f32,
    sample_step: usize,
}

fn fit_size(
    width: u32,
    height: u32,
    max_width: u32,
    max_height: u32,
    target_pixels: u64,
) -> FitSize {
    let mut scale = (max_width as f32 / width as f32)
        .min(max_height as f32 / height as f32)
        .min(1.0);
    let scaled_pixels = width as f32 * height as f32 * scale * scale;
    if scaled_pixels > target_pixels as f32 {
        scale = (target_pixels as f32 / (width as f32 * height as f32)).sqrt();
    }
    FitSize {
        width: ((width as f32 * scale).round() as u32).max(1),
        height: ((height as f32 * scale).round() as u32).max(1),
    }
}

fn project_depth_map_from_outputs(
    outputs: &SharpOutputs,
    prepared: &PreparedImage,
    target_width: u32,
    target_height: u32,
) -> Result<ProjectedDepth> {
    if outputs.mean_vectors.ncols() != 3 {
        bail!("mean_vectors output must have 3 channels");
    }
    let mut depth = vec![f32::INFINITY; (target_width as usize) * (target_height as usize)];
    let z_min = outputs
        .mean_vectors
        .column(2)
        .iter()
        .copied()
        .filter(|z| z.is_finite() && *z > 1e-5)
        .fold(f32::INFINITY, f32::min);
    if !z_min.is_finite() {
        bail!("ONNX output did not contain valid depth samples");
    }

    let focal_ndc = 2.0 * prepared.focal_length_px / prepared.original_width as f32;
    let scale_factor = 1.0 / (z_min * focal_ndc);
    let scale_x = target_width as f32 / prepared.original_width as f32;
    let scale_y = target_height as f32 / prepared.original_height as f32;
    // Rust preprocessing can spend more time here than the browser prototype.
    // Use a denser point projection so depth bands don't get pepper noise from
    // undersampling the SHARP output field.
    let desired_samples = ((target_width as usize) * (target_height as usize)) as f32 * 1.6;
    let sample_step =
        ((outputs.mean_vectors.nrows() as f32 / desired_samples).floor() as usize).max(1);
    let stamp_radius = if sample_step >= 4 { 1 } else { 0 };
    let mut filled = 0usize;

    for idx in (0..outputs.mean_vectors.nrows()).step_by(sample_step) {
        let z_raw = outputs.mean_vectors[[idx, 2]];
        if !(z_raw > 1e-5) || !z_raw.is_finite() {
            continue;
        }
        let z = z_raw / z_min;
        let x = outputs.mean_vectors[[idx, 0]] * scale_factor;
        let y = outputs.mean_vectors[[idx, 1]] * scale_factor;
        let u = prepared.focal_length_px * x / z + prepared.original_width as f32 / 2.0;
        let v = prepared.focal_length_px * y / z + prepared.original_height as f32 / 2.0;
        let px = (u * scale_x).round() as i32;
        let py = (v * scale_y).round() as i32;
        if px < 0 || px >= target_width as i32 || py < 0 || py >= target_height as i32 {
            continue;
        }
        for oy in -stamp_radius..=stamp_radius {
            for ox in -stamp_radius..=stamp_radius {
                let sx = px + ox;
                let sy = py + oy;
                if sx < 0 || sy < 0 || sx >= target_width as i32 || sy >= target_height as i32 {
                    continue;
                }
                let slot = sy as usize * target_width as usize + sx as usize;
                if z < depth[slot] {
                    if !depth[slot].is_finite() {
                        filled += 1;
                    }
                    depth[slot] = z;
                }
            }
        }
    }

    Ok(ProjectedDepth {
        depth,
        coverage: filled as f32 / ((target_width as usize) * (target_height as usize)) as f32,
        sample_step,
    })
}

fn fill_depth_holes(
    depth: &mut [f32],
    width: u32,
    height: u32,
    passes: usize,
) -> Result<(f32, f32)> {
    let len = depth.len();
    let mut src = depth.to_vec();
    let mut dst = vec![0.0f32; len];

    for _ in 0..passes {
        let mut changed = 0usize;
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let idx = y as usize * width as usize + x as usize;
                if src[idx].is_finite() {
                    dst[idx] = src[idx];
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0.0;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        if ox == 0 && oy == 0 {
                            continue;
                        }
                        let sx = x + ox;
                        let sy = y + oy;
                        if sx < 0 || sy < 0 || sx >= width as i32 || sy >= height as i32 {
                            continue;
                        }
                        let sample = src[sy as usize * width as usize + sx as usize];
                        if sample.is_finite() {
                            sum += sample;
                            count += 1.0;
                        }
                    }
                }
                if count > 0.0 {
                    dst[idx] = sum / count;
                    changed += 1;
                } else {
                    dst[idx] = f32::INFINITY;
                }
            }
        }
        std::mem::swap(&mut src, &mut dst);
        if changed == 0 {
            break;
        }
    }

    let mut min_depth = f32::INFINITY;
    let mut max_depth = 0.0f32;
    for value in src.iter().copied().filter(|v| v.is_finite()) {
        min_depth = min_depth.min(value);
        max_depth = max_depth.max(value);
    }
    if !min_depth.is_finite() || !max_depth.is_finite() {
        bail!("depth projection produced no valid samples");
    }

    for value in &mut src {
        if !value.is_finite() {
            *value = max_depth;
        }
    }
    depth.copy_from_slice(&src);
    Ok((min_depth, max_depth))
}

fn median_filter_depth(depth: &mut [f32], width: u32, height: u32) {
    const MEDIAN_PASSES: usize = 3;
    let mut src = depth.to_vec();
    let mut filtered = vec![0.0f32; depth.len()];
    let mut window = [0.0f32; 9];
    for _ in 0..MEDIAN_PASSES {
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let mut n = 0usize;
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        let sx = (x + ox).clamp(0, width as i32 - 1);
                        let sy = (y + oy).clamp(0, height as i32 - 1);
                        window[n] = src[sy as usize * width as usize + sx as usize];
                        n += 1;
                    }
                }
                window[..n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                filtered[y as usize * width as usize + x as usize] = window[n / 2];
            }
        }
        std::mem::swap(&mut src, &mut filtered);
    }
    depth.copy_from_slice(&src);
}

fn compute_depth_boundaries(depth: &[f32], layer_count: usize, gamma: f32) -> Vec<f32> {
    let stride = (depth.len() / 220_000).max(1);
    let mut samples = depth
        .iter()
        .step_by(stride)
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let last = samples.len().saturating_sub(1);
    let mut boundaries = vec![0.0f32; layer_count + 1];
    boundaries[0] = samples[0];
    for i in 1..layer_count {
        let t = i as f32 / layer_count as f32;
        let warped = t.powf(gamma);
        let index = ((last as f32) * warped).floor() as usize;
        boundaries[i] = samples[index.min(last)];
    }
    boundaries[layer_count] = samples[last] + 1e-5;
    for i in 1..boundaries.len() {
        if boundaries[i] <= boundaries[i - 1] {
            boundaries[i] = boundaries[i - 1] + 1e-5;
        }
    }
    boundaries
}

fn depth_band_for_value(value: f32, boundaries: &[f32]) -> usize {
    for i in 0..boundaries.len() - 1 {
        if value <= boundaries[i + 1] {
            return i;
        }
    }
    boundaries.len() - 2
}

fn build_layer_rgba(
    source: &RgbaImage,
    depth: &[f32],
    band: usize,
    boundaries: &[f32],
    total_span: f32,
) -> RgbaImage {
    // Layers are stored with premultiplied alpha so the bilinear-sampling
    // preview pipeline doesn't paint dark contour-tracing halos along every
    // opaque/transparent edge (which is what straight alpha + bilinear
    // filtering produces — RGB falls off as α² across edges).
    //
    // Crucially the premultiply must happen in LINEAR light, not sRGB. Doing
    // `rgb_srgb * alpha` and then storing through Rgba8UnormSrgb darkens
    // partial-coverage texels by roughly the gamma curve (e.g. an α=0.5 red
    // texel ends up sampling as ~0.21 instead of 0.5 in linear space), which
    // shows up as desaturated gray patches following every band edge.
    let (width, height) = source.dimensions();
    let mut output = RgbaImage::new(width, height);
    for (idx, pixel) in output.pixels_mut().enumerate() {
        let alpha_weight = layer_alpha_for_depth(depth[idx], band, boundaries, total_span);
        if alpha_weight <= 0.0 {
            *pixel = Rgba([0, 0, 0, 0]);
            continue;
        }
        let src = source
            .get_pixel((idx % width as usize) as u32, (idx / width as usize) as u32)
            .0;
        let coverage = (src[3] as f32 / 255.0) * alpha_weight;
        let coverage = coverage.clamp(0.0, 1.0);
        *pixel = Rgba([
            linear_to_srgb_u8(srgb_u8_to_linear(src[0]) * coverage),
            linear_to_srgb_u8(srgb_u8_to_linear(src[1]) * coverage),
            linear_to_srgb_u8(srgb_u8_to_linear(src[2]) * coverage),
            (coverage * 255.0).round().clamp(0.0, 255.0) as u8,
        ]);
    }
    output
}

fn srgb_u8_to_linear(value: u8) -> f32 {
    let v = value as f32 / 255.0;
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let l = linear.clamp(0.0, 1.0);
    let s = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
}

fn smooth_band_map(band_map: &mut [u8], width: usize, height: usize, layer_count: usize) {
    const PASSES: usize = 2;
    let mut src = band_map.to_vec();
    let mut dst = src.clone();
    for _ in 0..PASSES {
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let center = src[idx] as usize;
                let mut counts = [0u8; 256];
                counts[center] = counts[center].saturating_add(2);
                for neighbor in neighbor_indices(x, y, width, height) {
                    let band = src[neighbor] as usize;
                    counts[band] = counts[band].saturating_add(1);
                }
                let center_count = counts[center];
                let mut best_band = center;
                let mut best_count = center_count;
                for band in 0..layer_count {
                    if counts[band] > best_count {
                        best_count = counts[band];
                        best_band = band;
                    }
                }
                dst[idx] = if best_band != center && best_count >= center_count + 2 {
                    best_band as u8
                } else {
                    src[idx]
                };
            }
        }
        std::mem::swap(&mut src, &mut dst);
    }
    band_map.copy_from_slice(&src);
}

fn neighbor_indices(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = usize> {
    let x0 = x.saturating_sub(1);
    let y0 = y.saturating_sub(1);
    let x1 = (x + 1).min(width - 1);
    let y1 = (y + 1).min(height - 1);
    let mut neighbors = [usize::MAX; 8];
    let mut n = 0usize;
    for sy in y0..=y1 {
        for sx in x0..=x1 {
            if sx == x && sy == y {
                continue;
            }
            neighbors[n] = sy * width + sx;
            n += 1;
        }
    }
    neighbors.into_iter().take(n)
}

/// Build a content-aware backdrop: the parts of the image that belong to FAR
/// bands are kept as-is, while NEAR (foreground) regions are inpainted with
/// the spatially-nearest FAR pixel. The result is what should plausibly be
/// "behind" the foreground, so that when a near layer translates during
/// parallax, what gets revealed underneath is a reasonable continuation of
/// the background (sky, wall, ground) rather than a blurry ghost of the
/// foreground itself.
///
/// Uses Jump Flooding Algorithm (Rong & Tan 2006) to compute, for every
/// foreground pixel, an approximation of its spatially-nearest background
/// seed in O(N log max(W,H)) time. JFA is exact for most pixels and
/// off-by-one in pathological cases; a final Gaussian blur hides any seam.
fn build_inpainted_backdrop(
    source: &RgbaImage,
    band_map: &[u8],
    width: usize,
    height: usize,
    layer_count: usize,
    blur_scale: f32,
) -> RgbaImage {
    let w = width as u32;
    let h = height as u32;
    let len = width * height;
    if len == 0 {
        return RgbaImage::new(w, h);
    }
    // Stricter seed threshold + aggressive erosion: only pixels that are
    // genuinely deep inside the far-band region count as JFA seeds, never the
    // silhouette ring (whose colors are a mix of foreground and background and
    // would propagate "edge-of-foliage green" inward). The erode radius scales
    // with the preview size — small previews can use radius 4, large ones 8+,
    // because what matters is removing the sub-pixel-fuzzy depth silhouette.
    let seed_band_threshold = ((layer_count as f32) * 0.62).round() as u8;
    let seed_erode_radius =
        ((width.min(height) as f32 * 0.008).round() as i32).clamp(3, 12);

    let mut raw_seed = vec![false; len];
    for idx in 0..len {
        if band_map[idx] >= seed_band_threshold {
            raw_seed[idx] = true;
        }
    }
    // Erode by computing a horizontal-then-vertical min-pass on the raw mask.
    // Equivalent to the brute-force 2D radius-R neighborhood test but O(N·R)
    // instead of O(N·R²), which matters at radius ~10 on large previews.
    let mut horiz = vec![false; len];
    for y in 0..height {
        for x in 0..width {
            let mut all_far = true;
            for ox in -seed_erode_radius..=seed_erode_radius {
                let sx = x as i32 + ox;
                if sx < 0 || sx >= width as i32 || !raw_seed[y * width + sx as usize] {
                    all_far = false;
                    break;
                }
            }
            horiz[y * width + x] = all_far;
        }
    }
    let mut strict_seed = vec![false; len];
    for y in 0..height {
        for x in 0..width {
            let mut all_far = true;
            for oy in -seed_erode_radius..=seed_erode_radius {
                let sy = y as i32 + oy;
                if sy < 0 || sy >= height as i32 || !horiz[sy as usize * width + x] {
                    all_far = false;
                    break;
                }
            }
            strict_seed[y * width + x] = all_far;
        }
    }
    // Failsafe: if erosion removed everything (very foreground-heavy image),
    // fall back to the un-eroded mask so we still have anchors for the JFA.
    if !strict_seed.iter().any(|&v| v) {
        strict_seed = raw_seed.clone();
    }
    const NONE: i32 = i32::MIN;
    let mut buf_a: Vec<(i32, i32)> = vec![(NONE, NONE); len];
    for idx in 0..len {
        if strict_seed[idx] {
            let x = (idx % width) as i32;
            let y = (idx / width) as i32;
            buf_a[idx] = (x, y);
        }
    }
    let mut buf_b: Vec<(i32, i32)> = buf_a.clone();

    let mut step = (width.max(height) / 2).max(1);
    let mut read_from_a = true;
    loop {
        let (src_buf, dst_buf) = if read_from_a {
            (&buf_a, &mut buf_b)
        } else {
            (&buf_b, &mut buf_a)
        };
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let here = (y as usize) * width + x as usize;
                let mut best = src_buf[here];
                let mut best_dist = if best.0 == NONE {
                    i64::MAX
                } else {
                    let dx = (best.0 - x) as i64;
                    let dy = (best.1 - y) as i64;
                    dx * dx + dy * dy
                };
                let s = step as i32;
                for oy in &[-s, 0, s] {
                    for ox in &[-s, 0, s] {
                        if *ox == 0 && *oy == 0 {
                            continue;
                        }
                        let nx = x + *ox;
                        let ny = y + *oy;
                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            continue;
                        }
                        let candidate = src_buf[(ny as usize) * width + nx as usize];
                        if candidate.0 == NONE {
                            continue;
                        }
                        let dx = (candidate.0 - x) as i64;
                        let dy = (candidate.1 - y) as i64;
                        let d = dx * dx + dy * dy;
                        if d < best_dist {
                            best_dist = d;
                            best = candidate;
                        }
                    }
                }
                dst_buf[here] = best;
            }
        }
        read_from_a = !read_from_a;
        if step == 1 {
            break;
        }
        step /= 2;
    }
    let nearest = if read_from_a { &buf_a } else { &buf_b };

    let mut filled = RgbaImage::new(w, h);
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            let here = (y as usize) * width + x as usize;
            let (sx, sy) = nearest[here];
            let pixel = if sx == NONE {
                source.get_pixel(x, y).0
            } else {
                source.get_pixel(sx as u32, sy as u32).0
            };
            filled.put_pixel(x, y, Rgba(pixel));
        }
    }

    // Hide JFA's Voronoi cells with a large-radius diffusion applied ONLY in
    // the inpainted region. Genuine background pixels keep their original
    // detail. The transition is feathered using each pixel's distance to its
    // nearest seed (which JFA already gave us), so we don't get a hard
    // boundary between sharp source and blurry fill.
    let small_sigma = (w.min(h) as f32 * blur_scale).max(2.0);
    let heavy_sigma = (w.min(h) as f32 * 0.04).max(small_sigma * 4.0);
    let lightly_smoothed = imageops::blur(&filled, small_sigma);
    let heavily_smoothed = imageops::blur(&filled, heavy_sigma);
    // Feather radius (pixels): how far the seed→inpaint transition is spread.
    // Bigger = softer seam, but also bleeds blur into background detail.
    let feather_radius = ((w.min(h) as f32) * 0.012).max(6.0);
    let feather_sq = feather_radius * feather_radius;
    let mut output = RgbaImage::new(w, h);
    for y in 0..height as u32 {
        for x in 0..width as u32 {
            let here = (y as usize) * width + x as usize;
            let (sx, sy) = nearest[here];
            // Distance from this pixel to its nearest strict seed (in px²).
            let seed_dist_sq = if sx == NONE {
                feather_sq * 4.0
            } else {
                let dx = sx as f32 - x as f32;
                let dy = sy as f32 - y as f32;
                dx * dx + dy * dy
            };
            // Inpaint-mix ramps from 0 at a strict seed pixel to 1 once we're
            // a feather-radius away inside the foreground silhouette.
            let inpaint_t = smoothstep01(seed_dist_sq / feather_sq);
            // Source pixel for sharp content: at seeds we want the ORIGINAL
            // image (filled[seed] == source[seed] anyway), and inside the
            // silhouette we let the heavy blur dominate.
            let sharp = if strict_seed[here] {
                source.get_pixel(x, y).0
            } else {
                lightly_smoothed.get_pixel(x, y).0
            };
            let blurry = heavily_smoothed.get_pixel(x, y).0;
            let inv = 1.0 - inpaint_t;
            output.put_pixel(
                x,
                y,
                Rgba([
                    (sharp[0] as f32 * inv + blurry[0] as f32 * inpaint_t).round() as u8,
                    (sharp[1] as f32 * inv + blurry[1] as f32 * inpaint_t).round() as u8,
                    (sharp[2] as f32 * inv + blurry[2] as f32 * inpaint_t).round() as u8,
                    sharp[3],
                ]),
            );
        }
    }
    output
}

fn layer_alpha_for_depth(
    depth_value: f32,
    band: usize,
    boundaries: &[f32],
    total_span: f32,
) -> f32 {
    let lo = boundaries[band];
    let hi = boundaries[band + 1];
    let band_span = (hi - lo).max(1e-5);
    let denom = boundaries.len().saturating_sub(2).max(1) as f32;
    let near_weight = 1.0 - band as f32 / denom;
    // Wider overlap (~0.5 of band_span) so adjacent layers significantly
    // co-occupy each pixel near band borders. With bipolar parallax pulling
    // layers apart, narrow overlap leaves visible seams; broad overlap turns
    // discrete bands into a continuous depth gradient at compositing time.
    let overlap = (total_span * 0.012).max(band_span * (0.50 + (1.0 - near_weight) * 0.20));
    let outer_lo = lo - overlap;
    let outer_hi = hi + overlap;
    if depth_value < outer_lo || depth_value > outer_hi {
        return 0.0;
    }
    if depth_value < lo {
        return smoothstep01((depth_value - outer_lo) / (lo - outer_lo).max(1e-5));
    }
    if depth_value > hi {
        return smoothstep01((outer_hi - depth_value) / (outer_hi - hi).max(1e-5));
    }
    1.0
}

fn smoothstep01(t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}
