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
            layer_count: 10,
            preview_max_edge: 2560,
            preview_target_pixels: 2560 * 1440,
            disparity_factor: 1.0,
            depth_scale: 1.0,
            debug_ply_decimation: 1.0,
            depth_fill_passes: 6,
            gamma: 1.55,
            blur_backdrop_scale: 0.0045,
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
    smooth_band_map(
        &mut band_map,
        preview_size.width as usize,
        preview_size.height as usize,
        request.config.layer_count,
    );
    for &band in &band_map {
        band_pixels[band as usize] += 1;
    }

    let backdrop = build_ghost_reduced_backdrop(
        &source_preview,
        &band_map,
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

fn build_ghost_reduced_backdrop(
    source: &RgbaImage,
    band_map: &[u8],
    layer_count: usize,
    blur_scale: f32,
) -> RgbaImage {
    let (width, height) = source.dimensions();
    let sigma = (width.min(height) as f32 * blur_scale).max(2.0);
    let blurred = imageops::blur(source, sigma);
    let mut output = RgbaImage::new(width, height);
    let denom = layer_count.saturating_sub(1).max(1) as f32;
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let idx = y as usize * width as usize + x as usize;
            let band = band_map[idx];
            let near_weight = 1.0 - band as f32 / denom;
            let mut max_band_delta = 0u8;
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
                    let nidx = sy as usize * width as usize + sx as usize;
                    max_band_delta = max_band_delta.max(band.abs_diff(band_map[nidx]));
                }
            }
            let near_mix = smoothstep01((near_weight - 0.18) / 0.64);
            let edge_mix = smoothstep01(max_band_delta as f32 / 2.0);
            let blur_mix = (near_mix * (0.28 + edge_mix * 0.56)).clamp(0.0, 0.84);
            let src = source.get_pixel(x as u32, y as u32).0;
            let blur = blurred.get_pixel(x as u32, y as u32).0;
            let inv = 1.0 - blur_mix;
            output.put_pixel(
                x as u32,
                y as u32,
                Rgba([
                    (src[0] as f32 * inv + blur[0] as f32 * blur_mix).round() as u8,
                    (src[1] as f32 * inv + blur[1] as f32 * blur_mix).round() as u8,
                    (src[2] as f32 * inv + blur[2] as f32 * blur_mix).round() as u8,
                    src[3],
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
    let overlap = (total_span * 0.008).max(band_span * (0.18 + (1.0 - near_weight) * 0.14));
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
