use std::{
    cmp::Ordering,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::Result;

use crate::onnx::SharpOutputs;

pub fn write_debug_ply(
    outputs: &SharpOutputs,
    output_path: &Path,
    focal_length_px: f32,
    image_shape: (u32, u32),
    decimation: f32,
    depth_scale: f32,
) -> Result<()> {
    let mut selected = (0..outputs.mean_vectors.nrows()).collect::<Vec<_>>();
    if (0.0..1.0).contains(&decimation) {
        selected.sort_by(|&a, &b| {
            let ia = outputs.singular_values[[a, 0]]
                * outputs.singular_values[[a, 1]]
                * outputs.singular_values[[a, 2]]
                * outputs.opacities[a];
            let ib = outputs.singular_values[[b, 0]]
                * outputs.singular_values[[b, 1]]
                * outputs.singular_values[[b, 2]]
                * outputs.opacities[b];
            ib.partial_cmp(&ia).unwrap_or(Ordering::Equal)
        });
        let keep = ((selected.len() as f32) * decimation).max(1.0) as usize;
        selected.truncate(keep);
        selected.sort_unstable();
    }

    let z_min = selected
        .iter()
        .map(|&idx| outputs.mean_vectors[[idx, 2]])
        .filter(|z| z.is_finite() && *z > 1e-5)
        .fold(f32::INFINITY, f32::min);
    let (image_width, image_height) = image_shape;
    let focal_ndc = 2.0 * focal_length_px / image_width as f32;
    let scale_factor = 1.0 / (z_min * focal_ndc);

    let header = build_header(selected.len());
    let mut writer = BufWriter::new(File::create(output_path)?);
    writer.write_all(header.as_bytes())?;

    for &idx in &selected {
        let z_raw = outputs.mean_vectors[[idx, 2]];
        let z_normalized = normalize_depth(z_raw, z_min, depth_scale, &outputs.mean_vectors);
        let x = outputs.mean_vectors[[idx, 0]] * scale_factor;
        let y = outputs.mean_vectors[[idx, 1]] * scale_factor;
        let color = [
            rgb_to_sh(linear_to_srgb(outputs.colors[[idx, 0]])),
            rgb_to_sh(linear_to_srgb(outputs.colors[[idx, 1]])),
            rgb_to_sh(linear_to_srgb(outputs.colors[[idx, 2]])),
        ];
        let opacity = inverse_sigmoid(outputs.opacities[idx]);
        let scale0 = (outputs.singular_values[[idx, 0]] * scale_factor)
            .max(1e-10)
            .ln();
        let scale1 = (outputs.singular_values[[idx, 1]] * scale_factor)
            .max(1e-10)
            .ln();
        let scale2 = (outputs.singular_values[[idx, 2]] / z_min).max(1e-10).ln();

        for value in [
            x,
            y,
            z_normalized,
            color[0],
            color[1],
            color[2],
            opacity,
            scale0,
            scale1,
            scale2,
            outputs.quaternions[[idx, 0]],
            outputs.quaternions[[idx, 1]],
            outputs.quaternions[[idx, 2]],
            outputs.quaternions[[idx, 3]],
        ] {
            writer.write_all(&value.to_le_bytes())?;
        }
    }

    for value in [
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ] {
        writer.write_all(&value.to_le_bytes())?;
    }
    for value in [
        focal_length_px,
        0.0,
        image_width as f32 / 2.0,
        0.0,
        focal_length_px,
        image_height as f32 / 2.0,
        0.0,
        0.0,
        1.0,
    ] {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.write_all(&image_width.to_le_bytes())?;
    writer.write_all(&image_height.to_le_bytes())?;
    writer.write_all(&(1i32).to_le_bytes())?;
    writer.write_all(&(selected.len() as i32).to_le_bytes())?;

    let mut disparities = selected
        .iter()
        .map(|&idx| 1.0 / outputs.mean_vectors[[idx, 2]].max(1e-6))
        .collect::<Vec<_>>();
    disparities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let disparity_10 = percentile(&disparities, 0.10);
    let disparity_90 = percentile(&disparities, 0.90);
    writer.write_all(&disparity_10.to_le_bytes())?;
    writer.write_all(&disparity_90.to_le_bytes())?;
    writer.write_all(&[1u8])?;
    writer.write_all(&[1u8, 5u8, 0u8])?;
    writer.flush()?;
    Ok(())
}

fn build_header(vertex_count: usize) -> String {
    let mut header = String::new();
    header.push_str("ply\nformat binary_little_endian 1.0\n");
    header.push_str(&format!("element vertex {vertex_count}\n"));
    for name in [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ] {
        header.push_str(&format!("property float {name}\n"));
    }
    header.push_str("element extrinsic 1\n");
    for idx in 0..16 {
        header.push_str(&format!("property float extrinsic_{idx}\n"));
    }
    header.push_str("element intrinsic 1\n");
    for idx in 0..9 {
        header.push_str(&format!("property float intrinsic_{idx}\n"));
    }
    header
        .push_str("element image_size 1\nproperty uint image_size_0\nproperty uint image_size_1\n");
    header.push_str("element frame 1\nproperty int frame_0\nproperty int frame_1\n");
    header
        .push_str("element disparity 1\nproperty float disparity_0\nproperty float disparity_1\n");
    header.push_str("element color_space 1\nproperty uchar color_space\n");
    header.push_str("element version 1\nproperty uchar version_0\nproperty uchar version_1\nproperty uchar version_2\n");
    header.push_str("end_header\n");
    header
}

fn normalize_depth(
    z_raw: f32,
    z_min: f32,
    depth_scale: f32,
    mean_vectors: &ndarray::Array2<f32>,
) -> f32 {
    let z_normalized = z_raw / z_min;
    if (depth_scale - 1.0).abs() < f32::EPSILON {
        return z_normalized;
    }
    let mut z_values = mean_vectors.column(2).iter().copied().collect::<Vec<_>>();
    z_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let z_median = z_values[z_values.len() / 2] / z_min;
    z_median + (z_normalized - z_median) * depth_scale
}

fn percentile(values: &[f32], t: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = ((values.len() as f32 - 1.0) * t).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

fn rgb_to_sh(rgb: f32) -> f32 {
    let coeff_degree0 = 1.0 / (4.0 * std::f32::consts::PI).sqrt();
    (rgb - 0.5) / coeff_degree0
}

fn inverse_sigmoid(x: f32) -> f32 {
    let x = x.clamp(1e-6, 1.0 - 1e-6);
    (x / (1.0 - x)).ln()
}
