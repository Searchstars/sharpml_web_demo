use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionTuning {
    pub travel_base: f32,
    pub travel_gain: f32,
    pub rotation_gain: f32,
    pub axis_gain_x: f32,
    pub axis_gain_y: f32,
    pub backdrop_weight: f32,
    pub layer_weight_base: f32,
    pub layer_weight_gain: f32,
    pub layer_weight_power: f32,
    pub scale_base: f32,
    pub scale_gain: f32,
}

impl Default for MotionTuning {
    fn default() -> Self {
        Self {
            travel_base: 0.008,
            travel_gain: 0.105,
            rotation_gain: 0.24,
            axis_gain_x: 1.18,
            axis_gain_y: 0.94,
            backdrop_weight: 0.018,
            layer_weight_base: 0.04,
            layer_weight_gain: 0.19,
            layer_weight_power: 1.55,
            scale_base: 0.004,
            scale_gain: 0.010,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMotion {
    pub tx_weight: f32,
    pub ty_weight: f32,
    pub scale_base: f32,
    pub scale_gain: f32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct LayerTransform {
    pub tx_px: f32,
    pub ty_px: f32,
    pub scale: f32,
}

pub fn stage_rotation_deg(nx: f32, ny: f32, tilt_range_deg: f32, tuning: &MotionTuning) -> f32 {
    nx * ny * tilt_range_deg * tuning.rotation_gain
}

pub fn layer_motion_for_weight(
    is_backdrop: bool,
    near_weight: f32,
    tuning: &MotionTuning,
) -> LayerMotion {
    let tx_weight = if is_backdrop {
        tuning.backdrop_weight
    } else {
        tuning.layer_weight_base
            + near_weight.powf(tuning.layer_weight_power) * tuning.layer_weight_gain
    };
    let scale_base = if is_backdrop { 0.0 } else { tuning.scale_base };
    let scale_gain = if is_backdrop {
        0.0
    } else {
        near_weight * tuning.scale_gain
    };
    LayerMotion {
        tx_weight,
        ty_weight: tx_weight,
        scale_base,
        scale_gain,
    }
}

pub fn compute_layer_transform(
    nx: f32,
    ny: f32,
    display_width: f32,
    display_height: f32,
    parallax: f32,
    motion: &LayerMotion,
    tuning: &MotionTuning,
) -> LayerTransform {
    let motion_mag = nx.hypot(ny).min(1.0);
    let travel =
        display_width.min(display_height) * (tuning.travel_base + parallax * tuning.travel_gain);
    let tx_px = -nx * travel * motion.tx_weight * tuning.axis_gain_x;
    let ty_px = ny * travel * motion.ty_weight * tuning.axis_gain_y;
    let scale = 1.0 + motion_mag * (motion.scale_base + motion.scale_gain);
    LayerTransform {
        tx_px,
        ty_px,
        scale,
    }
}
