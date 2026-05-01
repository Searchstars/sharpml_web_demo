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
    /// Depth pivot for bipolar parallax. Layers nearer than this move with the
    /// gesture; farther layers move slightly against it.
    #[serde(default = "default_focal_near_weight")]
    pub focal_near_weight: f32,
    /// Far side is intentionally weaker than near side to avoid landscape
    /// layers tearing apart at large angles.
    #[serde(default = "default_far_gain_factor")]
    pub far_gain_factor: f32,
    /// Extra response applied below the saturation knee. This makes small and
    /// medium drags visibly parallax without increasing worst-case travel.
    #[serde(default = "default_mid_tilt_boost")]
    pub mid_tilt_boost: f32,
    /// Effective tilt magnitude where soft saturation starts.
    #[serde(default = "default_response_knee")]
    pub response_knee: f32,
    /// Maximum effective tilt magnitude after the soft knee.
    #[serde(default = "default_response_cap")]
    pub response_cap: f32,
}

fn default_focal_near_weight() -> f32 {
    0.52
}

fn default_far_gain_factor() -> f32 {
    0.25
}

fn default_mid_tilt_boost() -> f32 {
    0.55
}

fn default_response_knee() -> f32 {
    0.64
}

fn default_response_cap() -> f32 {
    0.84
}

impl Default for MotionTuning {
    fn default() -> Self {
        Self {
            travel_base: 0.009,
            travel_gain: 0.125,
            rotation_gain: 0.24,
            axis_gain_x: 1.18,
            axis_gain_y: 0.94,
            backdrop_weight: -0.055,
            layer_weight_base: 0.0,
            layer_weight_gain: 0.22,
            layer_weight_power: 1.15,
            scale_base: 0.004,
            scale_gain: 0.012,
            focal_near_weight: default_focal_near_weight(),
            far_gain_factor: default_far_gain_factor(),
            mid_tilt_boost: default_mid_tilt_boost(),
            response_knee: default_response_knee(),
            response_cap: default_response_cap(),
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
    let (nx_eff, ny_eff) = responsive_tilt(nx, ny, tuning);
    nx_eff * ny_eff * tilt_range_deg * tuning.rotation_gain
}

pub fn layer_motion_for_weight(
    is_backdrop: bool,
    near_weight: f32,
    tuning: &MotionTuning,
) -> LayerMotion {
    let tx_weight = if is_backdrop {
        tuning.backdrop_weight
    } else {
        let focal = tuning.focal_near_weight.clamp(1e-3, 1.0 - 1e-3);
        let offset = near_weight - focal;
        let normalized = if offset >= 0.0 {
            offset / (1.0 - focal)
        } else {
            offset / focal
        };
        let signed = normalized
            .abs()
            .powf(tuning.layer_weight_power)
            .copysign(normalized);
        let side_gain = if normalized >= 0.0 {
            1.0
        } else {
            tuning.far_gain_factor
        };
        tuning.layer_weight_base + signed * side_gain * tuning.layer_weight_gain
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

pub fn responsive_tilt(nx: f32, ny: f32, tuning: &MotionTuning) -> (f32, f32) {
    let raw_mag = nx.hypot(ny);
    if raw_mag <= 1e-6 {
        return (nx, ny);
    }
    let boosted = raw_mag.clamp(0.0, 1.0) * (1.0 + tuning.mid_tilt_boost * (1.0 - raw_mag));
    let knee = tuning.response_knee.clamp(1e-3, 1.0);
    let cap = tuning.response_cap.clamp(knee + 1e-3, 1.0);
    let effective_mag = if boosted <= knee {
        boosted
    } else {
        let tail = cap - knee;
        knee + tail * (1.0 - (-(boosted - knee) / tail).exp())
    };
    let ratio = effective_mag / raw_mag;
    (nx * ratio, ny * ratio)
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
    let (nx_eff, ny_eff) = responsive_tilt(nx, ny, tuning);
    let motion_mag = nx_eff.hypot(ny_eff).min(1.0);
    let travel =
        display_width.min(display_height) * (tuning.travel_base + parallax * tuning.travel_gain);
    let tx_px = -nx_eff * travel * motion.tx_weight * tuning.axis_gain_x;
    let ty_px = ny_eff * travel * motion.ty_weight * tuning.axis_gain_y;
    let scale = 1.0 + motion_mag * (motion.scale_base + motion.scale_gain);
    LayerTransform {
        tx_px,
        ty_px,
        scale,
    }
}
