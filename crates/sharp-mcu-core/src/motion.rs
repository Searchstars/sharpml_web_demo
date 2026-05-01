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
    /// Normalized near_weight (0..1) of the depth that should stay still while
    /// the camera tilts. Layers nearer than this slide with the gesture, layers
    /// farther slide opposite. Acts as the focal/pivot plane.
    #[serde(default = "default_focal_near_weight")]
    pub focal_near_weight: f32,
    /// Magnitude multiplier for the far (negative) side of the bipolar curve.
    /// Real parallax shrinks at distance, so far layers usually move a bit less
    /// than the symmetric near side.
    #[serde(default = "default_far_gain_factor")]
    pub far_gain_factor: f32,
    /// Tilt-magnitude saturation (>=0). Applied as `tanh(mag·k)/k` so:
    ///   * k = 0     → linear (no saturation, original behavior)
    ///   * k = 1.2   → full-tilt motion ≈ 70 % of linear (moderate cap)
    ///   * k = 2     → full-tilt motion ≈ 48 % (aggressive cap)
    /// Small tilts retain the linear response (`f'(0) = 1`); only extreme
    /// magnitudes get compressed. This caps the worst-case per-layer
    /// translation without affecting feel at moderate tilts, which directly
    /// limits how far adjacent bands can drift apart and how much ghost the
    /// flat-2D-stack approximation can produce.
    #[serde(default = "default_motion_saturation")]
    pub motion_saturation: f32,
}

fn default_focal_near_weight() -> f32 {
    0.45
}

fn default_far_gain_factor() -> f32 {
    0.8
}

fn default_motion_saturation() -> f32 {
    1.3
}

impl Default for MotionTuning {
    fn default() -> Self {
        Self {
            // Bipolar weights span ~[-0.27, +0.34] (with the defaults below),
            // about 3× the differential of the old single-sided 0.04..0.23
            // curve, so we keep travel_base/gain modest — small parallax still
            // produces a real depth-anchored shift.
            travel_base: 0.010,
            travel_gain: 0.135,
            rotation_gain: 0.24,
            axis_gain_x: 1.18,
            axis_gain_y: 0.94,
            // Backdrop is "the depth behind everything"; in bipolar parallax it
            // slides opposite to near layers, like the deep wall in a diorama.
            backdrop_weight: -0.20,
            layer_weight_base: 0.0,
            layer_weight_gain: 0.34,
            layer_weight_power: 1.10,
            scale_base: 0.004,
            scale_gain: 0.014,
            focal_near_weight: default_focal_near_weight(),
            far_gain_factor: default_far_gain_factor(),
            motion_saturation: default_motion_saturation(),
        }
    }
}

/// Squash an input tilt magnitude through a tanh-based saturation. Output is
/// almost linear at small `mag` and asymptotes toward `1/k` as `mag → ∞`.
/// At `k = 1.3`, `saturate_magnitude(1.0) ≈ 0.66` — i.e. full tilt yields
/// about two-thirds of linear motion, while a quarter-tilt input is barely
/// affected (`saturate_magnitude(0.25) ≈ 0.245`).
pub fn saturate_magnitude(mag: f32, k: f32) -> f32 {
    let m = mag.clamp(0.0, 1.0);
    if k <= 1e-3 { m } else { (m * k).tanh() / k }
}

/// Apply tilt saturation to a 2D input direction, preserving direction and
/// only shrinking magnitude. Returns the effective `(nx, ny)` for downstream
/// transform/rotation math.
pub fn saturated_tilt(nx: f32, ny: f32, tuning: &MotionTuning) -> (f32, f32) {
    let raw_mag = nx.hypot(ny);
    if raw_mag <= 1e-6 {
        return (nx, ny);
    }
    let target_mag = saturate_magnitude(raw_mag, tuning.motion_saturation);
    let ratio = target_mag / raw_mag;
    (nx * ratio, ny * ratio)
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
    let (nx_eff, ny_eff) = saturated_tilt(nx, ny, tuning);
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
        // Bipolar parallax: layers nearer than the focal plane slide WITH the
        // tilt direction, layers farther slide AGAINST it. Pivoting around a
        // mid-depth focal plane is what gives a real parallax sensation, vs.
        // the previous unipolar curve where everything just slid together at
        // different speeds.
        let focal = tuning.focal_near_weight.clamp(1e-3, 1.0 - 1e-3);
        let offset = near_weight - focal;
        let normalized = if offset >= 0.0 {
            // Near side: [focal, 1] → [0, 1]
            offset / (1.0 - focal)
        } else {
            // Far side: [0, focal] → [-1, 0]
            offset / focal
        };
        let mag = normalized.abs().powf(tuning.layer_weight_power);
        let signed = mag.copysign(normalized);
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
        // Symmetric scale boost — layers further from focal (in either
        // direction) get a touch more "depth-zoom" with motion magnitude.
        let dist = (near_weight - tuning.focal_near_weight).abs();
        dist * tuning.scale_gain
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
    let (nx_eff, ny_eff) = saturated_tilt(nx, ny, tuning);
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
