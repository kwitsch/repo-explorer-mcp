//! Temperature selection/clamp and the 2-logit softmax used by the judge head.

/// D6 step 7: `temperature_by_options["choice:2"]` if present, else
/// `temperature[0]`; a non-finite value becomes 1.0; clamped to `[0.5, 5.0]`.
pub fn select_temperature(by_options_choice2: Option<f32>, temperature0: f32) -> f32 {
    let t = by_options_choice2.unwrap_or(temperature0);
    let t = if t.is_finite() { t } else { 1.0 };
    t.clamp(0.5, 5.0)
}

/// Numerically stable softmax over two temperature-scaled logits.
pub fn softmax2(logits: [f32; 2], t: f32) -> [f32; 2] {
    let a = logits[0] / t;
    let b = logits[1] / t;
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    let s = ea + eb;
    [ea / s, eb / s]
}

/// P(relevant) as per-mille, rounded and clamped to `0..=1000`.
pub fn permille(p: f32) -> u32 {
    ((p * 1000.0).round() as i64).clamp(0, 1000) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_beats_array_and_clamps() {
        assert_eq!(select_temperature(Some(2.0), 3.0), 2.0);
        assert_eq!(select_temperature(None, 3.0), 3.0);
        assert_eq!(select_temperature(Some(0.1), 3.0), 0.5);
        assert_eq!(select_temperature(Some(9.0), 3.0), 5.0);
        assert_eq!(select_temperature(Some(f32::NAN), 3.0), 1.0);
    }

    #[test]
    fn softmax_of_zeros_is_half() {
        let p = softmax2([0.0, 0.0], 1.0);
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn permille_rounds_and_clamps() {
        assert_eq!(permille(0.9995), 1000);
        assert_eq!(permille(0.0), 0);
        assert_eq!(permille(2.0), 1000);
        assert_eq!(permille(0.5004), 500);
    }
}
