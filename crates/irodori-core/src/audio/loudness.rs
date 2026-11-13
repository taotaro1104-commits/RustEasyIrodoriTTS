//! ITU-R BS.1770-4 ラウドネス測定と正規化（audiotools `AudioSignal.normalize` 互換、モノラル）。
//!
//! audiotools の `Meter`（pyloudnorm 由来）と同じ手順:
//! K 特性（high-shelf +4dB/1500Hz, high-pass 38Hz）→ 400ms ブロック（75% オーバーラップ、末尾ゼロ埋め）
//! → 絶対ゲート -70 LUFS → 相対ゲート（-10 LU）→ 平均。0.5 秒未満は末尾をゼロ埋めして測る。

/// 測定下限（audiotools の `MIN_LOUDNESS`）。
pub const MIN_LOUDNESS: f64 = -70.0;

#[derive(Clone, Copy)]
struct Biquad {
    b: [f64; 3],
    a: [f64; 3],
}

impl Biquad {
    /// RBJ クックブックの high-shelf（pyloudnorm `IIRfilter`）。
    fn high_shelf(gain_db: f64, q: f64, fc: f64, rate: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * fc / rate;
        let alpha = w0.sin() / (2.0 * q);
        let (c, sq) = (w0.cos(), 2.0 * a.sqrt() * alpha);
        let b0 = a * ((a + 1.0) + (a - 1.0) * c + sq);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * c);
        let b2 = a * ((a + 1.0) + (a - 1.0) * c - sq);
        let a0 = (a + 1.0) - (a - 1.0) * c + sq;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * c);
        let a2 = (a + 1.0) - (a - 1.0) * c - sq;
        Self { b: [b0 / a0, b1 / a0, b2 / a0], a: [1.0, a1 / a0, a2 / a0] }
    }

    fn high_pass(q: f64, fc: f64, rate: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * fc / rate;
        let alpha = w0.sin() / (2.0 * q);
        let c = w0.cos();
        let a0 = 1.0 + alpha;
        Self { b: [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0], a: [1.0, -2.0 * c / a0, (1.0 - alpha) / a0] }
    }

    /// 初期状態 0 の直接形 I（`scipy.signal.lfilter` と同じ）。
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        x.iter()
            .map(|&x0| {
                let y0 = self.b[0] * x0 + self.b[1] * x1 + self.b[2] * x2 - self.a[1] * y1 - self.a[2] * y2;
                (x2, x1, y2, y1) = (x1, x0, y1, y0);
                y0
            })
            .collect()
    }
}

/// 統合ラウドネス（LUFS）。無音や絶対ゲートを通るブロックが無い場合は [`MIN_LOUDNESS`]。
pub fn integrated_loudness(samples: &[f32], rate: u32) -> f64 {
    let rate_f = rate as f64;
    let mut x: Vec<f64> = samples.iter().map(|&v| v as f64).collect();
    if (x.len() as f64) / rate_f < 0.5 {
        x.resize(x.len() + ((0.5 - x.len() as f64 / rate_f) * rate_f) as usize, 0.0);
    }
    let shelf = Biquad::high_shelf(4.0, 1.0 / 2f64.sqrt(), 1500.0, rate_f);
    let pass = Biquad::high_pass(0.5, 38.0, rate_f);
    let x = pass.apply(&shelf.apply(&x));

    let t_g = 0.400;
    let kernel = (t_g * rate_f) as usize;
    let stride = (t_g * rate_f * 0.25) as usize;
    let n_frames = (x.len().max(kernel) - kernel).div_ceil(stride) + 1;
    let norm = 1.0 / (t_g * rate_f);
    let z: Vec<f64> = (0..n_frames)
        .map(|i| {
            let start = i * stride;
            let end = (start + kernel).min(x.len());
            let sum: f64 = if start < x.len() { x[start..end].iter().map(|v| v * v).sum() } else { 0.0 };
            norm * sum
        })
        .collect();
    let lufs = |z: f64| -0.691 + 10.0 * z.log10();

    let abs_gated: Vec<f64> = z.iter().copied().filter(|&v| lufs(v) > MIN_LOUDNESS).collect();
    if abs_gated.is_empty() {
        return MIN_LOUDNESS;
    }
    let gamma_r = lufs(abs_gated.iter().sum::<f64>() / abs_gated.len() as f64) - 10.0;
    let both: Vec<f64> = z.iter().copied().filter(|&v| lufs(v) > MIN_LOUDNESS && lufs(v) > gamma_r).collect();
    if both.is_empty() {
        return MIN_LOUDNESS;
    }
    lufs(both.iter().sum::<f64>() / both.len() as f64).max(MIN_LOUDNESS)
}

/// `target_db` LUFS へゲインを掛け、ピークが 1.0 を超えるなら 1.0 に収める（audiotools の
/// `normalize` + `ensure_max_of_audio`）。
pub fn normalize_loudness(samples: &mut [f32], rate: u32, target_db: f64) {
    let loud = integrated_loudness(samples, rate);
    let gain = 10f64.powf((target_db - loud) / 20.0) as f32;
    for s in samples.iter_mut() {
        *s *= gain;
    }
    let peak = samples.iter().fold(0f32, |m, v| m.max(v.abs()));
    if peak > 1.0 {
        let k = 1.0 / peak;
        for s in samples.iter_mut() {
            *s *= k;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, amp: f32, secs: f32, rate: u32) -> Vec<f32> {
        (0..(secs * rate as f32) as usize)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn full_scale_997hz_sine_is_about_minus_3_lufs() {
        // 1kHz 付近の K 特性ゲインはほぼ +0.69dB で、BS.1770 の -0.691 と打ち消し合う → 平均パワー(dB)に一致する。
        let l = integrated_loudness(&sine(997.0, 1.0, 5.0, 48000), 48000);
        assert!((l - (-3.01)).abs() < 0.1, "{l}");
    }

    #[test]
    fn quiet_sine_loudness_scales_with_amplitude() {
        let l = integrated_loudness(&sine(997.0, 0.1, 5.0, 48000), 48000);
        assert!((l - (-23.01)).abs() < 0.1, "{l}");
    }

    #[test]
    fn silence_is_clamped_to_the_minimum_loudness() {
        assert_eq!(integrated_loudness(&vec![0.0; 48000], 48000), MIN_LOUDNESS);
    }

    #[test]
    fn signals_shorter_than_half_a_second_are_still_measured() {
        let l = integrated_loudness(&sine(997.0, 0.1, 0.2, 48000), 48000);
        assert!(l > MIN_LOUDNESS && l.is_finite(), "{l}");
    }

    #[test]
    fn normalize_reaches_the_target_loudness() {
        let mut x = sine(997.0, 0.1, 5.0, 48000);
        normalize_loudness(&mut x, 48000, -16.0);
        let l = integrated_loudness(&x, 48000);
        assert!((l - (-16.0)).abs() < 0.05, "{l}");
    }

    #[test]
    fn normalize_never_exceeds_full_scale() {
        // 目標が大きすぎてもピークは 1.0 に収める（ensure_max_of_audio）。
        let mut x = sine(997.0, 0.5, 5.0, 48000);
        normalize_loudness(&mut x, 48000, 0.0);
        let peak = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(peak <= 1.0 + 1e-6 && peak > 0.99, "{peak}");
    }

    #[test]
    fn real_audio_matches_audiotools_golden() {
        let Some(_) = crate::golden::dir() else {
            eprintln!("skip: IRODORI_GOLDEN_DIR not set");
            return;
        };
        let g = crate::golden::load("loudness").unwrap();
        let wav: Vec<f32> = g["wav"].flatten_all().unwrap().to_vec1().unwrap();
        let want: f32 = g["loudness"].flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        let got = integrated_loudness(&wav, 48000);
        assert!((got - want as f64).abs() < 0.05, "got {got}, want {want}");
        let mut n = wav.clone();
        normalize_loudness(&mut n, 48000, -16.0);
        let want_n: Vec<f32> = g["normalized"].flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = n.iter().zip(&want_n).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(max_diff < 5e-3, "max diff {max_diff}");
    }
}
