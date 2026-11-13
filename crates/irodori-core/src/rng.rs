//! シード付き正規乱数（SplitMix64 + Box-Muller）。torch の乱数列とは一致しない（ゴールデンはノイズを外から注入する）。

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `(0, 1]` の一様乱数（`ln(0)` を避ける）。
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }
}

/// 標準正規分布の乱数を `n` 個。同じ `seed` なら先頭から同じ列（長さに依存しない）。
pub fn normal_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix64(seed);
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (rng.next_unit(), rng.next_unit());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    out.truncate(n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_sequence() {
        assert_eq!(normal_vec(42, 17), normal_vec(42, 17));
    }

    #[test]
    fn different_seeds_give_different_sequences() {
        assert_ne!(normal_vec(1, 16), normal_vec(2, 16));
    }

    #[test]
    fn prefix_is_stable_across_lengths() {
        // 長さを変えても先頭が変わらない（候補数や長さの違いで同じ seed の音が変わらないように）。
        let a = normal_vec(7, 10);
        let b = normal_vec(7, 25);
        assert_eq!(&a[..], &b[..10]);
    }

    #[test]
    fn odd_lengths_are_supported() {
        assert_eq!(normal_vec(3, 5).len(), 5);
        assert!(normal_vec(3, 0).is_empty());
    }

    #[test]
    fn samples_are_standard_normal() {
        let v = normal_vec(123, 200_000);
        let n = v.len() as f64;
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
        let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.01, "{mean}");
        assert!((var - 1.0).abs() < 0.02, "{var}");
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
