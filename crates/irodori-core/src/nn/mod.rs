//! DiT の基本部品（上流 `model.py` 37–120, 428–437 行の移植）。
pub mod adaln;
pub mod qlinear;
pub mod norm;
pub mod rope;
pub mod swiglu;
pub mod timestep;

#[cfg(test)]
mod golden_tests {
    use super::{adaln::LowRankAdaLn, norm::RmsNorm, rope::*, swiglu::SwiGlu, timestep::timestep_embedding};
    use crate::golden::{self, assert_close};
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    #[test]
    fn nn_basics_match_python_golden() {
        if golden::dir().is_none() {
            eprintln!("skip: IRODORI_GOLDEN_DIR not set");
            return;
        }
        let g = golden::load("nn_basics").unwrap();
        let vb = VarBuilder::from_tensors(g.clone(), DType::F32, &Device::Cpu);

        let f = FreqsCis::new(64, 50, 10000.0, &Device::Cpu).unwrap();
        assert_close(&apply_rotary_emb(&g["rope.x"], &f).unwrap(), &g["rope.y"], 1e-5, 1e-5).unwrap();
        assert_close(&apply_rotary_half(&g["rope.x"], &f).unwrap(), &g["rope.half_y"], 1e-5, 1e-5).unwrap();

        let norm = RmsNorm::new(&[4, 64], 1e-5, vb.pp("rms")).unwrap();
        assert_close(&norm.forward(&g["rope.x"]).unwrap(), &g["rms.y"], 1e-5, 1e-5).unwrap();

        let ada = LowRankAdaLn::new(32, 8, 1e-5, vb.pp("ada")).unwrap();
        let (y, gate) = ada.forward(&g["ada.x"], &g["ada.cond"]).unwrap();
        assert_close(&y, &g["ada.y"], 1e-5, 1e-5).unwrap();
        assert_close(&gate, &g["ada.gate"], 1e-5, 1e-5).unwrap();

        let swi = SwiGlu::new(16, 40, vb.pp("swi")).unwrap();
        assert_close(&swi.forward(&g["swi.x"]).unwrap(), &g["swi.y"], 1e-5, 1e-5).unwrap();

        assert_close(&timestep_embedding(&g["ts.t"], 512).unwrap(), &g["ts.y"], 1e-4, 1e-4).unwrap();
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    pub fn vb(items: Vec<(&str, Tensor)>) -> VarBuilder<'static> {
        let map: HashMap<String, Tensor> =
            items.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
    }

    pub fn t(v: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape, &Device::Cpu).unwrap()
    }

    /// 名前から決まる決定的な擬似乱数重み。名前に `norm` を含むものは 1 付近。
    pub fn fake_tensor(name: &str, shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let seed = name.bytes().map(|b| b as f32).sum::<f32>();
        let norm = name.contains("norm");
        let v: Vec<f32> = (0..n)
            .map(|k| {
                let s = (0.83 * k as f32 + seed).sin();
                if norm { 1.0 + 0.1 * s } else { 0.25 * s }
            })
            .collect();
        Tensor::from_vec(v, shape.to_vec(), &Device::Cpu).unwrap()
    }

    /// `(テンソル名, 形状)` の一覧から合成 VarBuilder を作る。
    pub fn vb_from_spec(spec: Vec<(String, Vec<usize>)>) -> VarBuilder<'static> {
        vb_from_spec_dtype(spec, DType::F32)
    }

    /// 取得時に `dtype` へ変換する合成 VarBuilder（bf16 重みのテスト用）。
    pub fn vb_from_spec_dtype(spec: Vec<(String, Vec<usize>)>, dtype: DType) -> VarBuilder<'static> {
        let map: HashMap<String, Tensor> =
            spec.into_iter().map(|(n, s)| { let t = fake_tensor(&n, &s); (n, t) }).collect();
        VarBuilder::from_tensors(map, dtype, &Device::Cpu)
    }

    pub fn zeros(r: usize, c: usize) -> Tensor {
        Tensor::zeros((r, c), DType::F32, &Device::Cpu).unwrap()
    }
}
