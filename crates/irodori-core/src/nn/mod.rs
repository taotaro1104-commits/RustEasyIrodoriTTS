//! DiT の基本部品（上流 `model.py` 37–120, 428–437 行の移植）。
pub mod adaln;
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

    pub fn zeros(r: usize, c: usize) -> Tensor {
        Tensor::zeros((r, c), DType::F32, &Device::Cpu).unwrap()
    }
}
