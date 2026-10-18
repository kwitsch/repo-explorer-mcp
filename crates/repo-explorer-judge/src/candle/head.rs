//! The Laya decision head: a type embedding, `head_layers` pre-norm
//! `TransformerEncoderLayer`s (norm_first, relu, eps 1e-5), and the scorer MLP.
//! Weights are addressed by their PyTorch `state_dict` names.
// Wired into `LoadedModel` in a later task; unused in the lib target until then.
#![allow(dead_code)]

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

struct HeadLayer {
    in_proj: Linear,  // D -> 3D (fused q,k,v)
    out_proj: Linear, // D -> D
    linear1: Linear,  // D -> F
    linear2: Linear,  // F -> D
    norm1: LayerNorm,
    norm2: LayerNorm,
    n_heads: usize,
}

pub(crate) struct DecisionHead {
    type_emb0: Tensor, // [D], row 0 = choice
    layers: Vec<HeadLayer>,
    scorer_ln: LayerNorm,
    scorer_lin1: Linear, // D -> D
    scorer_lin2: Linear, // D -> 1
    hidden: usize,
}

const LN_EPS: f64 = 1e-5;

fn linear(vb: &VarBuilder, w: &str, b: &str, out_dim: usize, in_dim: usize) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), w)?;
    let bias = vb.get(out_dim, b)?;
    Ok(Linear::new(weight, Some(bias)))
}

fn layer_norm(vb: &VarBuilder, w: &str, b: &str, dim: usize) -> Result<LayerNorm> {
    let weight = vb.get(dim, w)?;
    let bias = vb.get(dim, b)?;
    Ok(LayerNorm::new(weight, bias, LN_EPS))
}

impl DecisionHead {
    pub(crate) fn load(vb: &VarBuilder, hidden: usize, head_layers: usize) -> Result<DecisionHead> {
        let d = hidden;
        let f = 4 * d;
        let n_heads = std::cmp::max(1, d / 64);
        let type_emb = vb.get((3, d), "type_emb.weight")?;
        let type_emb0 = type_emb.get(0)?.contiguous()?;
        let mut layers = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            let p = format!("head.layers.{i}");
            let in_proj = {
                let weight = vb.get((3 * d, d), &format!("{p}.self_attn.in_proj_weight"))?;
                let bias = vb.get(3 * d, &format!("{p}.self_attn.in_proj_bias"))?;
                Linear::new(weight, Some(bias))
            };
            let out_proj = linear(
                vb,
                &format!("{p}.self_attn.out_proj.weight"),
                &format!("{p}.self_attn.out_proj.bias"),
                d,
                d,
            )?;
            let linear1 = linear(
                vb,
                &format!("{p}.linear1.weight"),
                &format!("{p}.linear1.bias"),
                f,
                d,
            )?;
            let linear2 = linear(
                vb,
                &format!("{p}.linear2.weight"),
                &format!("{p}.linear2.bias"),
                d,
                f,
            )?;
            let norm1 = layer_norm(
                vb,
                &format!("{p}.norm1.weight"),
                &format!("{p}.norm1.bias"),
                d,
            )?;
            let norm2 = layer_norm(
                vb,
                &format!("{p}.norm2.weight"),
                &format!("{p}.norm2.bias"),
                d,
            )?;
            layers.push(HeadLayer {
                in_proj,
                out_proj,
                linear1,
                linear2,
                norm1,
                norm2,
                n_heads,
            });
        }
        let scorer_ln = layer_norm(vb, "scorer.0.weight", "scorer.0.bias", d)?;
        let scorer_lin1 = linear(vb, "scorer.1.weight", "scorer.1.bias", d, d)?;
        let scorer_lin2 = linear(vb, "scorer.3.weight", "scorer.3.bias", 1, d)?;
        Ok(DecisionHead {
            type_emb0,
            layers,
            scorer_ln,
            scorer_lin1,
            scorer_lin2,
            hidden: d,
        })
    }

    /// `h`: [B, L, D] encoder output; `attn_mask`: [B, L] u32 (1 keep, 0 pad);
    /// `marker_pos`: [B, 2] u32. Returns pre-temperature logits [B, 2].
    pub(crate) fn forward(
        &self,
        h: &Tensor,
        attn_mask: &Tensor,
        marker_pos: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, d) = h.dims3()?;
        debug_assert_eq!(d, self.hidden);
        let mut h = h.broadcast_add(&self.type_emb0.reshape((1, 1, d))?)?;

        // Additive key-padding bias [B, 1, 1, L]: 0 where keep, -1e30 where pad.
        let mask_f = attn_mask.to_dtype(DType::F32)?;
        let bias = (mask_f.affine(1.0, -1.0)? * 1e30f64)?.reshape((b, 1, 1, l))?;

        for layer in &self.layers {
            h = layer.forward(&h, &bias)?;
        }

        // Gather the two marker rows: [B, 2, D].
        let idx = marker_pos
            .unsqueeze(2)?
            .broadcast_as((b, 2, d))?
            .contiguous()?;
        let m = h.gather(&idx, 1)?;

        // Scorer: LayerNorm -> Linear -> gelu_erf -> Linear -> [B, 2].
        let x = self.scorer_ln.forward(&m)?;
        let x = self.scorer_lin1.forward(&x)?;
        let x = x.gelu_erf()?;
        let x = self.scorer_lin2.forward(&x)?; // [B, 2, 1]
        x.squeeze(2)
    }
}

impl HeadLayer {
    fn forward(&self, h: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let (b, l, d) = h.dims3()?;
        let hd = d / self.n_heads;
        let x = self.norm1.forward(h)?;
        let qkv = self.in_proj.forward(&x)?; // [B, L, 3D]
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;
        let shape = (b, l, self.n_heads, hd);
        let q = q.reshape(shape)?.transpose(1, 2)?.contiguous()?; // [B, H, L, hd]
        let k = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (hd as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?; // [B, H, L, L]
        let scores = scores.broadcast_add(bias)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = attn.matmul(&v)?; // [B, H, L, hd]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, l, d))?;
        let sa = self.out_proj.forward(&ctx)?;
        let h = (h + sa)?;
        let y = self.norm2.forward(&h)?;
        let y = self.linear1.forward(&y)?.relu()?;
        let y = self.linear2.forward(&y)?;
        h + y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};
    use std::collections::HashMap;

    const D: usize = 64;
    const LAYERS: usize = 2;

    fn build() -> (DecisionHead, Device) {
        let dev = Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let mut put = |name: &str, dims: &[usize]| {
            let n: usize = dims.iter().product();
            // deterministic pseudo-random in [-0.1, 0.1]
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 12.9898).sin() * 43758.547).fract() * 0.2 - 0.1)
                .collect();
            t.insert(
                name.to_string(),
                Tensor::from_vec(data, dims.to_vec(), &dev).unwrap(),
            );
        };
        put("type_emb.weight", &[3, D]);
        let f = 4 * D;
        for i in 0..LAYERS {
            let p = format!("head.layers.{i}");
            put(&format!("{p}.self_attn.in_proj_weight"), &[3 * D, D]);
            put(&format!("{p}.self_attn.in_proj_bias"), &[3 * D]);
            put(&format!("{p}.self_attn.out_proj.weight"), &[D, D]);
            put(&format!("{p}.self_attn.out_proj.bias"), &[D]);
            put(&format!("{p}.linear1.weight"), &[f, D]);
            put(&format!("{p}.linear1.bias"), &[f]);
            put(&format!("{p}.linear2.weight"), &[D, f]);
            put(&format!("{p}.linear2.bias"), &[D]);
            put(&format!("{p}.norm1.weight"), &[D]);
            put(&format!("{p}.norm1.bias"), &[D]);
            put(&format!("{p}.norm2.weight"), &[D]);
            put(&format!("{p}.norm2.bias"), &[D]);
        }
        put("scorer.0.weight", &[D]);
        put("scorer.0.bias", &[D]);
        put("scorer.1.weight", &[D, D]);
        put("scorer.1.bias", &[D]);
        put("scorer.3.weight", &[1, D]);
        put("scorer.3.bias", &[1]);
        let vb = VarBuilder::from_tensors(t, DType::F32, &dev);
        (DecisionHead::load(&vb, D, LAYERS).unwrap(), dev)
    }

    fn seq(dev: &Device, len: usize, keep: usize, m0: u32, m1: u32) -> (Tensor, Tensor, Tensor) {
        let h: Vec<f32> = (0..len * D).map(|i| (i as f32 * 7.1).sin() * 0.5).collect();
        let h = Tensor::from_vec(h, (1, len, D), dev).unwrap();
        let mask: Vec<u32> = (0..len).map(|i| if i < keep { 1 } else { 0 }).collect();
        let mask = Tensor::from_vec(mask, (1, len), dev).unwrap();
        let markers = Tensor::from_vec(vec![m0, m1], (1, 2), dev).unwrap();
        (h, mask, markers)
    }

    #[test]
    fn output_shape_is_b_by_2() {
        let (head, dev) = build();
        let (h, mask, markers) = seq(&dev, 12, 12, 3, 7);
        let out = head.forward(&h, &mask, &markers).unwrap();
        assert_eq!(out.dims(), &[1, 2]);
    }

    #[test]
    fn padding_invariance() {
        let (head, dev) = build();
        // Sequence of real length 10, markers at 3 and 7.
        let (h_short, mask_short, markers_short) = seq(&dev, 10, 10, 3, 7);
        let logits_short = head.forward(&h_short, &mask_short, &markers_short).unwrap();
        // Same content padded to length 16 (last 6 are pad).
        let mut h_data: Vec<f32> = h_short.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        h_data.extend(std::iter::repeat_n(0.0f32, 6 * D));
        let h_pad = Tensor::from_vec(h_data, (1, 16, D), &dev).unwrap();
        let (_, mask_pad, markers_pad) = seq(&dev, 16, 10, 3, 7);
        let logits_pad = head.forward(&h_pad, &mask_pad, &markers_pad).unwrap();
        let a = logits_short.to_vec2::<f32>().unwrap();
        let b = logits_pad.to_vec2::<f32>().unwrap();
        assert!((a[0][0] - b[0][0]).abs() < 1e-5, "{a:?} vs {b:?}");
        assert!((a[0][1] - b[0][1]).abs() < 1e-5, "{a:?} vs {b:?}");
    }

    #[test]
    fn marker_gather_picks_right_rows() {
        let (head, dev) = build();
        let (h, mask, _) = seq(&dev, 12, 12, 0, 0);
        // Compare the scorer input rows by gathering markers 2 and 9 manually.
        let mut hh = h
            .broadcast_add(&head.type_emb0.reshape((1, 1, D)).unwrap())
            .unwrap();
        let mask_f = mask.to_dtype(DType::F32).unwrap();
        let bias = (mask_f.affine(1.0, -1.0).unwrap() * 1e30f64)
            .unwrap()
            .reshape((1, 1, 1, 12))
            .unwrap();
        for layer in &head.layers {
            hh = layer.forward(&hh, &bias).unwrap();
        }
        let markers = Tensor::from_vec(vec![2u32, 9u32], (1, 2), &dev).unwrap();
        let idx = markers
            .unsqueeze(2)
            .unwrap()
            .broadcast_as((1, 2, D))
            .unwrap()
            .contiguous()
            .unwrap();
        let m = hh.gather(&idx, 1).unwrap();
        let row2 = hh.i((0, 2)).unwrap().to_vec1::<f32>().unwrap();
        let got = m.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        assert!((row2[0] - got[0]).abs() < 1e-6);
    }
}
