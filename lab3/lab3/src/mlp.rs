use rand::rngs::StdRng;
use rand::Rng;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MlpLayout {
    pub in_dim: usize,
    pub hidden: usize,
    pub residual_blocks: usize,
}

#[derive(Clone, Debug)]
pub struct ThetaSlices {
    pub off_fc: usize,
    pub blocks: Vec<(usize, usize)>,
    pub off_out: usize,
    pub total: usize,
}

impl MlpLayout {
    pub fn num_params(&self) -> usize {
        self.slices().total
    }

    /// Маска bias-параметров (true = bias, true исключается из L2).
    pub fn bias_mask(&self) -> Vec<bool> {
        let sl = self.slices();
        let d0 = self.in_dim;
        let h = self.hidden;
        let mut mask = vec![false; sl.total];
        let o = sl.off_fc;
        for i in 0..h {
            mask[o + d0 * h + i] = true;
        }
        for &(ow1, ow2) in &sl.blocks {
            for i in 0..h {
                mask[ow1 + h * h + i] = true;
                mask[ow2 + h * h + i] = true;
            }
        }
        let oo = sl.off_out;
        mask[oo + h] = true;
        mask
    }

    pub fn slices(&self) -> ThetaSlices {
        let d0 = self.in_dim;
        let h = self.hidden;
        let mut off = 0usize;
        let off_fc = off;
        off += d0 * h + h;
        let mut blocks = Vec::new();
        for _ in 0..self.residual_blocks {
            let w1 = off;
            off += h * h + h;
            let w2 = off;
            off += h * h + h;
            blocks.push((w1, w2));
        }
        let off_out = off;
        off += h + 1;
        ThetaSlices {
            off_fc,
            blocks,
            off_out,
            total: off,
        }
    }

    pub fn init_theta(&self, rng: &mut StdRng) -> Vec<f64> {
        let mut t = vec![0.0; self.num_params()];
        let sl = self.slices();
        let d0 = self.in_dim;
        let h = self.hidden;
        let lim_fc = (6.0 / d0 as f64).sqrt();
        let o = sl.off_fc;
        for i in 0..(d0 * h) {
            t[o + i] = rng.gen_range(-lim_fc..lim_fc);
        }
        let lim = (6.0 / h as f64).sqrt();
        for &(ow1, ow2) in &sl.blocks {
            for i in 0..(h * h) {
                t[ow1 + i] = rng.gen_range(-lim..lim);
                t[ow2 + i] = rng.gen_range(-lim..lim);
            }
        }
        let oo = sl.off_out;
        for i in 0..h {
            t[oo + i] = rng.gen_range(-lim..lim);
        }
        t
    }
}

pub fn prob_from_logit(z: f64) -> f64 {
    sigmoid(z)
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        0.0
    } else if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

pub fn bce_loss_logit(logit: f64, y: u8) -> (f64, f64) {
    let sign = if y == 1 { 1.0 } else { -1.0 };
    let loss = softplus(-sign * logit);
    let dlogit = -sign * sigmoid(-sign * logit);
    (loss, dlogit)
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_grad(pre: f64, g: f64) -> f64 {
    if pre > 0.0 {
        g
    } else {
        0.0
    }
}

pub struct ForwardState {
    pub z_fc: Vec<f64>,
    pub a0: Vec<f64>,
    pub a_ins: Vec<Vec<f64>>,
    pub pres: Vec<Vec<f64>>,
    pub zs: Vec<Vec<f64>>,
    pub a_head: Vec<f64>,
    pub logit: f64,
}

pub fn forward_state(layout: &MlpLayout, theta: &[f64], x: &[f64]) -> ForwardState {
    let sl = layout.slices();
    let d0 = layout.in_dim;
    let h = layout.hidden;
    let o = sl.off_fc;
    let mut z_fc = vec![0.0; h];
    for i in 0..h {
        let mut s = theta[o + d0 * h + i];
        for j in 0..d0 {
            s += theta[o + i * d0 + j] * x[j];
        }
        z_fc[i] = s;
    }
    let a0: Vec<f64> = z_fc.iter().copied().map(relu).collect();
    let mut a = a0.clone();
    let mut a_ins = Vec::new();
    let mut pres = Vec::new();
    let mut zs = Vec::new();
    for &(ow1, ow2) in &sl.blocks {
        a_ins.push(a.clone());
        let mut pre = vec![0.0; h];
        for i in 0..h {
            let mut s = theta[ow1 + h * h + i];
            for j in 0..h {
                s += theta[ow1 + i * h + j] * a[j];
            }
            pre[i] = s;
        }
        let z: Vec<f64> = pre.iter().copied().map(relu).collect();
        let mut an = vec![0.0; h];
        for i in 0..h {
            let mut s = a[i] + theta[ow2 + h * h + i];
            for j in 0..h {
                s += theta[ow2 + i * h + j] * z[j];
            }
            an[i] = s;
        }
        pres.push(pre);
        zs.push(z);
        a = an;
    }
    let oo = sl.off_out;
    let mut logit = theta[oo + h];
    for j in 0..h {
        logit += theta[oo + j] * a[j];
    }
    ForwardState {
        z_fc,
        a0,
        a_ins,
        pres,
        zs,
        a_head: a,
        logit,
    }
}

pub fn backward_accum(
    layout: &MlpLayout,
    theta: &[f64],
    x: &[f64],
    st: &ForwardState,
    dlogit: f64,
    grad: &mut [f64],
) {
    let sl = layout.slices();
    let d0 = layout.in_dim;
    let h = layout.hidden;
    let oo = sl.off_out;
    let mut g_a = vec![0.0; h];
    for j in 0..h {
        grad[oo + j] += dlogit * st.a_head[j];
        g_a[j] += dlogit * theta[oo + j];
    }
    grad[oo + h] += dlogit;
    for bi in (0..sl.blocks.len()).rev() {
        let (ow1, ow2) = sl.blocks[bi];
        let pre = &st.pres[bi];
        let z = &st.zs[bi];
        let a_in = &st.a_ins[bi];
        let mut g_z = vec![0.0; h];
        for i in 0..h {
            for j in 0..h {
                let w = theta[ow2 + i * h + j];
                g_z[j] += g_a[i] * w;
                grad[ow2 + i * h + j] += g_a[i] * z[j];
            }
            grad[ow2 + h * h + i] += g_a[i];
        }
        let mut g_pre = vec![0.0; h];
        for j in 0..h {
            g_pre[j] = relu_grad(pre[j], g_z[j]);
        }
        let mut g_a_in = g_a.clone();
        for i in 0..h {
            for j in 0..h {
                let w = theta[ow1 + i * h + j];
                g_a_in[j] += g_pre[i] * w;
                grad[ow1 + i * h + j] += g_pre[i] * a_in[j];
            }
            grad[ow1 + h * h + i] += g_pre[i];
        }
        g_a = g_a_in;
    }
    let o = sl.off_fc;
    for i in 0..h {
        let gz = relu_grad(st.z_fc[i], g_a[i]);
        for j in 0..d0 {
            grad[o + i * d0 + j] += gz * x[j];
        }
        grad[o + d0 * h + i] += gz;
    }
}

pub fn logits_batch(layout: &MlpLayout, theta: &[f64], xs: &[Vec<f64>]) -> Vec<f64> {
    xs.iter()
        .map(|x| forward_state(layout, theta, x).logit)
        .collect()
}
