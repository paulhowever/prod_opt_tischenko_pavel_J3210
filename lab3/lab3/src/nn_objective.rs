use lab1_met_opt::objective::Objective;

use crate::mlp::{backward_accum, bce_loss_logit, forward_state, MlpLayout};

pub struct NnObjective<'a> {
    pub layout: MlpLayout,
    pub x: &'a [Vec<f64>],
    pub y: &'a [u8],
    pub l2: f64,
    /// Вес положительного класса в потере (как в BCEWithLogitsLoss pos_weight). 1.0 — без взвешивания.
    pub pos_weight: f64,
}

impl<'a> NnObjective<'a> {
    fn loss_denom(&self) -> f64 {
        let mut d = 0.0;
        for &yi in self.y {
            d += if yi == 1 { self.pos_weight } else { 1.0 };
        }
        d.max(1e-18)
    }

    fn is_bias(&self) -> Vec<bool> {
        self.layout.bias_mask()
    }
}

impl<'a> Objective<f64> for NnObjective<'a> {
    fn dimension(&self) -> usize {
        self.layout.num_params()
    }

    fn value(&self, theta: &[f64]) -> f64 {
        let denom = self.loss_denom();
        let mut s = 0.0;
        for i in 0..self.x.len() {
            let st = forward_state(&self.layout, theta, &self.x[i]);
            let (li, _) = bce_loss_logit(st.logit, self.y[i]);
            let w = if self.y[i] == 1 { self.pos_weight } else { 1.0 };
            s += w * li;
        }
        let bias = self.is_bias();
        let mut sq = 0.0;
        for (j, &t) in theta.iter().enumerate() {
            if !bias[j] {
                sq += t * t;
            }
        }
        let l2_term = 0.5 * self.l2 * sq;
        s / denom + l2_term
    }

    fn gradient(&self, theta: &[f64]) -> Vec<f64> {
        let denom = self.loss_denom();
        let mut g = vec![0.0; self.dimension()];
        for i in 0..self.x.len() {
            let st = forward_state(&self.layout, theta, &self.x[i]);
            let (_, dlog) = bce_loss_logit(st.logit, self.y[i]);
            let w = if self.y[i] == 1 { self.pos_weight } else { 1.0 };
            let sc = w * dlog / denom;
            backward_accum(&self.layout, theta, &self.x[i], &st, sc, &mut g);
        }
        let bias = self.is_bias();
        for j in 0..g.len() {
            if !bias[j] {
                g[j] += self.l2 * theta[j];
            }
        }
        g
    }

    fn hessian(&self, _: &[f64]) -> Vec<Vec<f64>> {
        unimplemented!("hessian of NN BCE-loss is not implemented; use L-BFGS or steepest descent")
    }
}
