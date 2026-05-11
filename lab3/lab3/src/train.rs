use std::time::Instant;

use lab1_met_opt::objective::Objective;
use lab1_met_opt::optimizers::OptimizeResult;
use lab2_met_opt::optimizers::Lbfgs;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use crate::data::{Dataset, Normalizer};
use crate::metrics::{best_threshold_f1, f1_binary, logits_to_probs, predict_threshold};
use crate::mlp::{logits_batch, MlpLayout};
use crate::nn_objective::NnObjective;
use crate::optim::SteepestDescentArmijo;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimMethod {
    Lbfgs,
    SteepestArmijo,
}

impl OptimMethod {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "lbfgs" | "l-bfgs" => Some(Self::Lbfgs),
            "sd" | "steepest" | "gd" | "armijo" => Some(Self::SteepestArmijo),
            _ => None,
        }
    }
}

#[derive(Serialize)]
pub struct TrainReport {
    pub method: String,
    pub seed: u64,
    pub hidden: usize,
    pub residual_blocks: usize,
    pub l2: f64,
    pub pos_weight: f64,
    pub f1_val: f64,
    pub f1_val_default_threshold: f64,
    pub threshold: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub f1_test: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub f1_test_default_threshold: Option<f64>,
    pub best_loss: f64,
    pub iterations: usize,
    pub func_calls: usize,
    pub grad_calls: usize,
    pub seconds: f64,
    pub loss_curve: Vec<LossPoint>,
}

#[derive(Serialize)]
pub struct LossPoint {
    pub iter: usize,
    pub loss: f64,
}

pub struct TrainConfig {
    pub method: OptimMethod,
    pub seed: u64,
    pub hidden: usize,
    pub residual_blocks: usize,
    pub l2: f64,
    /// Вес положительного класса; `None` — вычислить из обучающей выборки при `balance_binary_loss`.
    pub pos_weight: Option<f64>,
    pub balance_binary_loss: bool,
    pub lbfgs_m: usize,
    pub max_iters_lbfgs: usize,
    pub max_iters_sd: usize,
    /// Запись траектории L-BFGS каждые N шагов (экономия памяти). 1 — каждый шаг.
    pub lbfgs_path_stride: usize,
    /// Запись траектории steepest descent каждые N шагов.
    pub sd_path_stride: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            method: OptimMethod::Lbfgs,
            seed: 42,
            hidden: 32,
            residual_blocks: 1,
            l2: 1e-4,
            pos_weight: None,
            balance_binary_loss: true,
            lbfgs_m: 10,
            max_iters_lbfgs: 2000,
            max_iters_sd: 4000,
            lbfgs_path_stride: 5,
            sd_path_stride: 20,
        }
    }
}

fn pos_weight_for_train(y: &[u8], cfg: &TrainConfig) -> f64 {
    if let Some(w) = cfg.pos_weight {
        return w.max(1e-8);
    }
    if !cfg.balance_binary_loss {
        return 1.0;
    }
    let n1 = y.iter().filter(|&&yi| yi == 1).count();
    let n0 = y.len().saturating_sub(n1);
    if n1 == 0 {
        return 1.0;
    }
    (n0 as f64 / n1 as f64).max(1e-8)
}

fn loss_curve(
    layout: &MlpLayout,
    x: &[Vec<f64>],
    y: &[u8],
    l2: f64,
    pos_weight: f64,
    path: &[Vec<f64>],
    path_iters: &[usize],
    every: usize,
) -> Vec<LossPoint> {
    let obj = NnObjective {
        layout: layout.clone(),
        x,
        y,
        l2,
        pos_weight,
    };
    let mut out = Vec::new();
    for (slot, theta) in path.iter().enumerate() {
        let iter_idx = if path_iters.is_empty() {
            slot
        } else {
            path_iters[slot]
        };
        if slot == 0 || slot + 1 == path.len() || iter_idx % every == 0 {
            let v = obj.value(theta);
            out.push(LossPoint {
                iter: iter_idx,
                loss: v,
            });
        }
    }
    out
}

pub fn train(
    train_ds: &Dataset,
    val_ds: &Dataset,
    test_ds: Option<&Dataset>,
    cfg: &TrainConfig,
) -> (Vec<f64>, Normalizer, TrainReport) {
    let layout = MlpLayout {
        in_dim: train_ds.feature_dim,
        hidden: cfg.hidden,
        residual_blocks: cfg.residual_blocks,
    };
    let pos_weight = pos_weight_for_train(&train_ds.y, cfg);
    let norm = Normalizer::fit(&train_ds.x, train_ds.feature_dim);
    let mut x_tr: Vec<Vec<f64>> = train_ds.x.clone();
    let y_tr = train_ds.y.clone();
    let mut x_va: Vec<Vec<f64>> = val_ds.x.clone();
    norm.transform(&mut x_tr);
    norm.transform(&mut x_va);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let theta0 = layout.init_theta(&mut rng);
    let obj = NnObjective {
        layout: layout.clone(),
        x: &x_tr,
        y: &y_tr,
        l2: cfg.l2,
        pos_weight,
    };
    let t0 = Instant::now();
    let result: OptimizeResult<f64> = match cfg.method {
        OptimMethod::Lbfgs => {
            let mut o = Lbfgs::new(cfg.max_iters_lbfgs, cfg.lbfgs_m);
            o.tol = 1e-7;
            o.path_stride = cfg.lbfgs_path_stride.max(1);
            o.minimize(&obj, &theta0)
        }
        OptimMethod::SteepestArmijo => {
            let mut o = SteepestDescentArmijo::new(cfg.max_iters_sd);
            o.path_stride = cfg.sd_path_stride.max(1);
            o.minimize(&obj, &theta0)
        }
    };
    let seconds = t0.elapsed().as_secs_f64();
    let theta = result.best_x.clone();
    let val_logits = logits_batch(&layout, &theta, &x_va);
    let val_probs = logits_to_probs(&val_logits);
    let pred05 = predict_threshold(&val_probs, 0.5);
    let f1_default = f1_binary(&val_ds.y, &pred05);
    let (thr, _best_f1_grid) = best_threshold_f1(&val_ds.y, &val_probs);
    let pred_t = predict_threshold(&val_probs, thr);
    let f1_val = f1_binary(&val_ds.y, &pred_t);
    let (f1_test, f1_test_def) = if let Some(te) = test_ds {
        let mut x_te: Vec<Vec<f64>> = te.x.clone();
        norm.transform(&mut x_te);
        let test_logits = logits_batch(&layout, &theta, &x_te);
        let test_probs = logits_to_probs(&test_logits);
        let p05 = predict_threshold(&test_probs, 0.5);
        let fdef = f1_binary(&te.y, &p05);
        let pt = predict_threshold(&test_probs, thr);
        let ft = f1_binary(&te.y, &pt);
        (Some(ft), Some(fdef))
    } else {
        (None, None)
    };
    let max_iter_tag = result.iterations.max(1);
    let every = max_iter_tag / 50 + 1;
    let curve = loss_curve(
        &layout,
        &x_tr,
        &y_tr,
        cfg.l2,
        pos_weight,
        &result.path,
        &result.path_iters,
        every,
    );
    let method_str = match cfg.method {
        OptimMethod::Lbfgs => "lbfgs",
        OptimMethod::SteepestArmijo => "steepest_armijo",
    }
    .to_string();
    let report = TrainReport {
        method: method_str,
        seed: cfg.seed,
        hidden: cfg.hidden,
        residual_blocks: cfg.residual_blocks,
        l2: cfg.l2,
        pos_weight,
        f1_val,
        f1_val_default_threshold: f1_default,
        threshold: thr,
        f1_test,
        f1_test_default_threshold: f1_test_def,
        best_loss: result.best_value,
        iterations: result.iterations,
        func_calls: result.func_calls,
        grad_calls: result.grad_calls,
        seconds,
        loss_curve: curve,
    };
    (theta, norm, report)
}
