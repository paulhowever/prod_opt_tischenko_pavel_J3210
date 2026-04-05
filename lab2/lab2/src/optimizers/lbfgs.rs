use crate::line_search::{armijo, dot, Counters};
use crate::Objective;
use lab1_met_opt::optimizers::OptimizeResult;

pub struct Lbfgs {
    pub max_iters: usize,
    pub m: usize,
    pub tol: f64,
    pub c1: f64,
    pub rho: f64,
}

impl Lbfgs {
    pub fn new(max_iters: usize, m: usize) -> Self {
        Self {
            max_iters,
            m,
            tol: 1e-8,
            c1: 1e-4,
            rho: 0.5,
        }
    }

    fn two_loop(
        g: &[f64],
        s_hist: &[Vec<f64>],
        y_hist: &[Vec<f64>],
        rho_hist: &[f64],
        gamma: f64,
    ) -> Vec<f64> {
        let m = s_hist.len();
        let n = g.len();
        let mut q = g.to_vec();
        let mut alpha = vec![0.0; m];
        for i in (0..m).rev() {
            alpha[i] = rho_hist[i] * dot(&s_hist[i], &q);
            for j in 0..n {
                q[j] -= alpha[i] * y_hist[i][j];
            }
        }
        let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();
        for i in 0..m {
            let beta = rho_hist[i] * dot(&y_hist[i], &r);
            for j in 0..n {
                r[j] += s_hist[i][j] * (alpha[i] - beta);
            }
        }
        r
    }

    pub fn minimize(&self, f: &impl Objective<f64>, start: &[f64]) -> OptimizeResult<f64> {
        let n = f.dimension();
        let mut x = start.to_vec();
        let mut path = vec![x.clone()];
        let mut func_calls = 1usize;
        let mut grad_calls = 0usize;

        let mut f_current = f.value(&x);
        let mut g = f.gradient(&x);
        grad_calls += 1;

        let mut best_x = x.clone();
        let mut best_val = f_current;

        let mut s_hist: Vec<Vec<f64>> = Vec::new();
        let mut y_hist: Vec<Vec<f64>> = Vec::new();
        let mut rho_hist: Vec<f64> = Vec::new();

        let mut work = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut s = vec![0.0; n];
        let mut y = vec![0.0; n];

        for it in 0..self.max_iters {
            let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
            if g_norm < self.tol {
                return OptimizeResult {
                    best_x,
                    best_value: best_val,
                    iterations: it,
                    func_calls,
                    grad_calls,
                    path,
                };
            }

            let gamma = if s_hist.is_empty() {
                1.0
            } else {
                let sn = s_hist.len();
                let s_last = &s_hist[sn - 1];
                let y_last = &y_hist[sn - 1];
                let ys = dot(s_last, y_last);
                let yy = dot(y_last, y_last);
                if yy.abs() < 1e-18 {
                    1.0
                } else {
                    ys / yy
                }
            };

            let r = Self::two_loop(&g, &s_hist, &y_hist, &rho_hist, gamma);
            for i in 0..n {
                p[i] = -r[i];
            }

            let g_dot_p = dot(&g, &p);
            let mut ls = Counters::new();
            let (alpha, f_new) = armijo(
                f,
                &x,
                &p,
                f_current,
                g_dot_p,
                self.c1,
                self.rho,
                &mut work,
                &mut ls,
            );
            func_calls += ls.func_calls;

            for i in 0..n {
                s[i] = alpha * p[i];
                x[i] += s[i];
            }
            f_current = f_new;
            if f_current < best_val {
                best_val = f_current;
                best_x = x.clone();
            }

            let g_old = g.clone();
            g = f.gradient(&x);
            grad_calls += 1;

            for i in 0..n {
                y[i] = g[i] - g_old[i];
            }

            let ys = dot(&y, &s);
            if ys > 1e-12 {
                let rho = 1.0 / ys;
                s_hist.push(s.clone());
                y_hist.push(y.clone());
                rho_hist.push(rho);
                if s_hist.len() > self.m {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho_hist.remove(0);
                }
            }

            path.push(x.clone());
        }

        OptimizeResult {
            best_x,
            best_value: best_val,
            iterations: self.max_iters,
            func_calls,
            grad_calls,
            path,
        }
    }
}
