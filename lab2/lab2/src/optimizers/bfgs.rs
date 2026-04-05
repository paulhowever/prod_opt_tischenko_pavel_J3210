use crate::line_search::{armijo, dot, Counters};
use crate::Objective;
use lab1_met_opt::optimizers::OptimizeResult;

pub struct Bfgs {
    pub max_iters: usize,
    pub tol: f64,
    pub c1: f64,
    pub rho: f64,
}

impl Bfgs {
    pub fn new(max_iters: usize) -> Self {
        Self {
            max_iters,
            tol: 1e-8,
            c1: 1e-4,
            rho: 0.5,
        }
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

        let mut h: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();

        let mut work = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut s = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut hy = vec![0.0; n];

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

            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += h[i][j] * g[j];
                }
                p[i] = -sum;
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
                for i in 0..n {
                    hy[i] = 0.0;
                    for j in 0..n {
                        hy[i] += h[i][j] * y[j];
                    }
                }
                let y_hy = dot(&y, &hy);
                for i in 0..n {
                    for j in 0..n {
                        h[i][j] -= rho * s[i] * hy[j];
                        h[i][j] -= rho * hy[i] * s[j];
                        h[i][j] += rho * rho * y_hy * s[i] * s[j];
                        h[i][j] += rho * s[i] * s[j];
                    }
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
