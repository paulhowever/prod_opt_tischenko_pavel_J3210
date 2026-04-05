use crate::line_search::{armijo, dot, Counters};
use crate::Objective;
use lab1_met_opt::optimizers::OptimizeResult;

pub struct NonlinearCgFr {
    pub max_iters: usize,
    pub tol: f64,
    pub c1: f64,
    pub rho: f64,
}

impl NonlinearCgFr {
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

        let mut d: Vec<f64> = g.iter().map(|gi| -gi).collect();
        let mut g_prev_sq = dot(&g, &g);

        let mut work = vec![0.0; n];

        for it in 0..self.max_iters {
            let g_norm = g_prev_sq.sqrt();
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

            if it > 0 && it % n == 0 {
                d = g.iter().map(|gi| -gi).collect();
            }

            let g_dot_p = dot(&g, &d);
            if g_dot_p >= 0.0 {
                d = g.iter().map(|gi| -gi).collect();
            }

            let g_dot_p = dot(&g, &d);
            let mut ls = Counters::new();
            let (alpha, f_new) = armijo(
                f,
                &x,
                &d,
                f_current,
                g_dot_p,
                self.c1,
                self.rho,
                &mut work,
                &mut ls,
            );
            func_calls += ls.func_calls;

            for i in 0..n {
                x[i] += alpha * d[i];
            }
            f_current = f_new;
            if f_current < best_val {
                best_val = f_current;
                best_x = x.clone();
            }

            g = f.gradient(&x);
            grad_calls += 1;

            let g_new_sq = dot(&g, &g);
            let beta = if g_prev_sq > 1e-30 {
                g_new_sq / g_prev_sq
            } else {
                0.0
            };
            g_prev_sq = g_new_sq;

            for i in 0..n {
                d[i] = -g[i] + beta * d[i];
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
