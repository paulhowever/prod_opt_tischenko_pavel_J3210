use lab1_met_opt::optimizers::OptimizeResult;
use lab2_met_opt::line_search::{armijo, dot, Counters};
use lab2_met_opt::Objective;

pub struct SteepestDescentArmijo {
    pub max_iters: usize,
    pub tol: f64,
    pub c1: f64,
    pub rho: f64,
    /// Если > 1, в `path` попадает каждый `path_stride`-й шаг (и первый/последний при необходимости).
    pub path_stride: usize,
}

impl SteepestDescentArmijo {
    pub fn new(max_iters: usize) -> Self {
        Self {
            max_iters,
            tol: 1e-7,
            c1: 1e-4,
            rho: 0.5,
            path_stride: 1,
        }
    }

    pub fn minimize(&self, f: &impl Objective<f64>, start: &[f64]) -> OptimizeResult<f64> {
        let n = f.dimension();
        let stride = self.path_stride.max(1);
        let mut x = start.to_vec();
        let mut path = vec![x.clone()];
        let mut path_iters: Vec<usize> = vec![0];
        let mut func_calls = 1usize;
        let mut grad_calls = 0usize;
        let mut f_current = f.value(&x);
        let mut g = f.gradient(&x);
        grad_calls += 1;
        let mut best_x = x.clone();
        let mut best_val = f_current;
        let mut work = vec![0.0; n];
        for it in 0..self.max_iters {
            let g_norm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
            if g_norm < self.tol {
                if *path_iters.last().unwrap() != it {
                    path.push(x.clone());
                    path_iters.push(it);
                }
                return OptimizeResult {
                    best_x,
                    best_value: best_val,
                    iterations: it,
                    func_calls,
                    grad_calls,
                    path,
                    path_iters,
                };
            }
            let p: Vec<f64> = g.iter().map(|gi| -gi).collect();
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
                x[i] += alpha * p[i];
            }
            f_current = f_new;
            if f_current < best_val {
                best_val = f_current;
                best_x = x.clone();
            }
            g = f.gradient(&x);
            grad_calls += 1;
            let step = it + 1;
            if step % stride == 0 || step == self.max_iters {
                path.push(x.clone());
                path_iters.push(step);
            }
        }
        OptimizeResult {
            best_x,
            best_value: best_val,
            iterations: self.max_iters,
            func_calls,
            grad_calls,
            path,
            path_iters,
        }
    }
}
