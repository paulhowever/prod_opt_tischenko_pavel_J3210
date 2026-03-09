use crate::objective::Objective;
use crate::scalar::Scalar;

use super::newton::OptimizeResult;

pub struct GradientDescent<T: Scalar> {
    learning_rate: T,
    max_iters: usize,
    tol: f64,
}

impl<T: Scalar> GradientDescent<T> {
    pub fn new(learning_rate: T, max_iters: usize) -> Self {
        Self {
            learning_rate,
            max_iters,
            tol: 1e-8,
        }
    }
}

impl<T: Scalar> GradientDescent<T> {
    pub fn minimize<F: Objective<T>>(&self, f: &F, start: &[T]) -> OptimizeResult<T> {
        let n = f.dimension();
        let mut x = start.to_vec();
        let mut path = Vec::new();
        path.push(x.clone());

        let mut func_calls = 0usize;
        let mut grad_calls = 0usize;

        let mut best_x = x.clone();
        let mut best_val = f.value(&x);
        func_calls += 1;

        for it in 0..self.max_iters {
            let g = f.gradient(&x);
            grad_calls += 1;

            let mut norm2 = 0.0;
            for gi in &g {
                let v = gi.to_f64();
                norm2 += v * v;
            }
            if norm2.sqrt() < self.tol {
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
                x[i] = x[i] - self.learning_rate * g[i];
            }

            let val = f.value(&x);
            func_calls += 1;

            if val < best_val {
                best_val = val;
                best_x = x.clone();
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

