use crate::objective::Objective;

pub struct OptimizeResult<T> {
    pub best_x: Vec<T>,
    pub best_value: T,
    pub iterations: usize,
    pub func_calls: usize,
    pub grad_calls: usize,
    pub path: Vec<Vec<T>>,
}

pub struct NewtonMethod {
    max_iters: usize,
    tol: f64,
}

impl NewtonMethod {
    pub fn new(max_iters: usize) -> Self {
        Self {
            max_iters,
            tol: 1e-10,
        }
    }
}

impl NewtonMethod {
    pub fn minimize<F: Objective<f64>>(&self, f: &F, start: &[f64]) -> OptimizeResult<f64> {
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
            let h = f.hessian(&x);

            let mut norm2 = 0.0;
            for gi in &g {
                norm2 += gi * gi;
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

            let p = solve_linear_system(h, g);
            for i in 0..n {
                x[i] -= p[i];
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

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for k in (i + 1)..n {
            if a[k][i].abs() > max_val {
                max_val = a[k][i].abs();
                max_row = k;
            }
        }
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }
        let pivot = a[i][i];
        if pivot.abs() < 1e-14 {
            continue;
        }
        for j in i..n {
            a[i][j] /= pivot;
        }
        b[i] /= pivot;
        for k in 0..n {
            if k == i {
                continue;
            }
            let factor = a[k][i];
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    b
}

