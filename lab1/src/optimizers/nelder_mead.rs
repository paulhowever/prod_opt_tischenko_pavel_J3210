use crate::objective::Objective;
use crate::scalar::Scalar;

use super::newton::OptimizeResult;

pub struct NelderMead {
    max_iters: usize,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
}

impl NelderMead {
    pub fn new(max_iters: usize) -> Self {
        Self {
            max_iters,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

impl NelderMead {
    pub fn minimize<T: Scalar, F: Objective<T>>(
        &self,
        f: &F,
        start: &[T],
    ) -> OptimizeResult<T> {
        let n = f.dimension();
        let mut simplex: Vec<Vec<T>> = Vec::with_capacity(n + 1);
        simplex.push(start.to_vec());
        for i in 0..n {
            let mut v = start.to_vec();
            v[i] = v[i] + T::from_f64(0.05);
            simplex.push(v);
        }

        let mut values: Vec<T> = simplex.iter().map(|x| f.value(x)).collect();
        let mut func_calls = simplex.len();

        let mut path = Vec::new();
        path.push(start.to_vec());

        for it in 0..self.max_iters {
            let mut order: Vec<usize> = (0..simplex.len()).collect();
            order.sort_by(|&i, &j| values[i].partial_cmp(&values[j]).unwrap());
            simplex = order.iter().map(|&i| simplex[i].clone()).collect();
            values = order.iter().map(|&i| values[i]).collect();

            let best_x = simplex[0].clone();
            let best_val = values[0];

            if it == self.max_iters - 1 {
                return OptimizeResult {
                    best_x,
                    best_value: best_val,
                    iterations: self.max_iters,
                    func_calls,
                    grad_calls: 0,
                    path,
                };
            }

            let nverts = simplex.len();

            let mut centroid = vec![T::zero(); n];
            for i in 0..(nverts - 1) {
                for k in 0..n {
                    centroid[k] = centroid[k] + simplex[i][k];
                }
            }
            let inv = T::from_f64(1.0 / (nverts as f64 - 1.0));
            for k in 0..n {
                centroid[k] = centroid[k] * inv;
            }

            let worst = &simplex[nverts - 1];
            let alpha = T::from_f64(self.alpha);
            let mut reflected = vec![T::zero(); n];
            for k in 0..n {
                reflected[k] = centroid[k] + alpha * (centroid[k] - worst[k]);
            }
            let reflected_val = f.value(&reflected);
            func_calls += 1;

            if reflected_val < values[0] {
                let gamma = T::from_f64(self.gamma);
                let mut expanded = vec![T::zero(); n];
                for k in 0..n {
                    expanded[k] = centroid[k] + gamma * (reflected[k] - centroid[k]);
                }
                let expanded_val = f.value(&expanded);
                func_calls += 1;

                if expanded_val < reflected_val {
                    simplex[nverts - 1] = expanded;
                    values[nverts - 1] = expanded_val;
                } else {
                    simplex[nverts - 1] = reflected;
                    values[nverts - 1] = reflected_val;
                }
            } else if reflected_val < values[nverts - 2] {
                simplex[nverts - 1] = reflected;
                values[nverts - 1] = reflected_val;
            } else {
                let target_index = nverts - 1;
                let base = if reflected_val < values[nverts - 1] {
                    &reflected
                } else {
                    &simplex[nverts - 1]
                };
                let rho = T::from_f64(self.rho);
                let mut contracted = vec![T::zero(); n];
                for k in 0..n {
                    contracted[k] = centroid[k] + rho * (base[k] - centroid[k]);
                }
                let contracted_val = f.value(&contracted);
                func_calls += 1;

                if contracted_val < values[nverts - 1] {
                    simplex[target_index] = contracted;
                    values[target_index] = contracted_val;
                } else {
                    let sigma = T::from_f64(self.sigma);
                    let best = simplex[0].clone();
                    for i in 1..nverts {
                        for k in 0..n {
                            simplex[i][k] =
                                best[k] + sigma * (simplex[i][k] - best[k]);
                        }
                        values[i] = f.value(&simplex[i]);
                        func_calls += 1;
                    }
                }
            }

            path.push(simplex[0].clone());
        }

        let best_x = simplex[0].clone();
        let best_val = values[0];

        OptimizeResult {
            best_x,
            best_value: best_val,
            iterations: self.max_iters,
            func_calls,
            grad_calls: 0,
            path,
        }
    }
}

