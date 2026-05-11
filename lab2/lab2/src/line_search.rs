use crate::Objective;

pub struct Counters {
    pub func_calls: usize,
    pub grad_calls: usize,
}

impl Counters {
    pub fn new() -> Self {
        Self {
            func_calls: 0,
            grad_calls: 0,
        }
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn add_scaled(x: &[f64], p: &[f64], alpha: f64, out: &mut [f64]) {
    for i in 0..x.len() {
        out[i] = x[i] + alpha * p[i];
    }
}

pub fn armijo(
    f: &impl Objective<f64>,
    x: &[f64],
    p: &[f64],
    f_x: f64,
    g_dot_p: f64,
    c1: f64,
    rho: f64,
    work: &mut [f64],
    counters: &mut Counters,
) -> (f64, f64) {
    let mut alpha = 1.0;
    for _ in 0..64 {
        add_scaled(x, p, alpha, work);
        let f_new = f.value(work);
        counters.func_calls += 1;
        if f_new <= f_x + c1 * alpha * g_dot_p {
            return (alpha, f_new);
        }
        alpha *= rho;
        if alpha < 1e-30 {
            add_scaled(x, p, alpha, work);
            let f_new = f.value(work);
            counters.func_calls += 1;
            return (alpha, f_new);
        }
    }
    add_scaled(x, p, alpha, work);
    let f_new = f.value(work);
    counters.func_calls += 1;
    (alpha, f_new)
}
