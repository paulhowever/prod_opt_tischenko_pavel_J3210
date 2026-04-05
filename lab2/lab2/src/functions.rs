use crate::objective::Objective;

pub struct RosenbrockN {
    pub n: usize,
}

impl RosenbrockN {
    pub fn new(n: usize) -> Self {
        assert!(n >= 2);
        Self { n }
    }

    fn value_inner(&self, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..self.n - 1 {
            let t1 = x[i + 1] - x[i] * x[i];
            let t2 = 1.0 - x[i];
            s += 100.0 * t1 * t1 + t2 * t2;
        }
        s
    }

    fn gradient_inner(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut g = vec![0.0; n];
        if n >= 2 {
            let t0 = x[1] - x[0] * x[0];
            g[0] = -400.0 * x[0] * t0 - 2.0 * (1.0 - x[0]);
        }
        for i in 1..n - 1 {
            let t_prev = x[i] - x[i - 1] * x[i - 1];
            let t_next = x[i + 1] - x[i] * x[i];
            g[i] = 200.0 * t_prev - 400.0 * x[i] * t_next - 2.0 * (1.0 - x[i]);
        }
        if n >= 2 {
            let t = x[n - 1] - x[n - 2] * x[n - 2];
            g[n - 1] = 200.0 * t;
        }
        g
    }

    fn hessian_fd(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let n = self.n;
        let h = 1e-5;
        let g0 = self.gradient_inner(x);
        let mut m = vec![vec![0.0; n]; n];
        for j in 0..n {
            let mut xp = x.to_vec();
            xp[j] += h;
            let gj = self.gradient_inner(&xp);
            for i in 0..n {
                m[i][j] = (gj[i] - g0[i]) / h;
            }
        }
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (m[i][j] + m[j][i]);
                m[i][j] = v;
                m[j][i] = v;
            }
        }
        m
    }
}

impl Objective<f64> for RosenbrockN {
    fn dimension(&self) -> usize {
        self.n
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.value_inner(x)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        self.gradient_inner(x)
    }

    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        self.hessian_fd(x)
    }
}

pub struct RastriginN {
    pub n: usize,
}

impl RastriginN {
    pub fn new(n: usize) -> Self {
        assert!(n >= 1);
        Self { n }
    }
}

impl Objective<f64> for RastriginN {
    fn dimension(&self) -> usize {
        self.n
    }

    fn value(&self, x: &[f64]) -> f64 {
        let pi = std::f64::consts::PI;
        let mut s = 10.0 * self.n as f64;
        for xi in x.iter().take(self.n) {
            s += xi * xi - 10.0 * (2.0 * pi * xi).cos();
        }
        s
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let pi = std::f64::consts::PI;
        let mut g = Vec::with_capacity(self.n);
        for xi in x.iter().take(self.n) {
            g.push(2.0 * xi + 20.0 * pi * (2.0 * pi * xi).sin());
        }
        g
    }

    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let pi = std::f64::consts::PI;
        let n = self.n;
        let mut h = vec![vec![0.0; n]; n];
        for i in 0..n {
            h[i][i] = 2.0 + 40.0 * pi * pi * (2.0 * pi * x[i]).cos();
        }
        h
    }
}

fn desmos_round_value(x: f64, y: f64) -> f64 {
    let r10 = (10.0 * y).sin().round();
    let r7 = (7.0 * x).sin().round();
    let u = x * (r10 + 2.0);
    let term1 = (u * u + y - 10.0).powi(2);
    let inner = y * (r7 + 2.0);
    let term2 = (x + inner * inner - 7.0).powi(2);
    term1 + term2
}

const DESMOS_FD_H: f64 = 1e-6;

fn desmos_round_grad_fd(x: f64, y: f64) -> (f64, f64) {
    let h = DESMOS_FD_H;
    let gx = (desmos_round_value(x + h, y) - desmos_round_value(x - h, y)) / (2.0 * h);
    let gy = (desmos_round_value(x, y + h) - desmos_round_value(x, y - h)) / (2.0 * h);
    (gx, gy)
}

pub struct DesmosBadie2d;

impl Objective<f64> for DesmosBadie2d {
    fn dimension(&self) -> usize {
        2
    }

    fn value(&self, x: &[f64]) -> f64 {
        desmos_round_value(x[0], x[1])
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let (gx, gy) = desmos_round_grad_fd(x[0], x[1]);
        vec![gx, gy]
    }

    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let h = 1e-5;
        let g0 = self.gradient(x);
        let mut m = vec![vec![0.0f64; 2]; 2];
        for j in 0..2 {
            let mut xp = x.to_vec();
            xp[j] += h;
            let gj = self.gradient(&xp);
            for i in 0..2 {
                m[i][j] = (gj[i] - g0[i]) / h;
            }
        }
        m[0][1] = 0.5 * (m[0][1] + m[1][0]);
        m[1][0] = m[0][1];
        m
    }
}

pub struct DesmosBadieBlocks {
    pub n: usize,
}

impl DesmosBadieBlocks {
    pub fn new(n: usize) -> Self {
        assert!(n >= 2);
        assert!(n % 2 == 0);
        Self { n }
    }
}

impl Objective<f64> for DesmosBadieBlocks {
    fn dimension(&self) -> usize {
        self.n
    }

    fn value(&self, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for k in (0..self.n).step_by(2) {
            s += desmos_round_value(x[k], x[k + 1]);
        }
        s
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; self.n];
        for k in (0..self.n).step_by(2) {
            let (gx, gy) = desmos_round_grad_fd(x[k], x[k + 1]);
            g[k] = gx;
            g[k + 1] = gy;
        }
        g
    }

    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let n = self.n;
        let h = 1e-5;
        let g0 = self.gradient(x);
        let mut m = vec![vec![0.0; n]; n];
        for j in 0..n {
            let mut xp = x.to_vec();
            xp[j] += h;
            let gj = self.gradient(&xp);
            for i in 0..n {
                m[i][j] = (gj[i] - g0[i]) / h;
            }
        }
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (m[i][j] + m[j][i]);
                m[i][j] = v;
                m[j][i] = v;
            }
        }
        m
    }
}
