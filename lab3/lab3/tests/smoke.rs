use lab3_met_opt::mlp::MlpLayout;
use lab3_met_opt::nn_objective::NnObjective;
use lab3_met_opt::Objective;

struct Sphere {
    n: usize,
}

impl Objective<f64> for Sphere {
    fn dimension(&self) -> usize {
        self.n
    }

    fn value(&self, x: &[f64]) -> f64 {
        x.iter().take(self.n).map(|t| t * t).sum()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter().take(self.n).map(|t| 2.0 * t).collect()
    }

    fn hessian(&self, _: &[f64]) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut h = vec![vec![0.0; n]; n];
        for i in 0..n {
            h[i][i] = 2.0;
        }
        h
    }
}

#[test]
fn objective_link_sanity() {
    let f = Sphere { n: 3 };
    let x = vec![1.0, 2.0, 3.0];
    assert!((f.value(&x) - 14.0).abs() < 1e-12);
}

#[test]
fn nn_gradient_fd() {
    let layout = MlpLayout {
        in_dim: 2,
        hidden: 4,
        residual_blocks: 0,
    };
    let xs = vec![vec![0.5, -0.3]];
    let ys = vec![1u8];
    let mut theta = vec![0.07; layout.num_params()];
    let obj = NnObjective {
        layout: layout.clone(),
        x: &xs,
        y: &ys,
        l2: 0.01,
        pos_weight: 1.0,
    };
    let g = obj.gradient(&theta);
    let v0 = obj.value(&theta);
    let eps = 1e-6;
    for i in 0..theta.len() {
        theta[i] += eps;
        let vp = obj.value(&theta);
        theta[i] -= eps;
        let fd = (vp - v0) / eps;
        assert!((fd - g[i]).abs() < 2e-5, "i {} fd {} g {}", i, fd, g[i]);
    }
}

#[test]
fn nn_gradient_fd_residual_and_batch() {
    let layout = MlpLayout {
        in_dim: 2,
        hidden: 4,
        residual_blocks: 2,
    };
    let xs = vec![
        vec![0.5, -0.3],
        vec![-1.0, 2.0],
        vec![0.1, 0.2],
    ];
    let ys = vec![1u8, 0u8, 1u8];
    let mut theta = vec![0.05; layout.num_params()];
    let obj = NnObjective {
        layout: layout.clone(),
        x: &xs,
        y: &ys,
        l2: 0.02,
        pos_weight: 1.7,
    };
    let g = obj.gradient(&theta);
    let v0 = obj.value(&theta);
    let eps = 1e-6;
    for i in 0..theta.len() {
        theta[i] += eps;
        let vp = obj.value(&theta);
        theta[i] -= eps;
        let fd = (vp - v0) / eps;
        assert!((fd - g[i]).abs() < 3e-5, "i {} fd {} g {}", i, fd, g[i]);
    }
}
