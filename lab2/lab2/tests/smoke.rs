use lab2_met_opt::objective::Objective;
use lab2_met_opt::optimizers::{Bfgs, Lbfgs};
use lab2_met_opt::functions::RosenbrockN;

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
fn bfgs_sphere() {
    let f = Sphere { n: 4 };
    let x0 = vec![1.0, 2.0, 3.0, 4.0];
    let mut o = Bfgs::new(200);
    o.tol = 1e-10;
    let r = o.minimize(&f, &x0);
    assert!(r.best_value < 1e-8);
}

#[test]
fn lbfgs_sphere() {
    let f = Sphere { n: 4 };
    let x0 = vec![1.0, 2.0, 3.0, 4.0];
    let mut o = Lbfgs::new(500, 8);
    o.tol = 1e-10;
    let r = o.minimize(&f, &x0);
    assert!(r.best_value < 1e-8);
}

#[test]
fn bfgs_rosen2() {
    let f = RosenbrockN::new(2);
    let x0 = vec![-1.2, 1.0];
    let mut o = Bfgs::new(2000);
    o.tol = 1e-6;
    let r = o.minimize(&f, &x0);
    assert!(r.best_value < 1e-4);
}
