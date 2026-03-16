use lab1_met_opt::constructive_number::ConstructiveNumber;
use lab1_met_opt::functions::{Quad4IllCond, Quad6WellCond, Rosenbrock3};
use lab1_met_opt::objective::Objective;
use lab1_met_opt::optimizers::{GradientDescent, NewtonMethod};

#[test]
fn constructive_from_bounds_order() {
    let u = ConstructiveNumber::from_bounds(3.0, 1.0);
    assert!((u.left() - 1.0).abs() < 1e-12);
    assert!((u.right() - 3.0).abs() < 1e-12);
}

#[test]
fn constructive_real_alpha() {
    let u = ConstructiveNumber::from_bounds(0.0, 10.0);
    assert!((u.real(0.0) - 0.0).abs() < 1e-12);
    assert!((u.real(1.0) - 10.0).abs() < 1e-12);
    assert!((u.real(0.5) - 5.0).abs() < 1e-12);
}

#[test]
fn constructive_add_interval() {
    let a = ConstructiveNumber::from_bounds(1.0, 2.0);
    let b = ConstructiveNumber::from_bounds(3.0, 4.0);
    let s = a + b;
    assert!((s.left() - 4.0).abs() < 1e-12);
    assert!((s.right() - 6.0).abs() < 1e-12);
}

#[test]
fn quad6_at_zero() {
    let f = Quad6WellCond::<f64>::new();
    let x = vec![0.0; 6];
    assert!(f.value(&x).abs() < 1e-12);
    let g = f.gradient(&x);
    assert!(g.iter().all(|v| v.abs() < 1e-12));
}

#[test]
fn quad4_last_axis() {
    let f = Quad4IllCond::<f64>::new();
    let x = vec![0.0, 0.0, 0.0, 1.0];
    assert!((f.value(&x) - 100.0).abs() < 1e-9);
    let g = f.gradient(&x);
    assert!((g[3] - 200.0).abs() < 1e-9);
}

#[test]
fn rosenbrock_minimum() {
    let f = Rosenbrock3::<f64>::new();
    let x = vec![1.0, 1.0, 0.0];
    assert!(f.value(&x).abs() < 1e-9);
    let g = f.gradient(&x);
    assert!(g.iter().all(|v| v.abs() < 1e-5));
}

#[test]
fn newton_quad6_to_origin() {
    let f = Quad6WellCond::<f64>::new();
    let start = vec![1.0; 6];
    let opt = NewtonMethod::new(20);
    let r = opt.minimize(&f, &start);
    assert!(r.best_value.abs() < 1e-8);
    assert!(r.best_x.iter().all(|v| v.abs() < 1e-6));
}

#[test]
fn gradient_descent_reduces_quad6() {
    let f = Quad6WellCond::<f64>::new();
    let start = vec![1.0; 6];
    let v0 = f.value(&start);
    let opt = GradientDescent::new(0.05, 5000);
    let r = opt.minimize(&f, &start);
    assert!(r.best_value < v0);
    assert!(r.best_value < 0.5);
}
