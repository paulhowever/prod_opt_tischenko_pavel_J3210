use crate::objective::Objective;
use crate::scalar::Scalar;

pub struct Quad6WellCond<T: Scalar> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Quad6WellCond<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> Objective<T> for Quad6WellCond<T> {
    fn dimension(&self) -> usize {
        6
    }

    fn value(&self, x: &[T]) -> T {
        let mut s = T::zero();
        for xi in x.iter().take(6) {
            s = s + (*xi * *xi);
        }
        s
    }

    fn gradient(&self, x: &[T]) -> Vec<T> {
        let mut g = Vec::with_capacity(6);
        let two = T::from_f64(2.0);
        for xi in x.iter().take(6) {
            g.push(two * *xi);
        }
        g
    }

    fn hessian(&self, _x: &[T]) -> Vec<Vec<T>> {
        let n = 6;
        let mut h = vec![vec![T::zero(); n]; n];
        let two = T::from_f64(2.0);
        for i in 0..n {
            h[i][i] = two;
        }
        h
    }
}

pub struct Quad4IllCond<T: Scalar> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Quad4IllCond<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> Objective<T> for Quad4IllCond<T> {
    fn dimension(&self) -> usize {
        4
    }

    fn value(&self, x: &[T]) -> T {
        let d = [1.0, 1.0, 1.0, 100.0];
        let mut s = T::zero();
        for i in 0..4 {
            let di = T::from_f64(d[i]);
            s = s + di * x[i] * x[i];
        }
        s
    }

    fn gradient(&self, x: &[T]) -> Vec<T> {
        let d = [1.0, 1.0, 1.0, 100.0];
        let mut g = Vec::with_capacity(4);
        let two = T::from_f64(2.0);
        for i in 0..4 {
            let di = T::from_f64(d[i]);
            g.push(two * di * x[i]);
        }
        g
    }

    fn hessian(&self, _x: &[T]) -> Vec<Vec<T>> {
        let d = [1.0, 1.0, 1.0, 100.0];
        let n = 4;
        let mut h = vec![vec![T::zero(); n]; n];
        let two = T::from_f64(2.0);
        for i in 0..n {
            let di = T::from_f64(d[i]);
            h[i][i] = two * di;
        }
        h
    }
}

pub struct Rosenbrock3<T: Scalar> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Rosenbrock3<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> Objective<T> for Rosenbrock3<T> {
    fn dimension(&self) -> usize {
        3
    }

    fn value(&self, x: &[T]) -> T {
        let x0 = x[0];
        let x1 = x[1];
        let x2 = x[2];
        let hundred = T::from_f64(100.0);
        let one = T::one();
        let term1 = x1 - x0 * x0;
        let term2 = one - x0;
        hundred * term1 * term1 + term2 * term2 + x2 * x2
    }

    fn gradient(&self, x: &[T]) -> Vec<T> {
        let x0 = x[0];
        let x1 = x[1];
        let x2 = x[2];
        let hundred = T::from_f64(100.0);
        let four = T::from_f64(4.0);
        let two = T::from_f64(2.0);
        let one = T::one();
        let term = x1 - x0 * x0;
        let df_dx0 = T::zero() - four * hundred * x0 * term - two * (one - x0);
        let df_dx1 = two * hundred * term;
        let df_dx2 = two * x2;
        vec![df_dx0, df_dx1, df_dx2]
    }

    fn hessian(&self, x: &[T]) -> Vec<Vec<T>> {
        let x0 = x[0];
        let x1 = x[1];
        let hundred = T::from_f64(100.0);
        let two = T::from_f64(2.0);
        let four = T::from_f64(4.0);
        let eight = T::from_f64(8.0);
        let df2_dx0dx0 = two + eight * hundred * x0 * x0 - four * hundred * (x1 - x0 * x0);
        let df2_dx0dx1 = T::zero() - four * hundred * x0;
        let df2_dx1dx1 = two * hundred;
        let df2_dx2dx2 = two;
        vec![
            vec![df2_dx0dx0, df2_dx0dx1, T::zero()],
            vec![df2_dx0dx1, df2_dx1dx1, T::zero()],
            vec![T::zero(), T::zero(), df2_dx2dx2],
        ]
    }
}
