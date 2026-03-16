use crate::scalar::Scalar;

pub trait Objective<T: Scalar> {
    fn dimension(&self) -> usize;
    fn value(&self, x: &[T]) -> T;
    fn gradient(&self, x: &[T]) -> Vec<T>;
    fn hessian(&self, x: &[T]) -> Vec<Vec<T>>;
}

