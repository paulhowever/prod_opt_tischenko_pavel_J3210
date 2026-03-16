use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::scalar::Scalar;

#[derive(Copy, Clone)]
pub struct ConstructiveNumber {
    a: f64,
    b: f64,
}

impl ConstructiveNumber {
    pub fn from_bounds(a: f64, b: f64) -> Self {
        if a <= b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }

    pub fn from_value_eps(x: f64, eps: f64) -> Self {
        Self::from_bounds(x - eps, x + eps)
    }
    #[allow(dead_code)]
    pub fn left(&self) -> f64 {
        self.a
    }
    #[allow(dead_code)]
    pub fn right(&self) -> f64 {
        self.b
    }

    pub fn mid(&self) -> f64 {
        (self.a + self.b) * 0.5
    }

    pub fn width(&self) -> f64 {
        self.b - self.a
    }

    pub fn epsilon(&self) -> f64 {
        self.width() * 0.5
    }

    pub fn real(&self, alpha: f64) -> f64 {
        self.a + alpha * (self.b - self.a)
    }
}

impl Debug for ConstructiveNumber {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("ConstructiveNumber")
            .field("a", &self.a)
            .field("b", &self.b)
            .finish()
    }
}

impl Scalar for ConstructiveNumber {
    fn zero() -> Self {
        Self::from_value_eps(0.0, 0.0)
    }

    fn one() -> Self {
        Self::from_value_eps(1.0, 0.0)
    }

    fn from_f64(v: f64) -> Self {
        Self::from_value_eps(v, 0.0)
    }

    fn to_f64(self) -> f64 {
        self.mid()
    }
}

impl Add for ConstructiveNumber {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl Sub for ConstructiveNumber {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.b,
            b: self.b - rhs.a,
        }
    }
}

impl Mul for ConstructiveNumber {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let p1 = self.a * rhs.a;
        let p2 = self.a * rhs.b;
        let p3 = self.b * rhs.a;
        let p4 = self.b * rhs.b;
        let min = p1.min(p2).min(p3).min(p4);
        let max = p1.max(p2).max(p3).max(p4);
        Self { a: min, b: max }
    }
}

impl Div for ConstructiveNumber {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        if rhs.a <= 0.0 && rhs.b >= 0.0 {
            return Self::from_value_eps(f64::NAN, f64::INFINITY);
        }
        let inv_a = 1.0 / rhs.a;
        let inv_b = 1.0 / rhs.b;
        let inv = if inv_a <= inv_b {
            Self { a: inv_a, b: inv_b }
        } else {
            Self {
                a: inv_b,
                b: inv_a,
            }
        };
        self * inv
    }
}

impl Neg for ConstructiveNumber {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            a: -self.b,
            b: -self.a,
        }
    }
}

impl PartialEq for ConstructiveNumber {
    fn eq(&self, other: &Self) -> bool {
        self.mid() == other.mid()
    }
}

impl PartialOrd for ConstructiveNumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.mid().partial_cmp(&other.mid())
    }
}

