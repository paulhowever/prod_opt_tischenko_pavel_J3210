mod gradient_descent;
mod nelder_mead;
mod newton;

pub use gradient_descent::GradientDescent;
pub use nelder_mead::NelderMead;
pub use newton::NewtonMethod;
pub use newton::OptimizeResult;

