mod scalar;
mod constructive_number;
mod objective;
mod functions;
mod optimizers;

use constructive_number::ConstructiveNumber;
use functions::{Quad4IllCond, Quad6WellCond, Rosenbrock3};
use optimizers::{GradientDescent, NelderMead, NewtonMethod, OptimizeResult};
use crate::objective::Objective;

fn print_section(title: &str) {
    println!("\n{}", "=".repeat(88));
    println!("{}", title);
    println!("{}", "=".repeat(88));
}

fn print_subsection(title: &str) {
    println!("\n{}", "-".repeat(88));
    println!("{}", title);
    println!("{}", "-".repeat(88));
}

fn format_real_number(value: f64) -> String {
    if value.abs() >= 1e4 || (value.abs() > 0.0 && value.abs() < 1e-4) {
        format!("{value:.6e}")
    } else {
        format!("{value:.6}")
    }
}

fn format_real_vec(values: &[f64]) -> String {
    let parts: Vec<String> = values.iter().map(|value| format_real_number(*value)).collect();
    format!("[{}]", parts.join(", "))
}

fn max_abs_real(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn verdict_real(result: &OptimizeResult<f64>, max_iters: usize) -> &'static str {
    let value_abs = result.best_value.abs();

    if value_abs < 1e-10 && result.iterations < max_iters {
        "excellent convergence"
    } else if value_abs < 1e-6 {
        "good convergence"
    } else if result.iterations >= max_iters {
        "iteration limit reached"
    } else {
        "acceptable result"
    }
}

fn print_real_result(name: &str, result: &OptimizeResult<f64>, max_iters: usize) {
    println!("\n{name}");
    println!("  status     : {}", verdict_real(result, max_iters));
    println!("  best value : {}", format_real_number(result.best_value));
    println!("  best x     : {}", format_real_vec(&result.best_x));
    println!("  max |x_i|  : {}", format_real_number(max_abs_real(&result.best_x)));
    println!("  iterations : {}", result.iterations);
    println!("  func calls : {}", result.func_calls);
    println!("  grad calls : {}", result.grad_calls);
    println!("  path len   : {}", result.path.len());
}

fn constructive_midpoint(value: &ConstructiveNumber) -> f64 {
    value.real(0.5)
}

fn max_width_constructive(values: &[ConstructiveNumber]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.width()))
}

fn format_constructive_number(value: &ConstructiveNumber) -> String {
    format!(
        "mid={:.6}, eps={:.6}, width={:.6}",
        constructive_midpoint(value),
        value.epsilon(),
        value.width(),
    )
}

fn format_constructive_vec(values: &[ConstructiveNumber]) -> String {
    let parts: Vec<String> = values
        .iter()
        .enumerate()
        .map(|(index, value)| format!("x{}: {}", index + 1, format_constructive_number(value)))
        .collect();
    parts.join(" | ")
}

fn verdict_constructive(result: &OptimizeResult<ConstructiveNumber>, max_iters: usize) -> &'static str {
    let value_width = result.best_value.width();

    if value_width < 1e-4 && result.iterations < max_iters {
        "narrow interval, strong result"
    } else if value_width < 1.0 {
        "usable interval estimate"
    } else if result.iterations >= max_iters {
        "iteration limit reached, interval still wide"
    } else {
        "interval remains too wide"
    }
}

fn print_constructive_result(
    name: &str,
    result: &OptimizeResult<ConstructiveNumber>,
    max_iters: usize,
) {
    println!("\n{name}");
    println!("  status     : {}", verdict_constructive(result, max_iters));
    println!("  best value : {}", format_constructive_number(&result.best_value));
    println!("  best x     : {}", format_constructive_vec(&result.best_x));
    println!("  max width  : {:.6}", max_width_constructive(&result.best_x));
    println!("  iterations : {}", result.iterations);
    println!("  func calls : {}", result.func_calls);
    println!("  grad calls : {}", result.grad_calls);
    println!("  path len   : {}", result.path.len());
}

fn run_experiments_real() {
    let quad6 = Quad6WellCond::<f64>::new();
    let quad4 = Quad4IllCond::<f64>::new();
    let rose = Rosenbrock3::<f64>::new();

    let start6 = vec![1.0; quad6.dimension()];
    let start4 = vec![1.0; quad4.dimension()];
    let start3 = vec![1.0; rose.dimension()];

    let gd_max_iters = 10_000;
    let nm_max_iters = 10_000;
    let newton_max_iters = 100;

    let gd = GradientDescent::new(0.01, gd_max_iters);
    let nm = NelderMead::new(nm_max_iters);
    let newton = NewtonMethod::new(newton_max_iters);

    let res_gd_6: OptimizeResult<f64> = gd.minimize(&quad6, &start6);
    let res_nm_6: OptimizeResult<f64> = nm.minimize(&quad6, &start6);
    let res_newton_6: OptimizeResult<f64> = newton.minimize(&quad6, &start6);

    let res_gd_4: OptimizeResult<f64> = gd.minimize(&quad4, &start4);
    let res_nm_4: OptimizeResult<f64> = nm.minimize(&quad4, &start4);
    let res_newton_4: OptimizeResult<f64> = newton.minimize(&quad4, &start4);

    let res_gd_3: OptimizeResult<f64> = gd.minimize(&rose, &start3);
    let res_nm_3: OptimizeResult<f64> = nm.minimize(&rose, &start3);
    let res_newton_3: OptimizeResult<f64> = newton.minimize(&rose, &start3);

    print_section("REAL NUMBERS (f64)");

    print_subsection("Quad6WellCond");
    print_real_result("Gradient Descent", &res_gd_6, gd_max_iters);
    print_real_result("Nelder Mead", &res_nm_6, nm_max_iters);
    print_real_result("Newton Method", &res_newton_6, newton_max_iters);

    print_subsection("Quad4IllCond");
    print_real_result("Gradient Descent", &res_gd_4, gd_max_iters);
    print_real_result("Nelder Mead", &res_nm_4, nm_max_iters);
    print_real_result("Newton Method", &res_newton_4, newton_max_iters);

    print_subsection("Rosenbrock3");
    print_real_result("Gradient Descent", &res_gd_3, gd_max_iters);
    print_real_result("Nelder Mead", &res_nm_3, nm_max_iters);
    print_real_result("Newton Method", &res_newton_3, newton_max_iters);
}

fn run_experiments_constructive() {
    let quad6 = Quad6WellCond::<ConstructiveNumber>::new();
    let quad4 = Quad4IllCond::<ConstructiveNumber>::new();
    let rose = Rosenbrock3::<ConstructiveNumber>::new();

    let start6 = vec![ConstructiveNumber::from_value_eps(1.0, 1e-3); quad6.dimension()];
    let start4 = vec![ConstructiveNumber::from_value_eps(1.0, 1e-3); quad4.dimension()];
    let start3 = vec![ConstructiveNumber::from_value_eps(1.0, 1e-3); rose.dimension()];

    let gd_max_iters = 10_000;
    let nm_max_iters = 10_000;

    let gd = GradientDescent::new(ConstructiveNumber::from_value_eps(0.01, 1e-6), gd_max_iters);
    let nm = NelderMead::new(nm_max_iters);

    let res_gd_6: OptimizeResult<ConstructiveNumber> = gd.minimize(&quad6, &start6);
    let res_nm_6: OptimizeResult<ConstructiveNumber> = nm.minimize(&quad6, &start6);

    let res_gd_4: OptimizeResult<ConstructiveNumber> = gd.minimize(&quad4, &start4);
    let res_nm_4: OptimizeResult<ConstructiveNumber> = nm.minimize(&quad4, &start4);

    let res_gd_3: OptimizeResult<ConstructiveNumber> = gd.minimize(&rose, &start3);
    let res_nm_3: OptimizeResult<ConstructiveNumber> = nm.minimize(&rose, &start3);

    print_section("CONSTRUCTIVE NUMBERS");

    print_subsection("Quad6WellCond");
    print_constructive_result("Gradient Descent", &res_gd_6, gd_max_iters);
    print_constructive_result("Nelder Mead", &res_nm_6, nm_max_iters);

    print_subsection("Quad4IllCond");
    print_constructive_result("Gradient Descent", &res_gd_4, gd_max_iters);
    print_constructive_result("Nelder Mead", &res_nm_4, nm_max_iters);

    print_subsection("Rosenbrock3");
    print_constructive_result("Gradient Descent", &res_gd_3, gd_max_iters);
    print_constructive_result("Nelder Mead", &res_nm_3, nm_max_iters);
}

fn main() {
    run_experiments_real();
    run_experiments_constructive();
}

