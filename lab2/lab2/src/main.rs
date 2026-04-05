use lab2_met_opt::functions::{DesmosBadie2d, DesmosBadieBlocks, RastriginN, RosenbrockN};
use lab2_met_opt::objective::Objective;
use lab2_met_opt::optimizers::{
    Bfgs, GradientDescent, Lbfgs, NelderMead, NewtonMethod, NonlinearCgFr, OptimizeResult,
};

const W: usize = 92;

fn grad_norm<O: Objective<f64>>(f: &O, x: &[f64]) -> f64 {
    let g = f.gradient(x);
    g.iter().map(|t| t * t).sum::<f64>().sqrt()
}

fn banner_line(ch: char) {
    println!("{}", ch.to_string().repeat(W));
}

fn print_section_header(label: &str, dim: usize) {
    println!();
    banner_line('═');
    println!("  {}    dim = {}", label, dim);
    banner_line('─');
    println!(
        "  {:<14} {:>22} {:>16} {:>8} {:>10} {:>10}",
        "method", "best f", "‖∇f‖", "iters", "#f", "#∇f"
    );
    banner_line('─');
}

fn print_row(method: &str, r: &OptimizeResult<f64>, gx: f64) {
    println!(
        "  {:<14} {:>22.6e} {:>16.6e} {:>8} {:>10} {:>10}",
        method, r.best_value, gx, r.iterations, r.func_calls, r.grad_calls
    );
}

fn section_footer() {
    banner_line('═');
}

fn run_all<O: Objective<f64>>(
    label: &str,
    f: &O,
    dim: usize,
    start: &[f64],
    tol: f64,
    max_it: usize,
) {
    print_section_header(label, dim);

    let mut bfgs = Bfgs::new(max_it);
    bfgs.tol = tol;
    let r = bfgs.minimize(f, start);
    print_row("bfgs", &r, grad_norm(f, &r.best_x));

    let mut lbfgs = Lbfgs::new(max_it, 12);
    lbfgs.tol = tol;
    let r = lbfgs.minimize(f, start);
    print_row("lbfgs", &r, grad_norm(f, &r.best_x));

    let mut nlcg = NonlinearCgFr::new(max_it);
    nlcg.tol = tol;
    let r = nlcg.minimize(f, start);
    print_row("nlcg_fr", &r, grad_norm(f, &r.best_x));

    let gd = GradientDescent::new(0.001_f64, max_it);
    let r = gd.minimize(f, start);
    print_row("gd", &r, grad_norm(f, &r.best_x));

    let newton = NewtonMethod::new(max_it);
    let r = newton.minimize(f, start);
    print_row("newton", &r, grad_norm(f, &r.best_x));

    let nm = NelderMead::new(max_it);
    let r = nm.minimize(f, start);
    print_row("nelder_mead", &r, grad_norm(f, &r.best_x));

    section_footer();
}

fn main() {
    banner_line('═');
    println!(
        "  {:^width$}",
        "lab2 — сравнение оптимизаторов",
        width = W - 4
    );
    banner_line('═');

    let tol = 1e-8;
    let max_it = 5000usize;

    let rosen2 = RosenbrockN::new(2);
    let x0_r2 = vec![-1.2, 1.0];
    run_all("rosen2", &rosen2, 2, &x0_r2, tol, max_it);

    let rosen10 = RosenbrockN::new(10);
    let x0_r10: Vec<f64> = (0..10).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
    run_all("rosen10", &rosen10, 10, &x0_r10, tol, max_it);

    let rast2 = RastriginN::new(2);
    let x0_ra2 = vec![0.5, -0.5];
    run_all("rastrigin2", &rast2, 2, &x0_ra2, tol, max_it);

    let rast8 = RastriginN::new(8);
    let x0_ra8: Vec<f64> = vec![0.5; 8];
    run_all("rastrigin8", &rast8, 8, &x0_ra8, tol, max_it);

    let bad2 = DesmosBadie2d;
    let x0_b2 = vec![0.0, 5.0];
    run_all("desmos2d", &bad2, 2, &x0_b2, tol, max_it);

    let bad8 = DesmosBadieBlocks::new(8);
    let x0_b8: Vec<f64> = (0..8).map(|i| if i % 2 == 0 { 0.0 } else { 5.0 }).collect();
    run_all("desmos8", &bad8, 8, &x0_b8, tol, max_it);

    println!();
}
