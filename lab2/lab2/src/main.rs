use std::collections::BTreeMap;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};

use lab2_met_opt::functions::{DesmosBadie2d, DesmosBadieBlocks, RastriginN, RosenbrockN};
use lab2_met_opt::objective::Objective;
use lab2_met_opt::optimizers::{
    Bfgs, GradientDescent, Lbfgs, NelderMead, NewtonMethod, NonlinearCgFr, OptimizeResult,
};

const W: usize = 118;

#[derive(Clone)]
struct TrialRow {
    function: String,
    dim: usize,
    start_id: usize,
    method: String,
    best_value: f64,
    grad_norm: f64,
    iterations: usize,
    func_calls: usize,
    grad_calls: usize,
}

#[derive(Clone, Default)]
struct Aggregate {
    function: String,
    dim: usize,
    method: String,
    trials: usize,
    mean_best_value: f64,
    var_best_value: f64,
    mean_grad_norm: f64,
    var_grad_norm: f64,
    mean_iterations: f64,
    var_iterations: f64,
    mean_func_calls: f64,
    var_func_calls: f64,
    mean_grad_calls: f64,
    var_grad_calls: f64,
}

fn grad_norm<O: Objective<f64>>(f: &O, x: &[f64]) -> f64 {
    let g = f.gradient(x);
    g.iter().map(|t| t * t).sum::<f64>().sqrt()
}

fn banner_line(ch: char) {
    println!("{}", ch.to_string().repeat(W));
}

fn run_method<O: Objective<f64>>(
    method: &str,
    f: &O,
    start: &[f64],
    tol: f64,
    max_it: usize,
) -> OptimizeResult<f64> {
    match method {
        "bfgs" => {
            let mut opt = Bfgs::new(max_it);
            opt.tol = tol;
            opt.minimize(f, start)
        }
        "lbfgs" => {
            let mut opt = Lbfgs::new(max_it, 12);
            opt.tol = tol;
            opt.minimize(f, start)
        }
        "nlcg_fr" => {
            let mut opt = NonlinearCgFr::new(max_it);
            opt.tol = tol;
            opt.minimize(f, start)
        }
        "gd" => {
            let opt = GradientDescent::new(0.001_f64, max_it);
            opt.minimize(f, start)
        }
        "newton" => {
            let opt = NewtonMethod::new(max_it);
            opt.minimize(f, start)
        }
        "nelder_mead" => {
            let opt = NelderMead::new(max_it);
            opt.minimize(f, start)
        }
        _ => panic!("unknown method: {method}"),
    }
}

fn run_problem<O: Objective<f64>>(
    function: &str,
    dim: usize,
    f: &O,
    starts: &[Vec<f64>],
    tol: f64,
    max_it: usize,
    out: &mut Vec<TrialRow>,
) {
    let methods = ["bfgs", "lbfgs", "nlcg_fr", "gd", "newton", "nelder_mead"];
    for (start_id, start) in starts.iter().enumerate() {
        for method in methods {
            let r = run_method(method, f, start, tol, max_it);
            out.push(TrialRow {
                function: function.to_string(),
                dim,
                start_id: start_id + 1,
                method: method.to_string(),
                best_value: r.best_value,
                grad_norm: grad_norm(f, &r.best_x),
                iterations: r.iterations,
                func_calls: r.func_calls,
                grad_calls: r.grad_calls,
            });
        }
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64], mu: f64) -> f64 {
    xs.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / xs.len() as f64
}

fn aggregate(rows: &[TrialRow]) -> Vec<Aggregate> {
    let mut groups: BTreeMap<(String, usize, String), Vec<&TrialRow>> = BTreeMap::new();
    for row in rows {
        groups
            .entry((row.function.clone(), row.dim, row.method.clone()))
            .or_default()
            .push(row);
    }

    let mut out = Vec::new();
    for ((function, dim, method), g) in groups {
        let best_vals: Vec<f64> = g.iter().map(|r| r.best_value).collect();
        let grad_norms: Vec<f64> = g.iter().map(|r| r.grad_norm).collect();
        let iters: Vec<f64> = g.iter().map(|r| r.iterations as f64).collect();
        let f_calls: Vec<f64> = g.iter().map(|r| r.func_calls as f64).collect();
        let g_calls: Vec<f64> = g.iter().map(|r| r.grad_calls as f64).collect();

        let mean_best = mean(&best_vals);
        let mean_grad = mean(&grad_norms);
        let mean_it = mean(&iters);
        let mean_fc = mean(&f_calls);
        let mean_gc = mean(&g_calls);

        out.push(Aggregate {
            function,
            dim,
            method,
            trials: g.len(),
            mean_best_value: mean_best,
            var_best_value: variance(&best_vals, mean_best),
            mean_grad_norm: mean_grad,
            var_grad_norm: variance(&grad_norms, mean_grad),
            mean_iterations: mean_it,
            var_iterations: variance(&iters, mean_it),
            mean_func_calls: mean_fc,
            var_func_calls: variance(&f_calls, mean_fc),
            mean_grad_calls: mean_gc,
            var_grad_calls: variance(&g_calls, mean_gc),
        });
    }
    out
}

fn write_raw_csv(rows: &[TrialRow], path: &str) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "function,dim,start_id,method,best_value,grad_norm,iterations,func_calls,grad_calls"
    )?;
    for r in rows {
        writeln!(
            w,
            "{},{},{},{},{:.12e},{:.12e},{},{},{}",
            r.function,
            r.dim,
            r.start_id,
            r.method,
            r.best_value,
            r.grad_norm,
            r.iterations,
            r.func_calls,
            r.grad_calls
        )?;
    }
    Ok(())
}

fn write_agg_csv(rows: &[Aggregate], path: &str) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "function,dim,method,trials,mean_best_value,var_best_value,mean_grad_norm,var_grad_norm,mean_iterations,var_iterations,mean_func_calls,var_func_calls,mean_grad_calls,var_grad_calls"
    )?;
    for r in rows {
        writeln!(
            w,
            "{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            r.function,
            r.dim,
            r.method,
            r.trials,
            r.mean_best_value,
            r.var_best_value,
            r.mean_grad_norm,
            r.var_grad_norm,
            r.mean_iterations,
            r.var_iterations,
            r.mean_func_calls,
            r.var_func_calls,
            r.mean_grad_calls,
            r.var_grad_calls
        )?;
    }
    Ok(())
}

fn print_console(agg: &[Aggregate]) {
    banner_line('═');
    println!(
        "  {:^width$}",
        "lab2 — мультистарт статистика (3 старта на задачу)",
        width = W - 4
    );
    banner_line('═');
    println!(
        "  {:<12} {:<14} {:>8} {:>14} {:>14} {:>14} {:>14}",
        "function", "method", "trials", "mean iters", "var iters", "mean #f", "var #f"
    );
    banner_line('─');
    for r in agg {
        println!(
            "  {:<12} {:<14} {:>8} {:>14.3} {:>14.3} {:>14.3} {:>14.3}",
            format!("{}({})", r.function, r.dim),
            r.method,
            r.trials,
            r.mean_iterations,
            r.var_iterations,
            r.mean_func_calls,
            r.var_func_calls
        );
    }
    banner_line('═');
}

fn main() -> std::io::Result<()> {
    let tol = 1e-8;
    let max_it = 5000usize;
    let mut raw_rows = Vec::new();

    let rosen2 = RosenbrockN::new(2);
    let rosen2_starts = vec![vec![-1.2, 1.0], vec![-2.0, 2.0], vec![1.5, 2.25]];
    run_problem("rosen2", 2, &rosen2, &rosen2_starts, tol, max_it, &mut raw_rows);

    let rosen10 = RosenbrockN::new(10);
    let rosen10_starts = vec![
        (0..10).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect(),
        (0..10).map(|i| if i % 2 == 0 { -1.8 } else { 2.0 }).collect(),
        vec![1.2; 10],
    ];
    run_problem(
        "rosen10",
        10,
        &rosen10,
        &rosen10_starts,
        tol,
        max_it,
        &mut raw_rows,
    );

    let rast2 = RastriginN::new(2);
    let rast2_starts = vec![vec![0.5, -0.5], vec![3.0, 3.0], vec![-3.0, 2.0]];
    run_problem("rastrigin2", 2, &rast2, &rast2_starts, tol, max_it, &mut raw_rows);

    let rast8 = RastriginN::new(8);
    let rast8_starts = vec![
        vec![0.5; 8],
        vec![2.5; 8],
        vec![-2.5, 2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5],
    ];
    run_problem("rastrigin8", 8, &rast8, &rast8_starts, tol, max_it, &mut raw_rows);

    let desmos2 = DesmosBadie2d;
    let desmos2_starts = vec![vec![0.0, 5.0], vec![1.5, 7.0], vec![-1.0, 3.0]];
    run_problem("desmos2d", 2, &desmos2, &desmos2_starts, tol, max_it, &mut raw_rows);

    let desmos8 = DesmosBadieBlocks::new(8);
    let desmos8_starts = vec![
        (0..8).map(|i| if i % 2 == 0 { 0.0 } else { 5.0 }).collect(),
        (0..8).map(|i| if i % 2 == 0 { 1.0 } else { 7.0 }).collect(),
        (0..8).map(|i| if i % 2 == 0 { -1.0 } else { 3.0 }).collect(),
    ];
    run_problem("desmos8", 8, &desmos8, &desmos8_starts, tol, max_it, &mut raw_rows);

    let agg_rows = aggregate(&raw_rows);
    print_console(&agg_rows);

    create_dir_all("results")?;
    write_raw_csv(&raw_rows, "results/multistart_raw.csv")?;
    write_agg_csv(&agg_rows, "results/multistart_agg.csv")?;

    Ok(())
}
