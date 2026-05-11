use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use lab3_met_opt::{
    confusion_report, load_csv, stratified_train_val, stratified_train_val_test, train, MlpLayout,
    Normalizer, OptimMethod, SavedModel, TrainConfig,
};
use rand::seq::SliceRandom;
use rand::SeedableRng;

fn usage() -> ! {
    eprintln!(
        "usage:\n  lab3_met_opt train ... [--confusion]\n  lab3_met_opt bench ...\n  lab3_met_opt eval ... [--confusion]\n  lab3_met_opt sanity --csv <path> [--seed n] [--hidden n] [--residual n]\n\n\
         train: ... [--three-way-split] [--no-balance] [--lbfgs-path-stride n] [--sd-path-stride n]\n\
         (по умолчанию строгие 80/20 train/test по ТЗ; --three-way-split вводит отдельный val для подбора порога)\n\
         eval: --eval-on full|val|test\n"
    );
    std::process::exit(1);
}

fn get_arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn has_flag(argv: &[String], name: &str) -> bool {
    argv.iter().any(|a| a == name)
}

fn print_split_confusion(
    title: &str,
    layout: &MlpLayout,
    theta: &[f64],
    norm: &Normalizer,
    xs: &[Vec<f64>],
    y: &[u8],
    thr: f64,
) {
    let mut x = xs.to_vec();
    norm.transform(&mut x);
    let logits = lab3_met_opt::mlp::logits_batch(layout, theta, &x);
    let probs = lab3_met_opt::metrics::logits_to_probs(&logits);
    let pred = lab3_met_opt::metrics::predict_threshold(&probs, thr);
    println!("{}", confusion_report(title, y, &pred));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }
    match args[1].as_str() {
        "train" => cmd_train(&args[2..]),
        "bench" => cmd_bench(&args[2..]),
        "eval" => cmd_eval(&args[2..]),
        "sanity" => cmd_sanity(&args[2..]),
        _ => usage(),
    }
}

fn cmd_train(argv: &[String]) {
    let csv = get_arg(argv, "--csv").unwrap_or_else(|| usage());
    let method_s = get_arg(argv, "--method").unwrap_or("lbfgs");
    let method = OptimMethod::parse(method_s).unwrap_or_else(|| {
        eprintln!("unknown method");
        usage();
    });
    let seed: u64 = get_arg(argv, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let hidden: usize = get_arg(argv, "--hidden")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let residual: usize = get_arg(argv, "--residual")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let l2: f64 = get_arg(argv, "--l2").and_then(|s| s.parse().ok()).unwrap_or(1e-4);
    let lbfgs_path_stride: usize = get_arg(argv, "--lbfgs-path-stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let sd_path_stride: usize = get_arg(argv, "--sd-path-stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let out_json = get_arg(argv, "--out-json").map(PathBuf::from);
    let weights_path = get_arg(argv, "--weights").map(PathBuf::from);
    let curve_path = get_arg(argv, "--curve").map(PathBuf::from);
    let ds = load_csv(csv).unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    });
    let three_way = has_flag(argv, "--three-way-split");
    let (tr, va, te_opt) = if three_way {
        let (tr, va, te) = stratified_train_val_test(ds, 0.2, 0.2, seed);
        (tr, va, Some(te))
    } else {
        // ТЗ: train/test = 80/20. Отдельного val нет: порог калибруется на той же 20%-й тест-выборке,
        // на ней же берётся F1 для ТЗ-формулы (val == test).
        let (tr, te) = stratified_train_val(ds, 0.8, seed);
        (tr, te.clone(), Some(te))
    };
    let mut cfg = TrainConfig::default();
    cfg.method = method;
    cfg.seed = seed;
    cfg.hidden = hidden;
    cfg.residual_blocks = residual;
    cfg.l2 = l2;
    cfg.balance_binary_loss = !has_flag(argv, "--no-balance");
    cfg.lbfgs_path_stride = lbfgs_path_stride;
    cfg.sd_path_stride = sd_path_stride;
    let test_ref = te_opt.as_ref();
    let layout = MlpLayout {
        in_dim: tr.feature_dim,
        hidden,
        residual_blocks: residual,
    };
    let (theta, norm, report) = train(&tr, &va, test_ref, &cfg);
    let mode_str = if three_way { "three-way 64/16/20" } else { "two-way 80/20 (val == test)" };
    if let (Some(f1_te), Some(f1_te05)) = (report.f1_test, report.f1_test_default_threshold) {
        println!(
            "{} [{}] | F1_val {:.12} F1_test {:.12} (thr={:.12}) F1_test@0.5 {:.12} | iters {} f {} g {} t {:.2}s",
            report.method,
            mode_str,
            report.f1_val,
            f1_te,
            report.threshold,
            f1_te05,
            report.iterations,
            report.func_calls,
            report.grad_calls,
            report.seconds
        );
    } else {
        println!(
            "{} [{}] | F1_val {:.12} thr {:.12} iters {} f {} g {} t {:.2}s",
            report.method,
            mode_str,
            report.f1_val,
            report.threshold,
            report.iterations,
            report.func_calls,
            report.grad_calls,
            report.seconds
        );
    }
    if has_flag(argv, "--confusion") {
        print_split_confusion(
            "val  pred (thr=val)",
            &layout,
            &theta,
            &norm,
            &va.x,
            &va.y,
            report.threshold,
        );
        print_split_confusion(
            "val  pred (thr=0.5)",
            &layout,
            &theta,
            &norm,
            &va.x,
            &va.y,
            0.5,
        );
        if let Some(te) = te_opt.as_ref() {
            print_split_confusion(
                "test pred (thr=val)",
                &layout,
                &theta,
                &norm,
                &te.x,
                &te.y,
                report.threshold,
            );
            print_split_confusion(
                "test pred (thr=0.5)",
                &layout,
                &theta,
                &norm,
                &te.x,
                &te.y,
                0.5,
            );
        }
    }
    if let Some(p) = out_json {
        let j = serde_json::to_string_pretty(&report).unwrap();
        std::fs::write(p, j).unwrap();
    }
    if let Some(p) = weights_path {
        let sm = SavedModel {
            layout: lab3_met_opt::MlpLayout {
                in_dim: tr.feature_dim,
                hidden,
                residual_blocks: residual,
            },
            theta,
            normalizer: norm,
            threshold: report.threshold,
        };
        let j = serde_json::to_string_pretty(&sm).unwrap();
        std::fs::write(p, j).unwrap();
    }
    if let Some(p) = curve_path {
        let mut f = File::create(p).unwrap();
        writeln!(f, "iter,loss").unwrap();
        for pt in &report.loss_curve {
            writeln!(f, "{},{}", pt.iter, pt.loss).unwrap();
        }
    }
}

fn cmd_bench(argv: &[String]) {
    let csv1 = get_arg(argv, "--csv1").unwrap_or_else(|| usage());
    let csv2 = get_arg(argv, "--csv2").unwrap_or_else(|| usage());
    let out_dir = get_arg(argv, "--out-dir").map(PathBuf::from);
    let seed: u64 = get_arg(argv, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let hidden: usize = get_arg(argv, "--hidden")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let residual: usize = get_arg(argv, "--residual")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let l2: f64 = get_arg(argv, "--l2").and_then(|s| s.parse().ok()).unwrap_or(1e-4);
    let lbfgs_path_stride: usize = get_arg(argv, "--lbfgs-path-stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let sd_path_stride: usize = get_arg(argv, "--sd-path-stride")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let three_way = has_flag(argv, "--three-way-split");
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut f1_test_d1 = [0.0f64; 2];
    let mut f1_test_d2 = [0.0f64; 2];
    for (di, path) in [(0usize, csv1), (1, csv2)] {
        let ds = load_csv(path).unwrap_or_else(|e| {
            eprintln!("{}", e);
            std::process::exit(1);
        });
        for (mi, &method) in [OptimMethod::Lbfgs, OptimMethod::SteepestArmijo]
            .iter()
            .enumerate()
        {
            let (tr, va, te) = if three_way {
                stratified_train_val_test(ds.clone(), 0.2, 0.2, seed)
            } else {
                let (tr, te) = stratified_train_val(ds.clone(), 0.8, seed);
                (tr, te.clone(), te)
            };
            let mut cfg = TrainConfig::default();
            cfg.method = method;
            cfg.seed = seed;
            cfg.hidden = hidden;
            cfg.residual_blocks = residual;
            cfg.l2 = l2;
            cfg.balance_binary_loss = !has_flag(argv, "--no-balance");
            cfg.lbfgs_path_stride = lbfgs_path_stride;
            cfg.sd_path_stride = sd_path_stride;
            let (_theta, _norm, report) = train(&tr, &va, Some(&te), &cfg);
            let f1_test = report.f1_test.unwrap_or(f64::NAN);
            if di == 0 {
                f1_test_d1[mi] = f1_test;
            } else {
                f1_test_d2[mi] = f1_test;
            }
            rows.push(serde_json::to_value(&report).unwrap());
        }
    }
    let est_lbfgs_test = 0.3 * f1_test_d1[0] + 0.3 * f1_test_d2[0];
    let est_sd_test = 0.3 * f1_test_d1[1] + 0.3 * f1_test_d2[1];
    let mode_str = if three_way { "three-way 64/16/20" } else { "two-way 80/20" };
    println!(
        "split: {} | F1_test(d1) lbfgs {:.12} sd {:.12}",
        mode_str, f1_test_d1[0], f1_test_d1[1]
    );
    println!(
        "split: {} | F1_test(d2) lbfgs {:.12} sd {:.12}",
        mode_str, f1_test_d2[0], f1_test_d2[1]
    );
    println!(
        "partial (без d3): 0.3*F1_test(d1)+0.3*F1_test(d2) — lbfgs {:.12}, steepest {:.12}",
        est_lbfgs_test, est_sd_test
    );
    println!(
        "итог по ТЗ: 0.3*F1(d1)+0.3*F1(d2)+0.4*F1(d3); закрытый d3 на защите. Порог зачёта 0.55 по полной формуле."
    );
    let summary = serde_json::json!({
        "runs": rows,
        "mode": mode_str,
        "partial_lbfgs_d1_d2_test": est_lbfgs_test,
        "partial_steepest_d1_d2_test": est_sd_test,
    });
    if let Some(dir) = out_dir {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("bench_summary.json"),
            serde_json::to_string_pretty(&summary).unwrap(),
        )
        .unwrap();
    }
}

fn cmd_eval(argv: &[String]) {
    let csv = get_arg(argv, "--csv").unwrap_or_else(|| usage());
    let weights = get_arg(argv, "--weights").unwrap_or_else(|| usage());
    let on = get_arg(argv, "--eval-on").unwrap_or("full");
    let seed: u64 = get_arg(argv, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let raw = std::fs::read_to_string(weights).unwrap();
    let model: SavedModel = serde_json::from_str(&raw).unwrap();
    let ds = load_csv(csv).unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    });
    let (rows, y_true, note) = match on {
        "val" => {
            let (_tr, va) = stratified_train_val(ds.clone(), 0.8, seed);
            (va.x, va.y, "stratified val (~20% per class), same seed as train default")
        }
        "test" => {
            let (_tr, _va, te) = stratified_train_val_test(ds.clone(), 0.2, 0.2, seed);
            (te.x, te.y, "stratified hold-out test (~20%), same 3-way split as train default")
        }
        "full" => {
            eprintln!(
                "warning: F1 на всех строках CSV (train+val+test в одном файле). Для честной оценки: --eval-on val или --eval-on test --seed {}",
                seed
            );
            (ds.x.clone(), ds.y.clone(), "all rows")
        }
        _ => {
            eprintln!("--eval-on must be full|val|test");
            usage();
        }
    };
    let probs = model.predict_probs(&rows);
    let pred = model.predict_labels(&rows);
    let f1 = lab3_met_opt::metrics::f1_binary(&y_true, &pred);
    println!("F1 {:.12} ({})", f1, note);
    if has_flag(argv, "--confusion") {
        println!(
            "{}",
            confusion_report("eval subset (model thr)", &y_true, &pred)
        );
    }
    for i in 0..probs.len().min(10) {
        println!("{} p {:.12} y {} yhat {}", i, probs[i], y_true[i], pred[i]);
    }
}

/// Контрольный прогон: чистые train-метки vs перемешанные train-метки (val/test без изменений).
fn cmd_sanity(argv: &[String]) {
    let csv = get_arg(argv, "--csv").unwrap_or_else(|| usage());
    let seed: u64 = get_arg(argv, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let hidden: usize = get_arg(argv, "--hidden")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let residual: usize = get_arg(argv, "--residual")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let ds = load_csv(csv).unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    });
    let (tr, va, te) = stratified_train_val_test(ds, 0.2, 0.2, seed);
    let mut cfg = TrainConfig::default();
    cfg.method = OptimMethod::Lbfgs;
    cfg.seed = seed;
    cfg.hidden = hidden;
    cfg.residual_blocks = residual;
    let layout = MlpLayout {
        in_dim: tr.feature_dim,
        hidden,
        residual_blocks: residual,
    };

    println!("--- sanity: {} | seed={} train={} val={} test={} ---", csv, seed, tr.x.len(), va.x.len(), te.x.len());

    let (theta, norm, rep_ok) = train(&tr, &va, Some(&te), &cfg);
    let f1_ok = rep_ok.f1_test.unwrap_or(f64::NAN);
    println!(
        "[OK labels] F1_val={:.12} F1_test={:.12} thr={:.12}",
        rep_ok.f1_val, f1_ok, rep_ok.threshold
    );
    print_split_confusion(
        "[OK] test (thr=val)",
        &layout,
        &theta,
        &norm,
        &te.x,
        &te.y,
        rep_ok.threshold,
    );
    print_split_confusion(
        "[OK] test (thr=0.5)",
        &layout,
        &theta,
        &norm,
        &te.x,
        &te.y,
        0.5,
    );

    let mut tr_shuf = tr.clone();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(0xC0FFEE));
    tr_shuf.y.shuffle(&mut rng);

    let (theta_s, norm_s, rep_bad) = train(&tr_shuf, &va, Some(&te), &cfg);
    let f1_bad = rep_bad.f1_test.unwrap_or(f64::NAN);
    println!(
        "[SHUFFLED train y] F1_val={:.12} F1_test={:.12} thr={:.12}",
        rep_bad.f1_val, f1_bad, rep_bad.threshold
    );
    print_split_confusion(
        "[shuf] test (thr=val)",
        &layout,
        &theta_s,
        &norm_s,
        &te.x,
        &te.y,
        rep_bad.threshold,
    );
    print_split_confusion(
        "[shuf] test (thr=0.5)",
        &layout,
        &theta_s,
        &norm_s,
        &te.x,
        &te.y,
        0.5,
    );

    const MIN_OK: f64 = 0.85;
    const MAX_SHUFFLED: f64 = 0.72;
    let pass_clean = f1_ok.is_finite() && f1_ok >= MIN_OK;
    let pass_noise = f1_bad.is_finite() && f1_bad <= MAX_SHUFFLED;

    if pass_clean && pass_noise {
        println!(
            "sanity: OK  F1_test(clean)={:.6}  F1_test(shuffled_train_y)={:.6}",
            f1_ok, f1_bad
        );
        std::process::exit(0);
    }
    println!(
        "sanity: FAIL  F1_test(clean)={:.6} (want >={:.2})  F1_test(shuffled_train_y)={:.6} (want <={:.2})",
        f1_ok, MIN_OK, f1_bad, MAX_SHUFFLED
    );
    std::process::exit(1);
}
