pub fn confusion(y_true: &[u8], y_pred: &[u8]) -> (usize, usize, usize, usize) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        match (t, p) {
            (1, 1) => tp += 1,
            (0, 1) => fp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fn_ += 1,
            _ => {}
        }
    }
    (tp, fp, tn, fn_)
}

/// Текстовая сводка TP/FP/TN/FN и F1 (без округления «до красоты» — полный float).
pub fn confusion_report(title: &str, y_true: &[u8], y_pred: &[u8]) -> String {
    let (tp, fp, tn, fn_) = confusion(y_true, y_pred);
    let f1 = f1_binary(y_true, y_pred);
    format!(
        "{}  TP={} FP={} TN={} FN={}  F1={:.12}",
        title, tp, fp, tn, fn_, f1
    )
}

pub fn f1_binary(y_true: &[u8], y_pred: &[u8]) -> f64 {
    let (tp, fp, _, fn_) = confusion(y_true, y_pred);
    let p = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let r = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    };
    if p + r <= 1e-18 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

pub fn logits_to_probs(logits: &[f64]) -> Vec<f64> {
    logits
        .iter()
        .copied()
        .map(crate::mlp::prob_from_logit)
        .collect()
}

pub fn predict_threshold(probs: &[f64], thr: f64) -> Vec<u8> {
    probs.iter().map(|&p| if p >= thr { 1 } else { 0 }).collect()
}

pub fn best_threshold_f1(y_true: &[u8], probs: &[f64]) -> (f64, f64) {
    let mut best_t = 0.5f64;
    let mut best_f1 = f1_binary(y_true, &predict_threshold(probs, best_t));
    let steps = 101usize;
    let denom = (steps.saturating_sub(1)).max(1) as f64;
    for i in 0..steps {
        let t = i as f64 / denom;
        let pred = predict_threshold(probs, t);
        let f = f1_binary(y_true, &pred);
        if f > best_f1 {
            best_f1 = f;
            best_t = t;
        }
    }
    (best_t, best_f1)
}
