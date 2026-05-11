use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use csv::ReaderBuilder;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct Dataset {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<u8>,
    pub feature_dim: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Normalizer {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl Normalizer {
    pub fn fit(rows: &[Vec<f64>], dim: usize) -> Self {
        let mut mean = vec![0.0; dim];
        let n = rows.len().max(1) as f64;
        for row in rows {
            for j in 0..dim {
                mean[j] += row[j];
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![0.0; dim];
        for row in rows {
            for j in 0..dim {
                let d = row[j] - mean[j];
                var[j] += d * d;
            }
        }
        let std: Vec<f64> = var
            .into_iter()
            .map(|v| (v / n).sqrt().max(1e-8))
            .collect();
        Self { mean, std }
    }

    pub fn transform_row(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, v)| (v - self.mean[j]) / self.std[j])
            .collect()
    }

    pub fn transform(&self, rows: &mut [Vec<f64>]) {
        for row in rows.iter_mut() {
            for j in 0..row.len() {
                row[j] = (row[j] - self.mean[j]) / self.std[j];
            }
        }
    }
}

pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<Dataset, String> {
    let f = File::open(path.as_ref()).map_err(|e| e.to_string())?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(f));
    let headers = rdr
        .headers()
        .map_err(|e| e.to_string())?
        .clone();
    let mut feature_cols: Vec<usize> = Vec::new();
    let mut target_col: Option<usize> = None;
    for (i, h) in headers.iter().enumerate() {
        let h = h.trim();
        if h == "target" {
            target_col = Some(i);
        } else if h.starts_with("feature_") {
            feature_cols.push(i);
        }
    }
    feature_cols.sort();
    let target_col = target_col.ok_or_else(|| "missing target column".to_string())?;
    if feature_cols.is_empty() {
        return Err("no feature columns".to_string());
    }
    let dim = feature_cols.len();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| e.to_string())?;
        let mut row = vec![0.0; dim];
        for (j, &ci) in feature_cols.iter().enumerate() {
            let v: f64 = rec
                .get(ci)
                .ok_or_else(|| "short row".to_string())?
                .trim()
                .parse()
                .map_err(|_| "bad float".to_string())?;
            row[j] = v;
        }
        let ty = rec
            .get(target_col)
            .ok_or_else(|| "short row".to_string())?
            .trim();
        let yi: u8 = if ty == "1" || ty == "1.0" {
            1
        } else if ty == "0" || ty == "0.0" {
            0
        } else {
            return Err(format!("bad target: {}", ty));
        };
        x.push(row);
        y.push(yi);
    }
    Ok(Dataset {
        feature_dim: dim,
        x,
        y,
    })
}

pub fn stratified_train_val(
    ds: Dataset,
    train_frac: f64,
    seed: u64,
) -> (Dataset, Dataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut i0: Vec<usize> = Vec::new();
    let mut i1: Vec<usize> = Vec::new();
    for (i, &yi) in ds.y.iter().enumerate() {
        if yi == 0 {
            i0.push(i);
        } else {
            i1.push(i);
        }
    }
    i0.shuffle(&mut rng);
    i1.shuffle(&mut rng);
    let mut n0_tr = ((i0.len() as f64) * train_frac).floor() as usize;
    let mut n1_tr = ((i1.len() as f64) * train_frac).floor() as usize;
    n0_tr = n0_tr.min(i0.len());
    n1_tr = n1_tr.min(i1.len());
    if n0_tr == i0.len() && i0.len() > 1 {
        n0_tr -= 1;
    }
    if n1_tr == i1.len() && i1.len() > 1 {
        n1_tr -= 1;
    }
    let mut train_idx: Vec<usize> = Vec::new();
    let mut val_idx: Vec<usize> = Vec::new();
    for (k, &idx) in i0.iter().enumerate() {
        if k < n0_tr {
            train_idx.push(idx);
        } else {
            val_idx.push(idx);
        }
    }
    for (k, &idx) in i1.iter().enumerate() {
        if k < n1_tr {
            train_idx.push(idx);
        } else {
            val_idx.push(idx);
        }
    }
    train_idx.shuffle(&mut rng);
    val_idx.shuffle(&mut rng);
    let mut xt = Vec::with_capacity(train_idx.len());
    let mut yt = Vec::with_capacity(train_idx.len());
    for &i in &train_idx {
        xt.push(ds.x[i].clone());
        yt.push(ds.y[i]);
    }
    let mut xv = Vec::with_capacity(val_idx.len());
    let mut yv = Vec::with_capacity(val_idx.len());
    for &i in &val_idx {
        xv.push(ds.x[i].clone());
        yv.push(ds.y[i]);
    }
    let dim = ds.feature_dim;
    (
        Dataset {
            x: xt,
            y: yt,
            feature_dim: dim,
        },
        Dataset {
            x: xv,
            y: yv,
            feature_dim: dim,
        },
    )
}

/// Разбиение 80/20 по ТЗ на `train_val` и `test`, затем доля `val_frac_of_train_val` от `train_val` уходит в валидацию (подбор порога).
///
/// При `test_frac = 0.2`, `val_frac_of_train_val = 0.2`: ~64% train, ~16% val, ~20% test.
pub fn stratified_train_val_test(
    ds: Dataset,
    test_frac: f64,
    val_frac_of_train_val: f64,
    seed: u64,
) -> (Dataset, Dataset, Dataset) {
    let tv_frac = (1.0 - test_frac.clamp(1e-9, 0.999999)).max(1e-9);
    let (train_val, test_ds) = stratified_train_val(ds, tv_frac, seed);
    let inner_train_frac =
        (1.0 - val_frac_of_train_val.clamp(1e-9, 0.999999)).max(1e-9);
    let (train, val) =
        stratified_train_val(train_val, inner_train_frac, seed.wrapping_add(1));
    (train, val, test_ds)
}
