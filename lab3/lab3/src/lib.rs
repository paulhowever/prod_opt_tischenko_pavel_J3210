pub mod data;
pub mod metrics;
pub mod mlp;
pub mod nn_objective;
pub mod optim;
pub mod train;

pub use data::{load_csv, stratified_train_val, stratified_train_val_test, Dataset, Normalizer};
pub use lab1_met_opt::objective::Objective;
pub use mlp::MlpLayout;
pub use metrics::{confusion, confusion_report, f1_binary};
pub use train::{train, OptimMethod, TrainConfig, TrainReport};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SavedModel {
    pub layout: MlpLayout,
    pub theta: Vec<f64>,
    pub normalizer: Normalizer,
    pub threshold: f64,
}

impl SavedModel {
    pub fn predict_probs(&self, rows: &[Vec<f64>]) -> Vec<f64> {
        let mut xs: Vec<Vec<f64>> = rows.to_vec();
        self.normalizer.transform(&mut xs);
        let logits = crate::mlp::logits_batch(&self.layout, &self.theta, &xs);
        crate::metrics::logits_to_probs(&logits)
    }

    pub fn predict_labels(&self, rows: &[Vec<f64>]) -> Vec<u8> {
        let p = self.predict_probs(rows);
        crate::metrics::predict_threshold(&p, self.threshold)
    }
}
