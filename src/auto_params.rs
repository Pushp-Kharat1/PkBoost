use crate::tree::TreeParams;

pub struct DataStats {
    pub n_rows: usize,
    pub n_cols: usize,
    pub pos_ratio: f64,          // positives / total
    pub missing_rate: f64,       // NaN / total cells
    pub max_cat_cards: usize,    // highest #unique in any cat column
}

impl DataStats {
    pub fn from_slices(
        n_rows: usize,
        n_cols: usize,
        pos: usize, 
        missing: usize,
        max_card: usize,
    ) -> Self {
        let total_cells = (n_rows * n_cols) as f64;
        Self {
            n_rows,
            n_cols,
            pos_ratio: pos as f64 / n_rows as f64,
            missing_rate: missing as f64 / total_cells,
            max_cat_cards: max_card,
        }
    }
}

pub fn auto_params(stats: &DataStats) -> AutoHyperParams {
    let imbalance = 1.0 - stats.pos_ratio;
    let scale_pos_weight = imbalance / stats.pos_ratio.max(1e-6);

    let learning_rate: f64 = if stats.n_rows < 20_000 { 0.05 } else { 0.02 };
    let n_estimators = ((2_000_f64).ln() / learning_rate.ln()).ceil() as usize;
    let n_estimators = n_estimators.max(100);

    let _max_depth = match stats.n_cols {
        0..=20 => 6,
        21..=100 => 5,
        _ => 4,
    };

    let min_child_weight = 10_f64.max(stats.pos_ratio * stats.n_rows as f64 * 0.01);
    let gamma = if stats.missing_rate > 0.1 { 0.1 } else { 0.0 };
    let reg_lambda = 1.0 + stats.max_cat_cards as f64 * 0.05;
    let mi_weight = 0.3 * (-stats.pos_ratio.ln()).exp();
    let early_stopping = (100_f64 / learning_rate).ceil() as usize;

    AutoHyperParams {
        base: TreeParams {
            min_samples_split: 20,
            min_child_weight,
            reg_lambda,
            gamma,
            mi_weight,
            n_bins_per_feature: vec![32; stats.n_cols],
            feature_elimination_threshold: 0.01, // ADD THIS LINE
        },
        n_estimators,
        learning_rate,
        early_stopping_rounds: early_stopping,
        scale_pos_weight,
        subsample: 0.8,
        colsample_bytree: 0.8,
    }
}

pub struct AutoHyperParams {
    pub base: TreeParams,
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub early_stopping_rounds: usize,
    pub scale_pos_weight: f64,
    pub subsample: f64,
    pub colsample_bytree: f64,
}