# NBA Injury Severity Classification — Source Code README

This repository/source-code folder reproduces the main results in the final report for the NBA injury severity classification project.

## Files

- `label_injuries.py` — creates rule-based severity labels (`MINOR`, `MODERATE`, `SEVERE`) from raw injury records.
- `split_data.py` — creates the shared stratified train/dev/test split used by all models.
- `bert_classifier(1).py` — runs the zero-shot DistilBERT audit against the rule-based labels.
- `train_distilbert(1).py` — fine-tunes `distilbert-base-uncased` on the injury severity classification task.
- `ablated_injury_severity_model.py` — removes high-signal leakage terms and trains/evaluates TF-IDF Naive Bayes and Logistic Regression models.

## Expected input data

The scripts expect the following input files in the same folder:

- `nba_injuries_finalized.csv` for the labeling pipeline. This project used publicly available NBA injury data collected from Basketball Reference.
- `nba_injuries_labeled.csv` after running `label_injuries.py`.
- `split_train.csv`, `split_dev.csv`, and `split_test.csv` after running `split_data.py`.

The submitted report uses 11,107 labeled injury records from Basketball Reference and a stratified 70/15/15 split:

- Train: 7,774 records
- Dev: 1,666 records
- Test: 1,667 records

## Environment setup

Recommended Python version: Python 3.10+

Install dependencies:

```bash
pip install pandas numpy scikit-learn transformers torch seaborn matplotlib
```

A GPU is recommended for `train_distilbert(1).py`, but the TF-IDF and ablation scripts can run on CPU.

## Run order

Run the scripts in this order from the project folder.

### 1. Create severity labels

```bash
python label_injuries.py
```

Expected outputs:

- `nba_injuries_labeled.csv`
- `nba_injuries_omitted.csv`

### 2. Create shared data splits

```bash
python split_data.py
```

Expected outputs:

- `split_train.csv`
- `split_dev.csv`
- `split_test.csv`

### 3. Run zero-shot BERT audit

```bash
python "bert_classifier(1).py"
```

Expected outputs:

- `bert_audit_results.csv`
- `bert_confusion_matrix.png`

### 4. Fine-tune DistilBERT

```bash
python "train_distilbert(1).py"
```

Expected output folder:

- `distilbert_output/`

Important files inside that folder:

- `best_model/`
- `predictions_test.csv`
- `confusion_matrix.png`
- `training_log.csv`

### 5. Run ablated leakage-check models

```bash
python ablated_injury_severity_model.py
```

Expected outputs:

- `ablated_nb_tuning_grid.csv`
- `ablated_lr_tuning_grid.csv`
- `ablated_model_results_summary.csv`
- `ablated_confusion_matrices_long.csv`
- `ablated_top_lr_features.csv`

## Key reported results

- Majority baseline: predicts `MINOR` for every example; approximately 68% accuracy because the dataset is imbalanced.
- Zero-shot DistilBERT audit: 0.26 accuracy and 0.1925 macro F1.
- Fine-tuned DistilBERT on original text: approximately 0.9966 macro F1, likely inflated by keyword leakage.
- Ablated TF-IDF Logistic Regression: 0.8476 accuracy and 0.7826 macro F1.

## Notes on reproducibility

The scripts use fixed random seeds where applicable, especially for the data split and model tuning. Small differences may occur across hardware, PyTorch versions, or GPU/CPU settings during DistilBERT fine-tuning.
