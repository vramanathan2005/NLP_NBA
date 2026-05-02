"""
Zero-shot BERT audit vs keyword labels
LIN 371 | Dhruv Kumar & Varun Ramanathan

Draws a stratified 33.3% sample from each severity class,
runs typeform/distilbert-base-uncased-mnli zero-shot classification,
and compares against keyword labels with a full report.
"""

import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_PATH   = "nba_injuries_labeled.csv"
OUTPUT_PATH  = "bert_audit_results.csv"
SAMPLE_FRAC  = 0.333
RANDOM_SEED  = 42
CANDIDATE_LABELS = ["minor injury", "moderate injury", "severe injury"]
LABEL_MAP = {
    "minor injury":    "MINOR",
    "moderate injury": "MODERATE",
    "severe injury":   "SEVERE",
}

# ── Load & sample ──────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)

sample = (
    df.groupby("severity", group_keys=False)
      .apply(lambda g: g.sample(frac=SAMPLE_FRAC, random_state=RANDOM_SEED))
      .reset_index(drop=True)
)
print(f"Sample size: {len(sample)}")
print(sample["severity"].value_counts())

# ── Build input text ───────────────────────────────────────────────────────────
sample["input_text"] = (
    sample["notes"].fillna("") + " " +
    sample["injury_type"].fillna("") + " " +
    sample["body_part"].fillna("")
).str.strip()

# ── Load DistilBERT pipeline ───────────────────────────────────────────────────
print("Loading typeform/distilbert-base-uncased-mnli ...")
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1
)

# ── Run inference ──────────────────────────────────────────────────────────────
print("Running zero-shot classification ...")
bert_preds = []
for i, text in enumerate(sample["input_text"]):
    result = classifier(text, CANDIDATE_LABELS, truncation=True)
    top_label = result["labels"][0]
    bert_preds.append(LABEL_MAP[top_label])
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(sample)} done")

sample["bert_pred"] = bert_preds

# ── Evaluation ─────────────────────────────────────────────────────────────────
y_true = sample["severity"]
y_pred = sample["bert_pred"]

print("\n=== Classification Report (DistilBERT vs Keyword Labels) ===")
print(classification_report(y_true, y_pred, labels=["MINOR","MODERATE","SEVERE"]))

macro_f1 = f1_score(y_true, y_pred, average="macro")
print(f"Macro F1: {macro_f1:.4f}")
print()

if macro_f1 > 0.90:
    print("⚠️  WARNING: Macro F1 > 0.90 — labels may be too easy (keyword leakage).")
    print("   Consider switching to manual labeling before proceeding.")
else:
    print("✅ Macro F1 below 0.90 — labels have meaningful complexity, good to proceed.")

# ── Confusion matrix ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=["MINOR","MODERATE","SEVERE"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["MINOR","MODERATE","SEVERE"],
            yticklabels=["MINOR","MODERATE","SEVERE"], ax=ax)
ax.set_xlabel("Model Prediction")
ax.set_ylabel("Keyword Label")
ax.set_title("Confusion Matrix: DistilBERT vs Keyword Labels")
plt.tight_layout()
plt.savefig("bert_confusion_matrix.png", dpi=150)
print("Confusion matrix saved to bert_confusion_matrix.png")

# ── Save results ───────────────────────────────────────────────────────────────
sample.to_csv(OUTPUT_PATH, index=False)
print(f"Full results saved to {OUTPUT_PATH}")