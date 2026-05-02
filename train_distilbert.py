"""
Fine-tunes distilbert on the NBA injury severity task.
Reads the pre-split train/dev/test CSVs produced by split_data.py.

outputs:
  best_model/ — best checkpoint by dev macro F1
  predictions_test.csv — test-set predictions with confidence scores
  confusion_matrix.png — confusion matrix on test set
  training_log.csv     — per-epoch train loss, dev loss, dev macro F1
"""

import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

#config
TRAIN_PATH   = "split_train.csv"
DEV_PATH     = "split_dev.csv"
TEST_PATH    = "split_test.csv"
OUTPUT_DIR   = "./distilbert_output"
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
RANDOM_SEED  = 42

LABEL2ID = {"MINOR": 0, "MODERATE": 1, "SEVERE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3

#reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

#ds
class InjuryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts  = df["input_text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


#class weights for imbalanced data
def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["label_id"].value_counts().sort_index()
    total  = len(train_df)
    # weight = total / (num_classes * class_count)  — standard inverse freq
    weights = total / (NUM_LABELS * counts.values)
    weights = torch.tensor(weights, dtype=torch.float)
    print("class weights:")
    for i, w in enumerate(weights):
        print(f"  {ID2LABEL[i]:<10} {w:.4f}")
    return weights


#eval function
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss  = total_loss / len(loader)
    macro_f1  = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


#main func
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #load splits
    print("loading splits...")
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df   = pd.read_csv(DEV_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"  Train: {len(train_df)}  Dev: {len(dev_df)}  Test: {len(test_df)}")
    #tokenizer
    print(f"\nloading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    #ds and loaders 
    train_ds = InjuryDataset(train_df, tokenizer, MAX_LEN)
    dev_ds   = InjuryDataset(dev_df,   tokenizer, MAX_LEN)
    test_ds  = InjuryDataset(test_df,  tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)
    #model
    print(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # loss with class weights for imbalance
    class_weights = compute_class_weights(train_df).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    #adam optimiizer
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\ntraining for {EPOCHS} epochs  |  {total_steps} total steps  "
          f"|  {warmup_steps} warmup steps\n")

    #training loop
    best_dev_f1   = 0.0
    training_log  = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()

            #read that gradient clipping is good for preventing exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

            if step % 100 == 0:
                print(f"epoch {epoch} | Step {step}/{len(train_loader)} "
                      f"| loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        dev_loss, dev_f1, _, _ = evaluate(model, dev_loader, loss_fn, device)

        print(f"\nepoch {epoch} summary:")
        print(f"train loss : {avg_train_loss:.4f}")
        print(f"dev loss   : {dev_loss:.4f}")
        print(f"dev Macro F1: {dev_f1:.4f}")

        training_log.append({
            "epoch":          epoch,
            "train_loss":     avg_train_loss,
            "dev_loss":       dev_loss,
            "dev_macro_f1":   dev_f1,
        })

        #save best checkpoint
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_path   = os.path.join(OUTPUT_DIR, "best_model")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"new best model saved (dev F1={dev_f1:.4f})")
        print()

    #save training log
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)
    print(f"log saved.")

    #final eval on test set
    print("\nLoading best model for test evaluation")
    best_model = DistilBertForSequenceClassification.from_pretrained(
        os.path.join(OUTPUT_DIR, "best_model")
    )
    best_model.to(device)

    _, test_f1, test_preds, test_labels = evaluate(
        best_model, test_loader, loss_fn, device
    )

    label_names = ["MINOR", "MODERATE", "SEVERE"]
    print("\ntest set classification report")
    print(classification_report(
        test_labels, test_preds,
        target_names=label_names,
    ))
    print(f"test Macro F1: {test_f1:.4f}")

    # ── Confusion matrix ──
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"DistilBERT Test Confusion Matrix  (Macro F1={test_f1:.3f})")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"confusion matrix saved to {cm_path}")

    #save test predictions
    test_df = test_df.copy()
    test_df["pred_label"] = [ID2LABEL[p] for p in test_preds]
    test_df["correct"]    = test_df["severity"] == test_df["pred_label"]
    pred_path = os.path.join(OUTPUT_DIR, "predictions_test.csv")
    test_df.to_csv(pred_path, index=False)
    print(f"predictions saved to {pred_path}")
if __name__ == "__main__":
    main()