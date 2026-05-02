"""
- produces a single stratified 70/15/15 train/dev/test split that is used by ALL models: Majority Class Baseline, NB, LR, and DistilBERT.
Output files (same directory as this script):
  split_train.csv
  split_dev.csv
  split_test.csv

each file has all original columns plus:
  input_text — notes + injury_type + body_part concatenated 
  label_id — integer encoding: MINOR=0, MODERATE=1, SEVERE=2
"""

import pandas as pd
from sklearn.model_selection import train_test_split
#config
INPUT_PATH  = "nba_injuries_labeled.csv"
RANDOM_SEED = 42
TRAIN_FRAC  = 0.70
DEV_FRAC    = 0.15

LABEL2ID = {"MINOR": 0, "MODERATE": 1, "SEVERE": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def build_input_text(df: pd.DataFrame) -> pd.Series:
    """
    Concatenate the three input fields into one string.
    This is what the TF-IDF and DistilBERT both receive.
    """
    return (
        df["notes"].fillna("") + " " +
        df["injury_type"].fillna("") + " " +
        df["body_part"].fillna("")
    ).str.strip()


def split_and_save(input_path: str) -> None:
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Total records: {len(df)}")

    # Build shared input text and integer label
    df["input_text"] = build_input_text(df)
    df["label_id"]   = df["severity"].map(LABEL2ID)

    #carve out train/dev/test splits with stratification
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - TRAIN_FRAC),
        stratify=df["severity"],
        random_state=RANDOM_SEED,
    )
    #split the temp_df into dev and test
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["severity"],
        random_state=RANDOM_SEED,
    )

    #report
    for name, split in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        print(f"\n{name.upper()} ({len(split)} records)")
        counts = split["severity"].value_counts()
        for label in ["MINOR", "MODERATE", "SEVERE"]:
            n   = counts.get(label, 0)
            pct = n / len(split) * 100
            print(f"  {label:<10} {n:>4}  ({pct:.1f}%)")

    #save
    train_df.to_csv("split_train.csv", index=False)
    dev_df.to_csv("split_dev.csv",     index=False)
    test_df.to_csv("split_test.csv",   index=False)

    print("\nSaved:")
    print("  split_train.csv")
    print("  split_dev.csv")
    print("  split_test.csv")
    print()
    print("Label encoding:")
    for label, idx in LABEL2ID.items():
        print(f"  {idx} = {label}")

if __name__ == "__main__":
    split_and_save(INPUT_PATH)