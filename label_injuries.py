"""
severity based on typical NBA recovery time:
  MINOR = days to ~1 week
  MODERATE = ~1–6 weeks
  SEVERE = 6+ weeks / season-ending / surgical

two-step classification:
  Step 1: Scan the 'notes' field with keyword matching
  Step 2: Fallback to 'injury_type' column for unmatched records
  Omit: Records where injury_type == 'injury' with no useful detail

modifiers after base label:
  1. Return-timeline signals in notes override everything:
       DTD / day-to-day / DNP = cap at MINOR
       "out for season" / "out indefinitely" = floor at SEVERE
  2. Grade language (grade 1/2/3, partial, complete, mild) shifts ±1 step
  3. Body-part healing rate shifts borderline MODERATE ±1 step,
       but ONLY when no return-timeline signal is present
"""

import re
import pandas as pd

#return-timeline override patterns

MINOR_OVERRIDE_SIGNALS = [
    r"\bdtd\b", r"\bday[\s\-]to[\s\-]day\b",
    r"\bdnp\b", 
    r"\bcbc\b", 
]

SEVERE_OVERRIDE_SIGNALS = [
    r"\bout\s+for\s+(the\s+)?season\b",
    r"\bout\s+indefinitely\b",
    r"\bseason[\s\-]ending\b",
]

#base keyword lists for initial classification from notes
SEVERE_KEYWORDS = [
    "acl tear", "acl", "achilles",
    "out for season", "out indefinitely",
    "surgery", "surgical",
    "torn", "rupture", "ruptured",
    "dislocation", "dislocated",
    "season-ending",
]

MODERATE_KEYWORDS = [
    "stress fracture", "stress reaction",
    "plantar fasciitis", "plantar fascia",
    "partial tear", "partial thickness",
    "hyperextension", "hyperextended",
    "impingement",
    "subluxation",
    "shin splints",
    "turf toe",
    "bursitis",
    "tendinopathy", "tendinitis", "tendonitis",
    "inflammation", "inflamed",
    "concussion",
    "laceration", "lacerated",
    "fracture", "fractured",
    "sprain", "sprained",
    "strain", "strained",
    "broken",
    "bruise", "bruised",
    "contusion",
    "loose bodies", "loose particle",
    "swelling", "swollen",
    "infection",
    "nerve issue", "nerve",
    "disc injury", "disc herniation",
]

MINOR_KEYWORDS = [
    "day-to-day", "dtd",
    "soreness", "sore",
    "tightness", "tight",
    "stiffness", "stiff",
    "illness", "sick",
    "spasm",
    "fatigue",
    "rest",
    "load management",
]

#fallback injury_type lists for unmatched records
SEVERE_INJURY_TYPES = {
    "acl tear", "achilles", "rupture", "dislocation",
    "surgery", "tear", "torn ligament",
}
MODERATE_INJURY_TYPES = {
    "fracture", "sprain", "strain", "bruise", "contusion",
    "hyperextension", "impingement", "subluxation", "plantar fasciitis",
    "bursitis", "tendinopathy", "tendinitis", "tendonitis",
    "inflammation", "laceration", "swelling", "infection",
    "stress reaction", "shin splints", "turf toe", "concussion",
    "loose bodies", "disc injury", "nerve issue",
}
MINOR_INJURY_TYPES = {
    "soreness", "tightness", "illness", "stiffness", "spasm", "fatigue",
}

#grade modifier patterns
GRADE_UPGRADE_PATTERNS = [
    r"\bgrade\s*3\b",
    r"\bcomplete\s+tear\b", r"\bcomplete\s+rupture\b",
    r"\bfull\s+thickness\b", r"\btotal\s+rupture\b",
]

GRADE_DOWNGRADE_PATTERNS = [
    r"\bgrade\s*1\b",
    r"\bmild\b",
    r"\bminor\b",
    r"\bpartial\b",
    r"\blow[\s\-]grade\b",
]

#body part healing moodifiers 
SLOW_HEALING_PARTS = {
    "achilles", "knee", "acl", "pcl", "mcl", "lcl",
    "hip", "back", "spine", "hamstring", "quadricep", "quad",
}

FAST_HEALING_PARTS = {
    "finger", "thumb", "toe", "hand", "wrist",
    "nose", "lip", "ear",
}


#helper functions
def _matches_any_keyword(text: str, keywords: list) -> bool:
    return any(kw in text for kw in keywords)

def _matches_any_pattern(text: str, patterns: list) -> bool:
    return any(re.search(p, text) for p in patterns)

def _severity_score(label: str) -> int:
    return {"MINOR": 0, "MODERATE": 1, "SEVERE": 2}[label]

def _score_to_label(score: int) -> str:
    return ["MINOR", "MODERATE", "SEVERE"][max(0, min(2, score))]


#1. classify from notes 

def classify_from_notes(notes: str) -> str | None:
    """
    scan notes for severity keywords.
    severe checked first so 'torn strain' = SEVERE, not MODERATE.
    returns 'MINOR', 'MODERATE', 'SEVERE', or None.
    """
    if not isinstance(notes, str) or notes.strip() == "":
        return None
    text = notes.lower()

    if _matches_any_keyword(text, SEVERE_KEYWORDS):
        return "SEVERE"
    if _matches_any_keyword(text, MODERATE_KEYWORDS):
        return "MODERATE"
    if _matches_any_keyword(text, MINOR_KEYWORDS):
        return "MINOR"
    return None


 # 2. fallback classification from injury_type for unmatched records
def classify_from_injury_type(injury_type: str) -> str | None:
    """
    fallback for unmatched records.
    generic 'injury' entries return None and will be omitted.
    """
    if not isinstance(injury_type, str):
        return None
    val = injury_type.lower().strip()
    if val == "injury":
        return None
    if val in SEVERE_INJURY_TYPES:
        return "SEVERE"
    if val in MODERATE_INJURY_TYPES:
        return "MODERATE"
    if val in MINOR_INJURY_TYPES:
        return "MINOR"
    return None


#modifier pipeline to adjust base labels
def apply_modifiers(base_label: str, notes: str, body_part: str) -> str:
    """
    adjusts base_label in this order:
    1. Return-timeline override (highest priority):
         DTD / day-to-day / DNP / CBC = cap label at MINOR
         out for season / out indefinitely = floor label at SEVERE
       These anchor the label and skip the remaining modifiers.
    2. Grade language (grade 1/2/3, partial, complete):
         grade 3 / complete = +1 step
         grade 1 / mild / partial = -1 step
    3. Body-part healing rate (only when no timeline signal present,
       only affects MODERATE):
         slow-healing (knee, hamstring, back …) = MODERATE = SEVERE
         fast-healing (finger, thumb, toe …)    = MODERATE = MINOR
    """
    if not isinstance(notes, str):
        notes = ""
    if not isinstance(body_part, str):
        body_part = ""

    text  = notes.lower()
    bpart = body_part.lower().strip()

    # return timeline override signals trump all other modifiers
    has_minor_signal  = _matches_any_pattern(text, MINOR_OVERRIDE_SIGNALS)
    has_severe_signal = _matches_any_pattern(text, SEVERE_OVERRIDE_SIGNALS)

    if has_severe_signal:
        return "SEVERE"

    if has_minor_signal:
        if base_label != "SEVERE":
            return "MINOR"

    #2. grade language modifier
    score = _severity_score(base_label)

    if _matches_any_pattern(text, GRADE_UPGRADE_PATTERNS):
        score = min(score + 1, 2)
    elif _matches_any_pattern(text, GRADE_DOWNGRADE_PATTERNS):
        score = max(score - 1, 0)

    #3. body-part healing rate
    if score == 1 and not has_minor_signal:
        if bpart in SLOW_HEALING_PARTS:
            score = 2   # MODERATE = SEVERE
        elif bpart in FAST_HEALING_PARTS:
            score = 0   # MODERATE = MINOR

    return _score_to_label(score)


# full row labeler
def label_row(row) -> str | None:
    base = classify_from_notes(row["notes"])
    if base is None:
        base = classify_from_injury_type(row["injury_type"])
    if base is None:
        return None
    return apply_modifiers(base, row["notes"], row["body_part"])


# main pipeline function to run the labeling and save results
def run_pipeline(input_path: str,
                 labeled_out: str,
                 omitted_out: str) -> None:

    print(f"loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  total records loaded:        {len(df)}")

    df["severity"] = df.apply(label_row, axis=1)

    labeled_df = df[df["severity"].notna()].copy()
    omitted_df = df[df["severity"].isna()].copy()

    print(f"  records labeled:             {len(labeled_df)}")
    print(f"  records omitted (no label):  {len(omitted_df)}")

    print("\nClass distribution:")
    counts = labeled_df["severity"].value_counts()
    for label in ["MINOR", "MODERATE", "SEVERE"]:
        n   = counts.get(label, 0)
        pct = n / len(labeled_df) * 100
        print(f"  {label:<10} {n:>5}  ({pct:.1f}%)")

    labeled_df.to_csv(labeled_out, index=False)
    omitted_df.to_csv(omitted_out, index=False)
    print(f"\nLabeled dataset saved to: {labeled_out}")
    print(f"Omitted records saved to: {omitted_out}")

    INPUT_PATH  = "nba_injuries_finalized.csv"
    LABELED_OUT = "nba_injuries_labeled.csv"
    OMITTED_OUT = "nba_injuries_omitted.csv"
    run_pipeline(INPUT_PATH, LABELED_OUT, OMITTED_OUT)