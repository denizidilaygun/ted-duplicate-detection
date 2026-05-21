import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from Levenshtein import ratio as levenshtein_ratio

TRAIN_FILE  = "synthetic_gold_standard_v3.csv"   # synthetic training data
EVAL_FILE   = "annotated_gold_standard.csv"       # real pairs for evaluation
OUTPUT_FILE = "real_pairs_scored.csv"             # output with predicted probabilities

FEATURES = [
    "cae_levenshtein", "cae_jaccard", "cae_jaro_winkler",
    "win_levenshtein", "win_jaccard", "win_jaro_winkler",
    "value_ratio", "cpv_match", "country_match",
]


# The real annotated pairs only have raw text columns.
# We need to compute the 9 features from scratch for each pair.

def compute_levenshtein(text_a, text_b):
    if not text_a or not text_b:
        return 0.0
    return levenshtein_ratio(str(text_a).lower(), str(text_b).lower())


def compute_jaccard(text_a, text_b):
    if not text_a or not text_b:
        return 0.0
    words_a = set(str(text_a).lower().split())
    words_b = set(str(text_b).lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def compute_jaro_winkler(text_a, text_b):
    try:
        import jellyfish
        return jellyfish.jaro_winkler_similarity(
            str(text_a).lower(), str(text_b).lower()
        )
    except ImportError:
        return compute_levenshtein(text_a, text_b)


def compute_features_for_real_pair(row):
    # Compute all 9 features for one real annotated pair.
    #   cae_name_1, cae_name_2, win_name_1, win_name_2,
    #   value_1, value_2, country_1, country_2, cpv, label
    features = {}

    cae_1 = str(row.get("cae_name_1", ""))
    cae_2 = str(row.get("cae_name_2", ""))
    features["cae_levenshtein"]  = compute_levenshtein(cae_1, cae_2)
    features["cae_jaccard"]      = compute_jaccard(cae_1, cae_2)
    features["cae_jaro_winkler"] = compute_jaro_winkler(cae_1, cae_2)

    win_1 = str(row.get("win_name_1", ""))
    win_2 = str(row.get("win_name_2", ""))
    features["win_levenshtein"]  = compute_levenshtein(win_1, win_2)
    features["win_jaccard"]      = compute_jaccard(win_1, win_2)
    features["win_jaro_winkler"] = compute_jaro_winkler(win_1, win_2)

    # value_ratio: min/max of contract values
    try:
        v1 = float(row.get("value_1", 0))
        v2 = float(row.get("value_2", 0))
        if v1 > 0 and v2 > 0:
            features["value_ratio"] = min(v1, v2) / max(v1, v2)
        else:
            features["value_ratio"] = np.nan
    except (TypeError, ValueError):
        features["value_ratio"] = np.nan

    # All annotated pairs were blocked on the same CPV, so cpv_match = 1
    features["cpv_match"] = 1

    c1 = str(row.get("country_1", ""))
    c2 = str(row.get("country_2", ""))
    features["country_match"] = int(c1 == c2 and c1 != "")

    return features


def main():
    print("Out-of-Distribution Evaluation")
    print(f"Train on: {TRAIN_FILE}")
    print(f"Evaluate: {EVAL_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    train_df = train_df[train_df["label"].isin([0, 1])].copy()

    # Fill missing values with the median of each feature column
    for feature in FEATURES:
        if feature not in train_df.columns:
            train_df[feature] = 0
        train_df[feature] = train_df[feature].fillna(train_df[feature].median())

    X_train = train_df[FEATURES].values   # the 9 feature columns
    y_train  = train_df["label"].values   # the labels (0 or 1)

    print(f"  Training pairs  : {len(train_df):,}")
    print(f"  Duplicates (1)  : {(y_train == 1).sum()}")
    print(f"  Non-dup    (0)  : {(y_train == 0).sum()}")

    real_df = pd.read_csv(EVAL_FILE)
    real_df = real_df[real_df["label"].isin([0, 1])].copy()
    real_df = real_df.reset_index(drop=True)

    n_real_dup = (real_df["label"] == 1).sum()
    n_real_non = (real_df["label"] == 0).sum()
    print(f"  Real pairs      : {len(real_df):,}")
    print(f"  Real duplicates : {n_real_dup}")
    print(f"  Real non-dup    : {n_real_non}")

    # Compute features row by row for each real pair
    feature_rows = []
    for _, row in real_df.iterrows():
        feature_rows.append(compute_features_for_real_pair(row))

    features_df = pd.DataFrame(feature_rows)

    # Fill missing values
    for feature in FEATURES:
        if feature not in features_df.columns:
            features_df[feature] = 0
        features_df[feature] = features_df[feature].fillna(
            features_df[feature].median()
        )

    X_eval = features_df[FEATURES].values
    y_eval  = real_df["label"].values

    # Print the feature values for the real duplicate so we can inspect them
    dup_indices = real_df[real_df["label"] == 1].index.tolist()
    if dup_indices:
        print(f"\n  Features for the real duplicate:")
        for feature in FEATURES:
            value = features_df.loc[dup_indices[0], feature]
            print(f"    {feature:<22}: {value:.4f}")

    model = xgb.XGBClassifier(
        n_estimators=200,    # number of decision trees
        max_depth=6,         # how deep each tree can grow
        learning_rate=0.1,   # how much each tree corrects the previous one
        random_state=42,     # for reproducibility
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    print(f"  Model trained.")


    # predict_proba returns two columns: P(non-duplicate), P(duplicate)
    # We take the second column [:, 1] which is P(duplicate)
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions   = (probabilities >= 0.5).astype(int)

    # Add results back to the real dataframe
    real_df = real_df.copy()
    real_df["p_duplicate"] = probabilities
    real_df["predicted"]   = predictions
    for feature in FEATURES:
        real_df[feature] = features_df[feature].values

    # Sort by predicted probability, highest first
    real_sorted = real_df.sort_values("p_duplicate", ascending=False)
    real_sorted = real_sorted.reset_index(drop=True)
    real_sorted.to_csv(OUTPUT_FILE, index=False)

    print("Results")

    # Show what happened to the real duplicate
    real_dups = real_df[real_df["label"] == 1]
    if len(real_dups) > 0:
        print(f"\n  --- The real duplicate ---")
        for _, row in real_dups.iterrows():
            print(f"  CAE_1 : {row.get('cae_name_1', '')}")
            print(f"  CAE_2 : {row.get('cae_name_2', '')}")
            print(f"  WIN_1 : {row.get('win_name_1', '')}")
            print(f"  WIN_2 : {row.get('win_name_2', '')}")
            print(f"\n  Feature values:")
            for feature in FEATURES:
                print(f"    {feature:<22}: {row[feature]:.4f}")
            print(f"\n  P(duplicate)  : {row['p_duplicate']:.4f}")
            if row["predicted"] == 1:
                print(f"  Predicted     : DUPLICATE (correct)")
            else:
                print(f"  Predicted     : non-duplicate (incorrect)")

    # Top 20 most suspicious pairs
    print(f"\n  --- Top 20 pairs by P(duplicate) ---")
    header = f"  {'Rank':<5} {'P(dup)':<8} {'True label':<12} {'WIN_1':<28} {'WIN_2':<28}"
    print(header)
    for rank, (_, row) in enumerate(real_sorted.head(20).iterrows(), 1):
        true_label = "DUPLICATE" if row["label"] == 1 else "non-dup"
        win1 = str(row.get("win_name_1", ""))[:26]
        win2 = str(row.get("win_name_2", ""))[:26]
        print(f"  {rank:<5} {row['p_duplicate']:<8.4f} {true_label:<12} "
              f"{win1:<28} {win2:<28}")

    # Summary of scores
    print(f"\n  --- Score distribution across all {len(real_df)} real pairs ---")
    print(f"  P(dup) > 0.90 : {(probabilities > 0.90).sum()} pairs")
    print(f"  P(dup) > 0.50 : {(probabilities > 0.50).sum()} pairs")
    print(f"  P(dup) > 0.10 : {(probabilities > 0.10).sum()} pairs")
    print(f"  P(dup) < 0.01 : {(probabilities < 0.01).sum()} pairs")

    print(f"\n  Mean P(dup) for non-duplicates : "
          f"{probabilities[y_eval == 0].mean():.4f}")
    if n_real_dup > 0:
        print(f"  Mean P(dup) for real duplicate : "
              f"{probabilities[y_eval == 1].mean():.4f}")

    print(f"\n  Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
