import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import xgboost as xgb
from Levenshtein import ratio as levenshtein_ratio
import jellyfish

RQ1_TRAIN_FILE = "synthetic_gold_standard_v3.csv"
REAL_DATA_FILE = "ted_for_genai.csv"
CTGAN_FILE     = "synthetic_ctgan.csv"
TVAE_FILE      = "synthetic_tvae.csv"

OUTPUT_CTGAN_SCORED = "ctgan_audit_results.csv"
OUTPUT_TVAE_SCORED  = "tvae_audit_results.csv"
OUTPUT_SUMMARY      = "audit_summary.csv"

FEATURES = [
    "cae_levenshtein", "cae_jaccard", "cae_jaro_winkler",
    "win_levenshtein", "win_jaccard", "win_jaro_winkler",
    "value_ratio", "cpv_match", "country_match",
]

NEAR_DUPLICATE_THRESHOLD = 0.5


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
    if not text_a or not text_b:
        return 0.0
    return jellyfish.jaro_winkler_similarity(
        str(text_a).lower(), str(text_b).lower()
    )


def compute_features_for_pair(synthetic_row, real_row):
    features = {}

    cae_syn  = str(synthetic_row.get("CAE_NAME", ""))
    cae_real = str(real_row.get("CAE_NAME", ""))
    features["cae_levenshtein"]  = compute_levenshtein(cae_syn, cae_real)
    features["cae_jaccard"]      = compute_jaccard(cae_syn, cae_real)
    features["cae_jaro_winkler"] = compute_jaro_winkler(cae_syn, cae_real)

    win_syn  = str(synthetic_row.get("WIN_NAME", ""))
    win_real = str(real_row.get("WIN_NAME", ""))
    features["win_levenshtein"]  = compute_levenshtein(win_syn, win_real)
    features["win_jaccard"]      = compute_jaccard(win_syn, win_real)
    features["win_jaro_winkler"] = compute_jaro_winkler(win_syn, win_real)

    val_syn = synthetic_row.get("VALUE_EURO", np.nan)
    val_real = real_row.get("VALUE_EURO", np.nan)
    try:
        v1, v2 = float(val_syn), float(val_real)
        if v1 > 0 and v2 > 0:
            features["value_ratio"] = min(v1, v2) / max(v1, v2)
        else:
            features["value_ratio"] = np.nan
    except (TypeError, ValueError):
        features["value_ratio"] = np.nan

    features["cpv_match"] = int(
        str(synthetic_row.get("CPV", "")) == str(real_row.get("CPV", ""))
    )
    features["country_match"] = int(
        str(synthetic_row.get("ISO_COUNTRY_CODE", "")) ==
        str(real_row.get("ISO_COUNTRY_CODE", ""))
    )

    return features


def find_nearest_real(synthetic_row, real_df_blocked):
    # For one synthetic record, find the most similar real record
    # in the blocked subset (same CPV and country).
    # The "nearest" record is the one with the highest combined similarity.

    if len(real_df_blocked) == 0:
        return None  

    win_syn = str(synthetic_row.get("WIN_NAME", ""))
    cae_syn = str(synthetic_row.get("CAE_NAME", ""))

    best_idx = None
    best_score = -1.0

    for idx, real_row in real_df_blocked.iterrows():
        win_real = str(real_row.get("WIN_NAME", ""))
        cae_real = str(real_row.get("CAE_NAME", ""))

        # Use Levenshtein as fast similarity measure
        sim_win = compute_levenshtein(win_syn, win_real)
        sim_cae = compute_levenshtein(cae_syn, cae_real)
        combined = max(sim_win, sim_cae)

        if combined > best_score:
            best_score = combined
            best_idx = idx

    return best_idx


def audit_model(synthetic_df, real_df, model_classifier, model_name):
    print(f"Auditing {model_name}")
    print(f"  Synthetic records : {len(synthetic_df):,}")
    print(f"  Real records      : {len(real_df):,}")

    # Build blocking index: group real records by (CPV, country) 
    print(f"\n  Building blocking index...")
    real_blocks = {}
    for idx, row in real_df.iterrows():
        key = (str(row["CPV"]), str(row["ISO_COUNTRY_CODE"]))
        if key not in real_blocks:
            real_blocks[key] = []
        real_blocks[key].append(idx)
    print(f"  Number of (CPV, country) blocks: {len(real_blocks):,}")

    # Audit each synthetic record
    print(f"\n  Auditing synthetic records...")
    results = []
    n_no_block = 0
    progress_step = max(1, len(synthetic_df) // 20)

    for i, (_, syn_row) in enumerate(synthetic_df.iterrows()):
        if i % progress_step == 0:
            print(f"    {i:,} / {len(synthetic_df):,} "
                  f"({i / len(synthetic_df) * 100:.0f}%)")

        # Look up real records in the same block
        key = (str(syn_row["CPV"]), str(syn_row["ISO_COUNTRY_CODE"]))
        if key not in real_blocks:
            n_no_block += 1
            results.append({
                "synthetic_index"     : i,
                "matched_real_index"  : None,
                "p_duplicate"         : 0.0,
                "near_duplicate"      : 0,
                "exact_match"         : 0,
                "blocking_match"      : 0,
            })
            continue

        real_block_df = real_df.loc[real_blocks[key]]

        # Find the nearest real record by string similarity
        nearest_idx = find_nearest_real(syn_row, real_block_df)
        if nearest_idx is None:
            n_no_block += 1
            continue

        nearest_real = real_df.loc[nearest_idx]

        # Compute the 9 features for the (synthetic, real) pair
        features = compute_features_for_pair(syn_row, nearest_real)

        # Fill any missing values
        feature_values = []
        for feat in FEATURES:
            val = features.get(feat, 0.0)
            if pd.isna(val):
                val = 0.0
            feature_values.append(val)

        # Apply the RQ1 classifier
        X = np.array([feature_values])
        proba = model_classifier.predict_proba(X)[0, 1]
        near_dup = int(proba >= NEAR_DUPLICATE_THRESHOLD)

        # Check exact match (all 5 columns identical)
        exact = int(
            str(syn_row["CAE_NAME"])         == str(nearest_real["CAE_NAME"])
            and str(syn_row["WIN_NAME"])     == str(nearest_real["WIN_NAME"])
            and float(syn_row["VALUE_EURO"]) == float(nearest_real["VALUE_EURO"])
            and str(syn_row["CPV"])          == str(nearest_real["CPV"])
            and str(syn_row["ISO_COUNTRY_CODE"]) == str(nearest_real["ISO_COUNTRY_CODE"])
        )

        results.append({
            "synthetic_index"    : i,
            "matched_real_index" : nearest_idx,
            "p_duplicate"        : proba,
            "near_duplicate"     : near_dup,
            "exact_match"        : exact,
            "blocking_match"     : 1,
            **features,
        })

    results_df = pd.DataFrame(results)
    print(f"    {len(synthetic_df):,} / {len(synthetic_df):,} (100%)")
    print(f"  Records with no block match: {n_no_block:,}")

    return results_df


def main():
    print("Memorization Audit (RQ2)")

    real_df = pd.read_csv(REAL_DATA_FILE)
    print(f"  Records: {len(real_df):,}")

    ctgan_df = pd.read_csv(CTGAN_FILE)
    tvae_df  = pd.read_csv(TVAE_FILE)
    print(f"  CTGAN synthetic: {len(ctgan_df):,}")
    print(f"  TVAE synthetic : {len(tvae_df):,}")

    train_df = pd.read_csv(RQ1_TRAIN_FILE)
    train_df = train_df[train_df["label"].isin([0, 1])].copy()
    for feat in FEATURES:
        if feat not in train_df.columns:
            train_df[feat] = 0
        train_df[feat] = train_df[feat].fillna(train_df[feat].median())

    X_train = train_df[FEATURES].values
    y_train = train_df["label"].values

    classifier = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="logloss", verbosity=0,
        use_label_encoder=False,
    )
    classifier.fit(X_train, y_train)
    print(f"  Classifier trained on {len(train_df):,} pairs.")

    ctgan_results = audit_model(ctgan_df, real_df, classifier, "CTGAN")
    ctgan_results.to_csv(OUTPUT_CTGAN_SCORED, index=False)
    print(f"\n  Saved: {OUTPUT_CTGAN_SCORED}")

    tvae_results = audit_model(tvae_df, real_df, classifier, "TVAE")
    tvae_results.to_csv(OUTPUT_TVAE_SCORED, index=False)
    print(f"\n  Saved: {OUTPUT_TVAE_SCORED}")


    summary_rows = []
    for model_name, results in [("CTGAN", ctgan_results), ("TVAE", tvae_results)]:
        n_total       = len(results)
        n_exact       = int(results["exact_match"].sum())
        n_near        = int(results["near_duplicate"].sum())
        gap           = n_near - n_exact

        exact_rate    = n_exact / n_total * 100
        near_rate     = n_near / n_total * 100
        gap_rate      = gap / n_total * 100

        print(f"\n  {model_name}:")
        print(f"    Total synthetic records       : {n_total:,}")
        print(f"    Exact-match memorization      : {n_exact:,} ({exact_rate:.2f}%)")
        print(f"    Near-duplicate memorization   : {n_near:,} ({near_rate:.2f}%)")
        print(f"    Additional caught by RQ1      : {gap:,} ({gap_rate:.2f}%)")

        summary_rows.append({
            "model"            : model_name,
            "total_records"    : n_total,
            "exact_match"      : n_exact,
            "near_duplicate"   : n_near,
            "gap"              : gap,
            "exact_rate_pct"   : round(exact_rate, 2),
            "near_rate_pct"    : round(near_rate, 2),
            "gap_rate_pct"     : round(gap_rate, 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\n  Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
