import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


FEATURES = [
    "cae_levenshtein", "cae_jaccard", "cae_jaro_winkler",
    "win_levenshtein", "win_jaccard", "win_jaro_winkler",
    "value_ratio", "cpv_match", "country_match",
]
MIN_PAIRS_PER_COUNTRY = 30   # need enough samples for stable rate estimation


def load_data():
    df = pd.read_csv("synthetic_duplicate_pairs.csv")
    print(f"Loaded {len(df)} pairs from synthetic gold standard")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    print(f"Number of unique countries: {df['COUNTRY_1'].nunique()}")
    return df


def get_cv_predictions(df):
    print("\nRunning 5-fold stratified cross-validation...")
    X = df[FEATURES].values
    y = df["label"].values

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=skf)

    df = df.copy()
    df["prediction"] = y_pred
    return df


def compute_per_country_rates(df):
    print(f"\nComputing per-country error rates "
          f"(min {MIN_PAIRS_PER_COUNTRY} pairs per country)...")

    rows = []
    for country in sorted(df["COUNTRY_1"].unique()):
        df_c = df[df["COUNTRY_1"] == country]
        n_total = len(df_c)
        if n_total < MIN_PAIRS_PER_COUNTRY:
            continue

        y_true = df_c["label"].values
        y_pred = df_c["prediction"].values

        # Confusion matrix (returns [[TN, FP], [FN, TP]])
        # Force 2x2 shape in case a country has only one class
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        # Rates, safely handling zero denominators
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan

        rows.append({
            "country": country,
            "n_pairs": n_total,
            "n_duplicates": int(y_true.sum()),
            "n_non_duplicates": int((y_true == 0).sum()),
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
            "FPR": round(fpr, 4) if not np.isnan(fpr) else np.nan,
            "FNR": round(fnr, 4) if not np.isnan(fnr) else np.nan,
        })

    result = pd.DataFrame(rows).sort_values("n_pairs", ascending=False)
    return result


def plot_results(df_rates, filename):
    # Order countries by sample size, largest first
    df_plot = df_rates.sort_values("n_pairs", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_plot))
    width = 0.38

    bars_fpr = ax.bar(x - width/2, df_plot["FPR"] * 100, width,
                      label="False Positive Rate (FPR)",
                      color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars_fnr = ax.bar(x + width/2, df_plot["FNR"] * 100, width,
                      label="False Negative Rate (FNR)",
                      color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Country (ordered by sample size)", fontsize=11)
    ax.set_ylabel("Error rate (%)", fontsize=11)
    ax.set_title("Disparate Impact Analysis: Per-Country Error Rates\n"
                 "(XGBoost on v3 Synthetic Gold Standard, 5-fold CV)",
                 fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["country"], rotation=0)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add sample size annotation below each country code
    for i, n in enumerate(df_plot["n_pairs"]):
        ax.text(i, -ax.get_ylim()[1]*0.04, f"n={n}",
                ha="center", va="top", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def main():
    df = load_data()
    df = get_cv_predictions(df)
    result = compute_per_country_rates(df)

    result.to_csv("disparate_impact_per_country.csv", index=False)
    print("\n  Saved: disparate_impact_per_country.csv")

    plot_results(result, "disparate_impact_per_country.png")

    print(result.to_string(index=False))

    print(f"\nFPR range: {result['FPR'].min()*100:.2f}% -- {result['FPR'].max()*100:.2f}%")
    print(f"FPR mean: {result['FPR'].mean()*100:.2f}%, std: {result['FPR'].std()*100:.2f}%")
    print(f"FNR range: {result['FNR'].min()*100:.2f}% -- {result['FNR'].max()*100:.2f}%")
    print(f"FNR mean: {result['FNR'].mean()*100:.2f}%, std: {result['FNR'].std()*100:.2f}%")


if __name__ == "__main__":
    main()
