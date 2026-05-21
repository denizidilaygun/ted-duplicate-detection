import argparse
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, f1_score, roc_auc_score,
                              precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STRING_FEATURES = [
    "cae_levenshtein", "cae_jaccard", "cae_jaro_winkler",
    "win_levenshtein", "win_jaccard", "win_jaro_winkler",
]
STRUCTURED_FEATURES = ["value_ratio", "cpv_match", "country_match"]
SBERT_FEATURES = ["sbert_similarity"]
BASE_FEATURES = STRING_FEATURES + STRUCTURED_FEATURES


def load_data(gold_path, sbert_path=None):
    gold = pd.read_csv(gold_path)

    if "label" not in gold.columns:
        raise ValueError("'label' column not found. Run annotation_tool.py first.")

    # Drop skipped pairs (label == -1)
    gold = gold[gold["label"].isin([0, 1])].copy()
    n_dup = (gold["label"] == 1).sum()
    n_non = (gold["label"] == 0).sum()
    print(f"Loaded: {len(gold)} labeled pairs ({n_dup} dup, {n_non} non-dup)")

    if n_dup == 0:
        raise ValueError("No duplicates in gold standard.")

    scale_pos_weight = n_non / n_dup

    # Optionally merge SBERT similarities
    features = BASE_FEATURES.copy()
    if sbert_path:
        sbert_df = pd.read_csv(sbert_path)
        if "sbert_similarity" in sbert_df.columns:
            if "pair_id" in gold.columns and "pair_id" in sbert_df.columns:
                gold = gold.merge(
                    sbert_df[["pair_id", "sbert_similarity"]],
                    on="pair_id", how="left"
                )
            else:
                gold["sbert_similarity"] = sbert_df["sbert_similarity"].values[:len(gold)]
            features += SBERT_FEATURES

    return gold, features, scale_pos_weight


def check_missing(df, features):
    # Fill any missing feature values with the column median
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
        elif df[feat].isna().sum() > 0:
            df[feat] = df[feat].fillna(df[feat].median())
    return df


def train_and_evaluate(X, y, model, model_name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=min(5, (y == 1).sum() - 1))

    if isinstance(model, LogisticRegression):
        pipeline = ImbPipeline([("smote", smote), ("scaler", StandardScaler()), ("model", model)])
    else:
        pipeline = ImbPipeline([("smote", smote), ("model", model)])

    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)

    print(f"\n{model_name}: F1={f1:.4f}, AUC={auc:.4f}, P={prec:.4f}, R={rec:.4f}")
    print(classification_report(y, y_pred, target_names=['Non-dup', 'Duplicate'], zero_division=0))

    return {"model": model_name, "F1": f1, "AUC-ROC": auc,
            "Precision": prec, "Recall": rec, "y_pred": y_pred, "y_prob": y_prob}


def plot_comparison(results, output_path="model_comparison.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["F1", "AUC-ROC", "Precision", "Recall"]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    for i, res in enumerate(results):
        values = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=res["model"], color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (5-fold CV)")
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def plot_shap(model, X, features, output_path="shap_plot.png"):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_shap = np.abs(shap_values).mean(axis=0)
        feat_imp = pd.Series(mean_shap, index=features).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2E86AB" if f in STRING_FEATURES else
                  "#F18F01" if f in STRUCTURED_FEATURES else "#A23B72"
                  for f in feat_imp.index]
        feat_imp.plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance (SHAP)")
        patches = [
            mpatches.Patch(color="#2E86AB", label="String similarity"),
            mpatches.Patch(color="#F18F01", label="Structured"),
            mpatches.Patch(color="#A23B72", label="SBERT semantic"),
        ]
        ax.legend(handles=patches, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

        print("Feature importance (SHAP):")
        for feat, val in feat_imp.sort_values(ascending=False).items():
            print(f"  {feat:<30}: {val:.4f}")
    except ImportError:
        print("SHAP not installed (pip install shap)")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ER classifiers")
    parser.add_argument("--input", default="annotated_gold_standard.csv")
    parser.add_argument("--sbert", default=None, help="Optional path to pairs_with_sbert.csv")
    args = parser.parse_args()

    df, features, scale_pos_weight = load_data(args.input, args.sbert)
    df = check_missing(df, features)

    X = df[features].values
    y = df["label"].values

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, scale_pos_weight=scale_pos_weight,
        max_depth=6, learning_rate=0.1, random_state=42,
        eval_metric="logloss", use_label_encoder=False, verbosity=0
    )

    results = []
    results.append(train_and_evaluate(X, y, lr, "Logistic Regression"))
    results.append(train_and_evaluate(X, y, rf, "Random Forest"))
    results.append(train_and_evaluate(X, y, xgb_model, "XGBoost"))

    # Train XGBoost on the full SMOTE-balanced set for SHAP interpretation
    smote = SMOTE(random_state=42, k_neighbors=min(5, (y == 1).sum() - 1))
    X_res, y_res = smote.fit_resample(X, y)
    best_xgb = xgb.XGBClassifier(
        n_estimators=200, scale_pos_weight=1,  # already balanced after SMOTE
        max_depth=6, learning_rate=0.1, random_state=42,
        eval_metric="logloss", use_label_encoder=False, verbosity=0
    )
    best_xgb.fit(X_res, y_res)

    plot_shap(best_xgb, X, features)

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["y_pred", "y_prob"]}
        for r in results
    ])
    results_df.to_csv("model_results.csv", index=False)
    plot_comparison(results)

    print("\nFinal results:")
    print(results_df.to_string(index=False))
    best = results_df.loc[results_df["F1"].idxmax()]
    print(f"Best model: {best['model']} (F1={best['F1']:.4f})")


if __name__ == "__main__":
    main()
