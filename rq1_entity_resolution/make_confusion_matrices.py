import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


FEATURES = [
    "cae_levenshtein", "cae_jaccard", "cae_jaro_winkler",
    "win_levenshtein", "win_jaccard", "win_jaro_winkler",
    "value_ratio", "cpv_match", "country_match",
]


def plot_confusion_matrix(cm, title, filename, class_names=("Non-duplicate", "Duplicate")):
    # cm: numpy array of shape (2, 2) from sklearn.metrics.confusion_matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=12, pad=15)

    # Add count + percent annotation inside each cell
    total = cm.sum()
    threshold = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100.0 * count / total if total > 0 else 0.0
            text_color = "white" if count > threshold else "black"
            ax.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color,
                fontsize=12, weight="bold",
            )

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def run_v3_synthetic_evaluation():
    print("\n[1/2] In-distribution evaluation on v3 synthetic gold standard")

    df = pd.read_csv("synthetic_gold_standard.csv")
    print(f"  Loaded {len(df)} pairs")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")

    X = df[FEATURES].values
    y = df["label"].values

    # XGBoost with standard configuration (matches real_model_training.py)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    # 5-fold stratified cross-validation predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=skf)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"            Pred 0    Pred 1")
    print(f"  True 0    {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"  True 1    {cm[1,0]:6d}    {cm[1,1]:6d}")

    plot_confusion_matrix(
        cm,
        title="Confusion Matrix: XGBoost on v3 Synthetic Gold Standard\n(5-fold cross-validation)",
        filename="confusion_matrix_v3_synthetic.png",
    )

    # Compute and return metrics
    metrics = {
        "evaluation": "v3 synthetic (5-fold CV)",
        "n_samples": len(y),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }
    return model, X, y, metrics


def run_ood_real_evaluation(X_train, y_train):
    print("\n[2/2] Out-of-distribution evaluation on 497 real pairs")

    df_features = pd.read_csv("candidate_pairs_with_features.csv")
    df_labels = pd.read_csv("annotated_gold_standard.csv")

    # Merge the manually annotated labels onto the feature rows
    merge_keys = ["cae_name_1", "cae_name_2", "win_name_1", "win_name_2",
                  "country_1", "country_2", "cpv"]
    df_real = df_features.drop(columns=["label"]).merge(
        df_labels[merge_keys + ["label"]],
        on=merge_keys,
        how="inner",
    )
    print(f"  Merged: {len(df_real)} pairs")
    print(f"  Class distribution: {df_real['label'].value_counts().to_dict()}")

    # Train on the full v3 synthetic data
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict on real pairs
    X_real = df_real[FEATURES].values
    y_real = df_real["label"].astype(int).values
    y_pred_real = model.predict(X_real)

    # Confusion matrix
    cm = confusion_matrix(y_real, y_pred_real)
    print(f"\n  Confusion matrix:")
    print(f"            Pred 0    Pred 1")
    print(f"  True 0    {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"  True 1    {cm[1,0]:6d}    {cm[1,1]:6d}")

    plot_confusion_matrix(
        cm,
        title="Confusion Matrix: XGBoost on 497 Real Annotated Pairs\n(Out-of-Distribution Evaluation)",
        filename="confusion_matrix_ood_real.png",
    )

    # Compute metrics
    try:
        precision = precision_score(y_real, y_pred_real, zero_division=0)
        recall = recall_score(y_real, y_pred_real, zero_division=0)
        f1 = f1_score(y_real, y_pred_real, zero_division=0)
    except Exception:
        precision = recall = f1 = 0.0

    metrics = {
        "evaluation": "OOD real pairs",
        "n_samples": len(y_real),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }
    return metrics


def main():
    model, X_train, y_train, metrics_v3 = run_v3_synthetic_evaluation()
    metrics_ood = run_ood_real_evaluation(X_train, y_train)

    df_metrics = pd.DataFrame([metrics_v3, metrics_ood])
    df_metrics.to_csv("confusion_matrix_metrics.csv", index=False)
    print("\n  Saved: confusion_matrix_metrics.csv")

    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    main()
