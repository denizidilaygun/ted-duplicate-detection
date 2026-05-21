import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def create_synthetic_label(row):
    # Heuristic for synthetic labels (used before gold standard is available)
    if row['cae_levenshtein'] > 0.8 and row['win_levenshtein'] > 0.8 and row['value_ratio'] > 0.8:
        return 1
    return 0


df = pd.read_csv('candidate_pairs_with_features.csv')
df['synthetic_label'] = df.apply(create_synthetic_label, axis=1)

# Fall back to a softer threshold if too few positives at 0.8
if (df['synthetic_label'] == 1).sum() < 10:
    df['synthetic_label'] = (
        (df['cae_levenshtein'] > 0.7) &
        (df['win_levenshtein'] > 0.7) &
        (df['value_ratio'] > 0.7)
    ).astype(int)

# cpv_match is constant (always 1 due to blocking), so it is dropped
feature_cols = [
    'cae_levenshtein', 'cae_jaccard', 'cae_jaro_winkler',
    'win_levenshtein', 'win_jaccard', 'win_jaro_winkler',
    'country_match', 'value_diff', 'value_ratio'
]

X = df[feature_cols].values
y = df['synthetic_label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression baseline
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr) if len(np.unique(y_test)) > 1 else 0.0
print(f"Logistic Regression: F1={f1_lr:.3f}, AUC={auc_lr:.3f}")
print(classification_report(y_test, y_pred_lr, target_names=['Non-duplicate', 'Duplicate']))

# XGBoost with class-weighting for imbalance
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
f1_xgb = f1_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb) if len(np.unique(y_test)) > 1 else 0.0
print(f"XGBoost: F1={f1_xgb:.3f}, AUC={auc_xgb:.3f}, scale_pos_weight={scale_pos_weight:.2f}")
print(classification_report(y_test, y_pred_xgb))

# Feature importance from XGBoost
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop features (XGBoost):")
for _, row in feature_importance_df.head(5).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)
print(f"\nConfusion Matrix (XGBoost):\n{cm}")
