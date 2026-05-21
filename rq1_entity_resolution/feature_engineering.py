import pandas as pd
from Levenshtein import distance as levenshtein_distance
from Levenshtein import jaro_winkler
import warnings
warnings.filterwarnings('ignore')


def normalized_levenshtein(s1, s2):
    # Returns 0-1, where 1 = identical
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    s1, s2 = str(s1), str(s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1 - (levenshtein_distance(s1, s2) / max_len)


def jaccard_similarity(s1, s2):
    # Word-level Jaccard
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    set1 = set(str(s1).lower().split())
    set2 = set(str(s2).lower().split())
    if not set1 and not set2:
        return 1.0
    union = len(set1.union(set2))
    return len(set1.intersection(set2)) / union if union > 0 else 0.0


def jaro_winkler_sim(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    return jaro_winkler(str(s1), str(s2))


df = pd.read_csv('ted_clean_100k.csv')
pairs_df = pd.read_csv('candidate_pairs_for_annotation.csv')
print(f"Loaded {len(pairs_df):,} candidate pairs")

features_list = []
for idx, row in pairs_df.iterrows():
    if idx % 100 == 0 and idx > 0:
        print(f"  Progress: {idx}/{len(pairs_df)}")

    cae1, cae2 = row['cae_name_1'], row['cae_name_2']
    win1, win2 = row['win_name_1'], row['win_name_2']
    val1, val2 = row['value_1'], row['value_2']
    country1, country2 = row['country_1'], row['country_2']

    # Value difference (normalized) and ratio
    if pd.notna(val1) and pd.notna(val2) and max(val1, val2) > 0:
        value_diff = abs(val1 - val2) / max(val1, val2)
        value_ratio = min(val1, val2) / max(val1, val2) if val1 > 0 and val2 > 0 else 0.0
    else:
        value_diff = 1.0
        value_ratio = 0.0

    features_list.append({
        'pair_id': idx,
        'cae_levenshtein': normalized_levenshtein(cae1, cae2),
        'cae_jaccard': jaccard_similarity(cae1, cae2),
        'cae_jaro_winkler': jaro_winkler_sim(cae1, cae2),
        'win_levenshtein': normalized_levenshtein(win1, win2),
        'win_jaccard': jaccard_similarity(win1, win2),
        'win_jaro_winkler': jaro_winkler_sim(win1, win2),
        'cpv_match': 1,                                              # always 1 due to blocking
        'country_match': 1 if country1 == country2 else 0,
        'value_diff': value_diff,
        'value_ratio': value_ratio,
        'cpv': row['cpv'],
        'country_1': country1,
        'country_2': country2,
    })

features_df = pd.DataFrame(features_list)

# Check for multicollinearity (>0.9 correlation between features)
feature_cols = [c for c in features_df.columns
                if c not in ['pair_id', 'cpv', 'country_1', 'country_2']]
corr_matrix = features_df[feature_cols].corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append(
                (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            )

if high_corr_pairs:
    print("High correlation pairs (>0.9):")
    for f1, f2, corr in high_corr_pairs:
        print(f"  {f1} <-> {f2}: {corr:.3f}")

# Merge features back onto the pair information and save
final_df = pairs_df.copy()
for col in feature_cols:
    final_df[col] = features_df[col]

final_df.to_csv('candidate_pairs_with_features.csv', index=False, encoding='utf-8')
