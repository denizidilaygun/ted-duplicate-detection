import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('ted_clean_100k.csv')
print(f"Loaded {len(df):,} records")

# Blocking on CPV + YEAR
if 'YEAR' not in df.columns:
    df['YEAR'] = pd.to_datetime(df['DT_DISPATCH'], errors='coerce').dt.year

df['BLOCK_KEY'] = df['CPV'].astype(str) + '_' + df['YEAR'].astype(str)
block_sizes = df.groupby('BLOCK_KEY').size()

# Estimate candidate pair counts before and after blocking
total_possible_pairs = sum(n * (n - 1) / 2 for n in block_sizes if n > 1)
naive_pairs = 100000 * 99999 / 2
reduction_pct = (1 - total_possible_pairs / naive_pairs) * 100
print(f"Blocks: {len(block_sizes):,}, candidate pairs: {total_possible_pairs:,.0f} "
      f"({reduction_pct:.1f}% reduction vs naive)")

# Sample 500 candidate pairs from the top 5 largest blocks for manual annotation
pairs_data = []
top5_blocks = block_sizes.nlargest(5)

for block_key in top5_blocks.index:
    block_df = df[df['BLOCK_KEY'] == block_key].reset_index(drop=True)
    n = len(block_df)

    for _ in range(min(100, n)):
        if n < 2:
            continue
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        r1, r2 = block_df.iloc[idx1], block_df.iloc[idx2]

        pairs_data.append({
            'cae_name_1': r1['CAE_NAME'],
            'cae_name_2': r2['CAE_NAME'],
            'win_name_1': r1['WIN_NAME'],
            'win_name_2': r2['WIN_NAME'],
            'value_1': r1['VALUE_EURO'],
            'value_2': r2['VALUE_EURO'],
            'country_1': r1['ISO_COUNTRY_CODE'],
            'country_2': r2['ISO_COUNTRY_CODE'],
            'cpv': r1['CPV'],
            'label': ''  # filled in by annotation_tool.py
        })

        if len(pairs_data) >= 500:
            break

    if len(pairs_data) >= 500:
        break

pairs_df = pd.DataFrame(pairs_data)
pairs_df.to_csv('candidate_pairs_for_annotation.csv', index=False)
