import pandas as pd
import re
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


df = pd.read_csv('export_CAN_2023.csv', encoding='latin-1', low_memory=False)
print(f"Loaded {len(df):,} records")

df['CAE_NAME_CLEAN'] = df['CAE_NAME'].apply(clean_text)
df['WIN_NAME_CLEAN'] = df['WIN_NAME'].apply(clean_text)
df['TITLE_CLEAN'] = df['TITLE'].apply(clean_text)

df['COMBINED_TEXT'] = (
    df['CAE_NAME_CLEAN'] + ' ' +
    df['WIN_NAME_CLEAN'].fillna('') + ' ' +
    df['TITLE_CLEAN'].fillna('')
)
df['COMBINED_TEXT'] = df['COMBINED_TEXT'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Drop records missing WIN_NAME
before = len(df)
df = df.dropna(subset=['WIN_NAME'])
after = len(df)
print(f"Dropped {before - after:,} records with missing WIN_NAME; {after:,} remain")

# CPV-group median imputation for VALUE_EURO
cpv_medians = df.groupby('CPV')['VALUE_EURO'].median()
df['VALUE_EURO'] = df.apply(
    lambda row: cpv_medians[row['CPV']]
    if pd.isna(row['VALUE_EURO']) and row['CPV'] in cpv_medians.index
    else row['VALUE_EURO'],
    axis=1
)

# Stratified sample of 100K records by (CPV, country)
df['STRATUM'] = df['CPV'].astype(str) + '_' + df['ISO_COUNTRY_CODE'].astype(str)
sample_size = 100000

if len(df) <= sample_size:
    df_sample = df.copy()
else:
    stratum_counts = df['STRATUM'].value_counts()
    valid_stratums = stratum_counts[stratum_counts >= 2].index
    df_stratified = df[df['STRATUM'].isin(valid_stratums)]
    df_other = df[~df['STRATUM'].isin(valid_stratums)]

    sample_fraction = sample_size / len(df)
    if len(df_stratified) > 0:
        n_stratified = int(len(df_stratified) * sample_fraction)
        df_sample_1, _ = train_test_split(
            df_stratified,
            train_size=n_stratified,
            stratify=df_stratified['STRATUM'],
            random_state=42
        )
    else:
        df_sample_1 = pd.DataFrame()

    if len(df_other) > 0:
        n_other = min(sample_size - len(df_sample_1), len(df_other))
        df_sample_2 = df_other.sample(n=n_other, random_state=42)
    else:
        df_sample_2 = pd.DataFrame()

    df_sample = pd.concat([df_sample_1, df_sample_2], ignore_index=True)

print(f"Sample size: {len(df_sample):,} records")

columns_to_keep = [
    'ID_NOTICE_CAN',
    'CAE_NAME', 'CAE_NAME_CLEAN',
    'WIN_NAME', 'WIN_NAME_CLEAN',
    'TITLE', 'TITLE_CLEAN',
    'COMBINED_TEXT',
    'CPV',
    'ISO_COUNTRY_CODE',
    'VALUE_EURO',
    'YEAR',
    'DT_DISPATCH'
]

df_clean = df_sample[columns_to_keep].copy()
df_clean.to_csv('ted_clean_100k.csv', index=False, encoding='utf-8')
print(f"Saved: ted_clean_100k.csv ({len(df_clean):,} rows)")
