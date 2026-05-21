import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('export_CAN_2023.csv', encoding='latin-1', low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Missing values by column
missing = (df.isnull().mean() * 100).sort_values(ascending=False)
missing = missing[missing > 0]
print(f"\nColumns with missing values:\n{missing.round(1).to_string()}")

# Country distribution
country_col = None
for possible in ['ISO_COUNTRY_CODE', 'CAE_COUNTRY', 'COUNTRY', 'country']:
    if possible in df.columns:
        country_col = possible
        break

if country_col:
    country_counts = df[country_col].value_counts()
    print(f"\nCountry column: {country_col}")
    print(f"Unique countries: {df[country_col].nunique()}")
    print(country_counts.to_string())

# CPV category distribution
cpv_col = None
for possible in ['CPV', 'cpv', 'CPV_CODE', 'MAIN_CPV_CODE']:
    if possible in df.columns:
        cpv_col = possible
        break

if cpv_col:
    cpv_counts = df[cpv_col].value_counts()
    print(f"\nCPV column: {cpv_col}")
    print(f"Unique CPV codes: {df[cpv_col].nunique()}")
    print(f"Top 15:\n{cpv_counts.head(15).to_string()}")

# Text fields (for SBERT input candidates)
text_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        avg_len = df[col].dropna().str.len().mean()
        if avg_len and avg_len > 10:
            text_cols.append((col, avg_len))
text_cols.sort(key=lambda x: x[1], reverse=True)

print("\nText-containing columns (by avg length):")
for col, avg_len in text_cols:
    null_pct = df[col].isnull().mean() * 100
    sample = str(df[col].dropna().iloc[0])[:60] if len(df[col].dropna()) > 0 else "empty"
    print(f"  {col}: avg_len={avg_len:.0f}, null={null_pct:.1f}%, sample='{sample}'")

# Contract value statistics
value_col = None
for possible in ['VALUE_EURO', 'CONTRACT_VALUE', 'VALUE', 'AWARD_VALUE_EURO']:
    if possible in df.columns:
        value_col = possible
        break

if value_col:
    vals = df[value_col].dropna()
    q99 = vals.quantile(0.99)
    print(f"\nValue column: {value_col}")
    print(f"  Missing: {df[value_col].isnull().sum():,} "
          f"({df[value_col].isnull().mean()*100:.1f}%)")
    print(f"  Min={vals.min():,.0f}, Median={vals.median():,.0f}, "
          f"Mean={vals.mean():,.0f}, Max={vals.max():,.0f}")
    print(f"  99th percentile: {q99:,.0f}, outliers above: {(vals > q99).sum()}")
