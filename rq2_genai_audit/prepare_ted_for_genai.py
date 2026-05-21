import pandas as pd
import numpy as np
import json
import random

random.seed(42)
np.random.seed(42)

INPUT_FILE       = "ted_clean_100k.csv"
OUTPUT_DATA_FILE = "ted_for_genai.csv"
OUTPUT_METADATA  = "ted_metadata.json"

# Reduced from 20,000 to 10,000 for training tractability
SAMPLE_SIZE = 10000

# Cardinality reduction thresholds
MIN_FREQUENCY_CAE = 5
MIN_FREQUENCY_WIN = 10
CPV_TRUNCATE_DIGITS = 4

RARE_CAE_PLACEHOLDER = "RARE_AUTHORITY"
RARE_WIN_PLACEHOLDER = "RARE_SUPPLIER"

COLUMNS_TO_KEEP = [
    "CAE_NAME", "WIN_NAME", "VALUE_EURO", "CPV", "ISO_COUNTRY_CODE"
]


def reduce_cardinality(df, column_name, min_frequency, placeholder):
    value_counts = df[column_name].value_counts()
    print(f"  {column_name}:")
    print(f"    Unique values before: {len(value_counts):,}")
    print(f"    Min frequency       : {min_frequency}")

    common_values = value_counts[value_counts >= min_frequency].index.tolist()
    print(f"    Common values kept  : {len(common_values):,}")

    rare_count = (~df[column_name].isin(common_values)).sum()
    df[column_name] = df[column_name].where(
        df[column_name].isin(common_values),
        other=placeholder
    )

    print(f"    Unique values after : {df[column_name].nunique():,}")
    print(f"    Records replaced    : {rare_count:,} "
          f"({rare_count / len(df) * 100:.1f}%)")
    return df


def truncate_cpv(df, n_digits):
    print(f"  CPV truncation to {n_digits} digits:")
    print(f"    Unique codes before: {df['CPV'].nunique():,}")
    df["CPV"] = df["CPV"].astype(str).str[:n_digits]
    df = df[df["CPV"] != ""]
    df["CPV"] = df["CPV"].astype(int)
    print(f"    Unique codes after : {df['CPV'].nunique():,}")
    return df


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"  Records loaded: {len(df):,}")

    df = df[COLUMNS_TO_KEEP].copy()
    print(f"  Columns kept: {COLUMNS_TO_KEEP}")

    before = len(df)
    df = df.dropna(subset=["CAE_NAME", "WIN_NAME", "VALUE_EURO", "CPV"])
    df = df[df["WIN_NAME"] != ""]
    df = df[df["CAE_NAME"] != ""]
    print(f"  Before: {before:,}  -->  After: {len(df):,}")

    value_cap = df["VALUE_EURO"].quantile(0.99)
    n_capped = (df["VALUE_EURO"] > value_cap).sum()
    df["VALUE_EURO"] = df["VALUE_EURO"].clip(upper=value_cap)
    print(f"  Cap value      : {value_cap:,.2f} EUR")
    print(f"  Records capped : {n_capped:,}")

    df = truncate_cpv(df, CPV_TRUNCATE_DIGITS)

    df = reduce_cardinality(df, "CAE_NAME", MIN_FREQUENCY_CAE, RARE_CAE_PLACEHOLDER)
    df = reduce_cardinality(df, "WIN_NAME", MIN_FREQUENCY_WIN, RARE_WIN_PLACEHOLDER)

    if len(df) <= SAMPLE_SIZE:
        df_sample = df.copy()
    else:
        df_sample = df.groupby("ISO_COUNTRY_CODE", group_keys=False).apply(
            lambda group: group.sample(
                n=max(1, int(len(group) * SAMPLE_SIZE / len(df))),
                random_state=42
            )
        )
        if len(df_sample) > SAMPLE_SIZE:
            df_sample = df_sample.sample(n=SAMPLE_SIZE, random_state=42)
        elif len(df_sample) < SAMPLE_SIZE:
            remaining = df.drop(df_sample.index)
            n_more = SAMPLE_SIZE - len(df_sample)
            df_sample = pd.concat([df_sample, remaining.sample(n=n_more, random_state=42)])

    df_sample = df_sample.reset_index(drop=True)
    print(f"  Final sample: {len(df_sample):,}")

    country_counts = df_sample["ISO_COUNTRY_CODE"].value_counts()
    print(f"  Unique countries: {len(country_counts)}")
    for country, count in country_counts.head(5).items():
        print(f"    {country}: {count:,} ({count / len(df_sample) * 100:.1f}%)")

    metadata = {
        "columns": {
            "CAE_NAME"        : {"sdtype": "categorical"},
            "WIN_NAME"        : {"sdtype": "categorical"},
            "VALUE_EURO"      : {"sdtype": "numerical"},
            "CPV"             : {"sdtype": "categorical"},
            "ISO_COUNTRY_CODE": {"sdtype": "categorical"},
        }
    }

    df_sample.to_csv(OUTPUT_DATA_FILE, index=False)
    print(f"  Saved: {OUTPUT_DATA_FILE}")
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {OUTPUT_METADATA}")

    print(f"  Final dataset size : {len(df_sample):,}")
    print(f"  Unique CAE names   : {df_sample['CAE_NAME'].nunique():,}")
    print(f"  Unique WIN names   : {df_sample['WIN_NAME'].nunique():,}")
    print(f"  Unique CPV codes   : {df_sample['CPV'].nunique():,}")
    print(f"  Unique countries   : {df_sample['ISO_COUNTRY_CODE'].nunique():,}")

    total_cardinality = (
        df_sample['CAE_NAME'].nunique() +
        df_sample['WIN_NAME'].nunique() +
        df_sample['CPV'].nunique() +
        df_sample['ISO_COUNTRY_CODE'].nunique()
    )
    print(f"  Total cardinality  : {total_cardinality:,} internal columns")


if __name__ == "__main__":
    main()
