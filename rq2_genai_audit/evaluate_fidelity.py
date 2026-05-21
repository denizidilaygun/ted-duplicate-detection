import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import ks_2samp
import json

# Try to use SDV's built-in evaluator, fall back gracefully if API changed
try:
    from sdv.evaluation.single_table import evaluate_quality
    SDV_EVAL_AVAILABLE = True
except ImportError:
    SDV_EVAL_AVAILABLE = False

from sdv.metadata import SingleTableMetadata

REAL_DATA_FILE  = "ted_for_genai.csv"
CTGAN_FILE      = "synthetic_ctgan.csv"
TVAE_FILE       = "synthetic_tvae.csv"
METADATA_FILE   = "ted_metadata.json"
OUTPUT_FILE     = "fidelity_results.csv"

NUMERICAL_COLUMNS = ["VALUE_EURO"]
CATEGORICAL_COLUMNS = ["CPV", "ISO_COUNTRY_CODE", "CAE_NAME", "WIN_NAME"]


def compute_ks_statistic(real_values, synthetic_values):
    real_values      = real_values.dropna().values
    synthetic_values = synthetic_values.dropna().values
    if len(real_values) == 0 or len(synthetic_values) == 0:
        return np.nan
    statistic, _ = ks_2samp(real_values, synthetic_values)
    return statistic


def compute_total_variation_distance(real_values, synthetic_values):
    real_props = real_values.value_counts(normalize=True)
    synthetic_props = synthetic_values.value_counts(normalize=True)

    all_categories = set(real_props.index) | set(synthetic_props.index)

    total_distance = 0.0
    for category in all_categories:
        p_real = real_props.get(category, 0.0)
        p_synthetic = synthetic_props.get(category, 0.0)
        total_distance += abs(p_real - p_synthetic)
    return total_distance / 2


def evaluate_one_model(real_df, synthetic_df, model_name, metadata=None):
    print(f"Evaluating {model_name}")

    results = {"model": model_name}

    print(f"\n  Numerical columns (Kolmogorov-Smirnov, lower is better):")
    for col in NUMERICAL_COLUMNS:
        if col in real_df.columns and col in synthetic_df.columns:
            ks = compute_ks_statistic(real_df[col], synthetic_df[col])
            print(f"    {col:<22}: {ks:.4f}")
            results[f"ks_{col}"] = round(ks, 4)

    print(f"\n  Categorical columns (Total Variation Distance, lower is better):")
    for col in CATEGORICAL_COLUMNS:
        if col in real_df.columns and col in synthetic_df.columns:
            tvd = compute_total_variation_distance(real_df[col], synthetic_df[col])
            print(f"    {col:<22}: {tvd:.4f}")
            results[f"tvd_{col}"] = round(tvd, 4)

    if SDV_EVAL_AVAILABLE and metadata is not None:
        print(f"\n  SDV overall quality score (0 to 1, higher is better):")
        try:
            quality_report = evaluate_quality(
                real_data=real_df,
                synthetic_data=synthetic_df,
                metadata=metadata,
            )
            overall_score = quality_report.get_score()
            print(f"    Overall            : {overall_score:.4f}")
            results["sdv_quality_score"] = round(overall_score, 4)
        except Exception as e:
            print(f"    SDV evaluation failed: {e}")
            results["sdv_quality_score"] = np.nan
    else:
        results["sdv_quality_score"] = np.nan

    return results


def main():
    print("Fidelity Evaluation (RQ2)")

    real_df  = pd.read_csv(REAL_DATA_FILE)
    ctgan_df = pd.read_csv(CTGAN_FILE)
    tvae_df  = pd.read_csv(TVAE_FILE)
    print(f"  Real    : {len(real_df):,}")
    print(f"  CTGAN   : {len(ctgan_df):,}")
    print(f"  TVAE    : {len(tvae_df):,}")

    metadata = None
    try:
        with open(METADATA_FILE, "r") as f:
            metadata_dict = json.load(f)
        metadata = SingleTableMetadata()
        for column_name, column_info in metadata_dict["columns"].items():
            metadata.add_column(column_name, sdtype=column_info["sdtype"])
        print(f"  Metadata loaded.")
    except Exception as e:
        print(f"  Could not load metadata: {e}")

    ctgan_results = evaluate_one_model(real_df, ctgan_df, "CTGAN", metadata)
    tvae_results  = evaluate_one_model(real_df, tvae_df,  "TVAE",  metadata)

    summary_df = pd.DataFrame([ctgan_results, tvae_results])
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"\n{summary_df.to_string(index=False)}")

    # Winner-by-metric summary
    for col in NUMERICAL_COLUMNS:
        key = f"ks_{col}"
        if key in ctgan_results and key in tvae_results:
            ctgan_val = ctgan_results[key]
            tvae_val  = tvae_results[key]
            winner = "CTGAN" if ctgan_val < tvae_val else "TVAE"
            print(f"  {col} distribution: {winner} is closer to real "
                  f"(CTGAN={ctgan_val:.4f}, TVAE={tvae_val:.4f})")

    for col in CATEGORICAL_COLUMNS:
        key = f"tvd_{col}"
        if key in ctgan_results and key in tvae_results:
            ctgan_val = ctgan_results[key]
            tvae_val  = tvae_results[key]
            winner = "CTGAN" if ctgan_val < tvae_val else "TVAE"
            print(f"  {col} distribution: {winner} is closer to real "
                  f"(CTGAN={ctgan_val:.4f}, TVAE={tvae_val:.4f})")

    if not np.isnan(ctgan_results.get("sdv_quality_score", np.nan)):
        c = ctgan_results["sdv_quality_score"]
        t = tvae_results["sdv_quality_score"]
        winner = "CTGAN" if c > t else "TVAE"
        print(f"  SDV overall: {winner} has higher quality "
              f"(CTGAN={c:.4f}, TVAE={t:.4f})")


if __name__ == "__main__":
    main()
