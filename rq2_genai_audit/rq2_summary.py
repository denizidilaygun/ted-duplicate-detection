import pandas as pd

AUDIT_FILE    = "audit_summary.csv"
FIDELITY_FILE = "fidelity_results.csv"
OUTPUT_FILE   = "rq2_final_results.csv"


def main():
    audit_df = pd.read_csv(AUDIT_FILE)
    fidelity_df = pd.read_csv(FIDELITY_FILE)
    print(f"  Audit results loaded   : {len(audit_df)} rows")
    print(f"  Fidelity results loaded: {len(fidelity_df)} rows")

    combined = pd.merge(audit_df, fidelity_df, on="model", how="outer")

    thesis_columns = [
        "model",
        # Memorization metrics
        "exact_rate_pct",
        "near_rate_pct",
        "gap_rate_pct",
        # Fidelity metrics
        "ks_VALUE_EURO",
        "tvd_CPV",
        "tvd_ISO_COUNTRY_CODE",
        "tvd_CAE_NAME",
        "tvd_WIN_NAME",
        "sdv_quality_score",
    ]
    available = [c for c in thesis_columns if c in combined.columns]
    final = combined[available].copy()

    final.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE}")

    print(final.to_string(index=False))

    if len(final) >= 2:
        ctgan = final[final["model"] == "CTGAN"].iloc[0]
        tvae  = final[final["model"] == "TVAE"].iloc[0]

        # Memorization comparison
        mem_winner = "CTGAN" if ctgan["near_rate_pct"] < tvae["near_rate_pct"] else "TVAE"
        print(f"  Memorization (lower is safer):")
        print(f"    CTGAN near-duplicate rate : {ctgan['near_rate_pct']:.2f}%")
        print(f"    TVAE near-duplicate rate  : {tvae['near_rate_pct']:.2f}%")
        print(f"    Safer model               : {mem_winner}")

        # Fidelity comparison 
        fid_winner = "CTGAN" if ctgan["sdv_quality_score"] > tvae["sdv_quality_score"] else "TVAE"
        print(f"\n  Fidelity (higher is better):")
        print(f"    CTGAN SDV quality score   : {ctgan['sdv_quality_score']:.4f}")
        print(f"    TVAE SDV quality score    : {tvae['sdv_quality_score']:.4f}")
        print(f"    More faithful model       : {fid_winner}")

        # Framework value
        print(f"\n  Additional memorizations caught beyond exact-match:")
        for _, row in final.iterrows():
            print(f"    {row['model']:<6}: {row['gap_rate_pct']:.2f}% of synthetic records")


if __name__ == "__main__":
    main()
