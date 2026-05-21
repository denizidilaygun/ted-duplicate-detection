import pandas as pd
import time

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer

CTGAN_MODEL_FILE  = "ctgan_model.pkl"
TVAE_MODEL_FILE   = "tvae_model.pkl"
CTGAN_OUTPUT      = "synthetic_ctgan.csv"
TVAE_OUTPUT       = "synthetic_tvae.csv"
REAL_DATA_FILE    = "ted_for_genai.csv"  # for comparison

# Same as training set size, so fidelity comparison is fair
N_SYNTHETIC = 10000


def generate_and_save(model_class, model_file, output_file, model_name, n_records):
    print(f"\n{model_name}:")
    start_time = time.time()
    model = model_class.load(model_file)
    print(f"  Generating {n_records:,} synthetic records...")
    synthetic = model.sample(num_rows=n_records)
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f} seconds.")
    print(f"  Synthetic records shape: {synthetic.shape}")

    synthetic.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")

    return synthetic


def main():
    print("\nLoading real data for comparison...")
    real_df = pd.read_csv(REAL_DATA_FILE)
    print(f"  Real data shape: {real_df.shape}")

    ctgan_synthetic = generate_and_save(
        CTGANSynthesizer, CTGAN_MODEL_FILE, CTGAN_OUTPUT, "CTGAN", N_SYNTHETIC
    )

    tvae_synthetic = generate_and_save(
        TVAESynthesizer, TVAE_MODEL_FILE, TVAE_OUTPUT, "TVAE", N_SYNTHETIC
    )

    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 200)

    print("\nReal records (sample of 5):")
    print(real_df.sample(5, random_state=42).to_string(index=False))

    print("\nCTGAN synthetic records (sample of 5):")
    print(ctgan_synthetic.sample(5, random_state=42).to_string(index=False))

    print("\nTVAE synthetic records (sample of 5):")
    print(tvae_synthetic.sample(5, random_state=42).to_string(index=False))

    print(f"\n  VALUE_EURO statistics:")
    print(f"    Real    : mean={real_df['VALUE_EURO'].mean():>12,.0f}, "
          f"median={real_df['VALUE_EURO'].median():>12,.0f}")
    print(f"    CTGAN   : mean={ctgan_synthetic['VALUE_EURO'].mean():>12,.0f}, "
          f"median={ctgan_synthetic['VALUE_EURO'].median():>12,.0f}")
    print(f"    TVAE    : mean={tvae_synthetic['VALUE_EURO'].mean():>12,.0f}, "
          f"median={tvae_synthetic['VALUE_EURO'].median():>12,.0f}")

    print(f"\n  Top 5 countries (real):")
    for c, n in real_df["ISO_COUNTRY_CODE"].value_counts().head(5).items():
        print(f"    {c}: {n/len(real_df)*100:.1f}%")

    print(f"\n  Top 5 countries (CTGAN):")
    for c, n in ctgan_synthetic["ISO_COUNTRY_CODE"].value_counts().head(5).items():
        print(f"    {c}: {n/len(ctgan_synthetic)*100:.1f}%")

    print(f"\n  Top 5 countries (TVAE):")
    for c, n in tvae_synthetic["ISO_COUNTRY_CODE"].value_counts().head(5).items():
        print(f"    {c}: {n/len(tvae_synthetic)*100:.1f}%")


if __name__ == "__main__":
    main()
