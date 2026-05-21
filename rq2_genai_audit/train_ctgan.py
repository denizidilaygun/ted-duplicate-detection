import pandas as pd
import json
import time
from datetime import datetime

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

INPUT_DATA     = "ted_for_genai.csv"
INPUT_METADATA = "ted_metadata.json"
OUTPUT_MODEL   = "ctgan_model.pkl"
OUTPUT_LOG     = "ctgan_training_log.txt"

# Increased from 100 to 300 (SDV default) to improve fidelity
N_EPOCHS   = 300
BATCH_SIZE = 500
VERBOSE    = True


def main():
    start_time = time.time()

    df = pd.read_csv(INPUT_DATA)
    print(f"  Records   : {len(df):,}")
    print(f"  Columns   : {list(df.columns)}")

    with open(INPUT_METADATA, "r") as f:
        metadata_dict = json.load(f)

    metadata = SingleTableMetadata()
    for column_name, column_info in metadata_dict["columns"].items():
        metadata.add_column(column_name, sdtype=column_info["sdtype"])
    print(f"  Columns registered: {len(metadata_dict['columns'])}")

    print(f"  Epochs     : {N_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")

    model = CTGANSynthesizer(
        metadata=metadata,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE
    )

    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    model.fit(df)

    elapsed = time.time() - start_time
    elapsed_minutes = elapsed / 60
    print(f"  Training took {elapsed_minutes:.1f} minutes")

    model.save(OUTPUT_MODEL)
    print(f"  Saved: {OUTPUT_MODEL}")

    with open(OUTPUT_LOG, "w") as f:
        f.write(f"CTGAN Training Log\n")
        f.write(f"==================\n")
        f.write(f"Date            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Records         : {len(df):,}\n")
        f.write(f"Columns         : {list(df.columns)}\n")
        f.write(f"Epochs          : {N_EPOCHS}\n")
        f.write(f"Batch size      : {BATCH_SIZE}\n")
        f.write(f"Training time   : {elapsed_minutes:.1f} minutes\n")
        f.write(f"Output model    : {OUTPUT_MODEL}\n")
    print(f"  Saved: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
