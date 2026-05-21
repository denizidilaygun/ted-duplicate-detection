import pandas as pd
import os
import json

INPUT_FILE = "candidate_pairs_for_annotation.csv"
OUTPUT_FILE = "annotated_gold_standard.csv"
PROGRESS_FILE = "annotation_progress.json"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "labels": {}}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def truncate(text, n=60):
    text = str(text) if pd.notna(text) else "(empty)"
    return text[:n] + "..." if len(text) > n else text


def show_pair(row, idx, total):
    print(f"\nPair {idx+1} / {total}  |  pair_id: {row.get('pair_id', idx)}")

    print(f"\n{BLUE}-- RECORD A --{RESET}")
    print(f"  CAE (Authority)  : {truncate(row.get('cae_name_1', row.get('CAE_NAME_1', '')))}")
    print(f"  WIN (Supplier)   : {truncate(row.get('win_name_1', row.get('WIN_NAME_1', '')))}")
    print(f"  Title            : {truncate(row.get('title_1', row.get('TITLE_1', '')))}")
    print(f"  Value (EUR)      : {row.get('value_euro_1', row.get('VALUE_EURO_1', 'N/A'))}")
    print(f"  CPV              : {row.get('cpv_1', row.get('CPV_1', 'N/A'))}")
    print(f"  Country          : {row.get('country_1', row.get('COUNTRY_1', 'N/A'))}")

    print(f"\n{YELLOW}-- RECORD B --{RESET}")
    print(f"  CAE (Authority)  : {truncate(row.get('cae_name_2', row.get('CAE_NAME_2', '')))}")
    print(f"  WIN (Supplier)   : {truncate(row.get('win_name_2', row.get('WIN_NAME_2', '')))}")
    print(f"  Title            : {truncate(row.get('title_2', row.get('TITLE_2', '')))}")
    print(f"  Value (EUR)      : {row.get('value_euro_2', row.get('VALUE_EURO_2', 'N/A'))}")
    print(f"  CPV              : {row.get('cpv_2', row.get('CPV_2', 'N/A'))}")
    print(f"  Country          : {row.get('country_2', row.get('COUNTRY_2', 'N/A'))}")

    print(f"\n{BOLD}Similarity scores:{RESET}")
    for feat in ['win_levenshtein', 'win_jaccard', 'win_jaro_winkler',
                 'cae_levenshtein', 'cae_jaccard', 'cae_jaro_winkler',
                 'value_ratio', 'cpv_match', 'country_match']:
        val = row.get(feat)
        if pd.notna(val):
            bar = "#" * int(float(val) * 20) if feat != 'cpv_match' else ""
            print(f"  {feat:<22}: {float(val):.3f}  {bar}")


def _save_output(df, labels, output_file, total):
    df["label"] = df.index.map(lambda i: labels.get(str(i), -1))
    labeled = df[df["label"] != -1].copy()
    labeled.to_csv(output_file, index=False)
    n_dup = (labeled["label"] == 1).sum()
    n_non = (labeled["label"] == 0).sum()
    print(f"\nSaved: {output_file}")
    print(f"  Annotated: {len(labeled)} / {total}")
    print(f"  Duplicate: {n_dup} ({n_dup/len(labeled)*100:.1f}%)")
    print(f"  Non-duplicate: {n_non} ({n_non/len(labeled)*100:.1f}%)")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"{RED}ERROR: {INPUT_FILE} not found.{RESET}")
        return

    df = pd.read_csv(INPUT_FILE)
    total = len(df)
    print(f"Gold Standard Annotation Tool")
    print(f"  Total pairs : {total}")
    print(f"  Input       : {INPUT_FILE}")
    print(f"  Output      : {OUTPUT_FILE}")
    print(f"\nCommands:")
    print(f"  1 = Duplicate")
    print(f"  0 = Non-duplicate")
    print(f"  s = Skip")
    print(f"  q = Save and quit")
    print(f"  b = Back to previous pair")

    progress = load_progress()
    labels = progress["labels"]
    completed_set = set(progress["completed"])

    remaining = [i for i in range(total) if str(i) not in completed_set]
    if len(remaining) < total:
        print(f"\nResuming: {total - len(remaining)}/{total} already done")

    i_list = remaining.copy()
    i_ptr = 0

    while i_ptr < len(i_list):
        idx = i_list[i_ptr]
        row = df.iloc[idx]
        show_pair(row, idx, total)

        n_done = len(labels)
        n_dup = sum(1 for v in labels.values() if v == 1)
        print(f"\nProgress: {n_done}/{total} done  |  "
              f"Duplicate: {n_dup}  |  Non-dup: {n_done - n_dup}")

        while True:
            ans = input(f"\nLabel (1/0/s/q/b): ").strip().lower()
            if ans in ("1", "0"):
                labels[str(idx)] = int(ans)
                completed_set.add(str(idx))
                progress["completed"] = list(completed_set)
                progress["labels"] = labels
                save_progress(progress)
                i_ptr += 1
                break
            elif ans == "s":
                i_ptr += 1
                break
            elif ans == "q":
                _save_output(df, labels, OUTPUT_FILE, total)
                return
            elif ans == "b":
                if i_ptr > 0:
                    i_ptr -= 1
                break
            else:
                print(f"  Invalid input. Use 1, 0, s, q or b.")

    _save_output(df, labels, OUTPUT_FILE, total)


if __name__ == "__main__":
    main()
