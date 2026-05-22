import pandas as pd
import numpy as np
import random
import re
import warnings
warnings.filterwarnings("ignore")

from Levenshtein import ratio as lev_ratio

random.seed(42)
np.random.seed(42)

INPUT_FILE   = "ted_clean_100k.csv"
OUTPUT_PAIRS = "synthetic_duplicate_pairs_v2.csv"
OUTPUT_GOLD  = "synthetic_gold_standard_v2.csv"

N_DUPLICATES = 2000
N_NON_DUPES  = 2000

VALUE_NOISE_MIN = 0.85
VALUE_NOISE_MAX = 1.15
CPV_MISMATCH_RATE = 0.15

ABBREVIATIONS = {
    "spitalul": "sp.", "hospital": "hosp.", "ministerul": "min.",
    "universitatea": "univ.", "universitary": "univ.", "university": "univ.",
    "national": "nat.", "municipal": "mun.", "general": "gen.",
    "regional": "reg.", "clinic": "clin.", "clinique": "clin.",
    "department": "dept.", "services": "serv.", "international": "intl.",
    "management": "mgmt.", "distribution": "distr.",
    "pharmaceutical": "pharm.", "farmaceutica": "farm.",
    "medical": "med.", "urgenta": "urg.", "judetean": "jud.",
}

ENCODING_MAP = {
    "ô": "Ã´", "ó": "Ã³", "ö": "Ã¶", "ü": "Ã¼", "ä": "Ã¤",
    "ą": "Ä\x85", "ę": "Ä\x99", "ś": "Å\x9b", "ł": "Å\x82",
    "ż": "Å¼",  "ź": "Å\xba", "ć": "Ä\x87", "ń": "Å\x84",
    "ș": "È\x99", "ț": "È\x9b", "â": "Ã¢",  "î": "Ã®",
    "ă": "Äƒ",  "č": "Ä\x8d", "š": "Å\xa1", "ž": "Å¾",
}


#Perturbation functions

def perturb_case(text):
    choice = random.choice(["lower", "title", "mixed"])
    if choice == "lower":
        return text.lower()
    elif choice == "title":
        return text.title()
    else:
        words = text.split()
        return " ".join(w.upper() if random.random() > 0.5 else w.lower() for w in words)


def perturb_encoding(text):
    result = text
    for char, broken in ENCODING_MAP.items():
        if char in result and random.random() > 0.5:
            result = result.replace(char, broken, 1)
        if char.upper() in result and random.random() > 0.5:
            result = result.replace(char.upper(), broken, 1)
    return result


def perturb_abbreviation(text):
    words = text.lower().split()
    result = []
    changed = False
    for w in words:
        clean_w = re.sub(r'[^\w]', '', w)
        if clean_w in ABBREVIATIONS and random.random() > 0.4:
            result.append(ABBREVIATIONS[clean_w])
            changed = True
        else:
            result.append(w)
    if not changed and words:
        for i, w in enumerate(result):
            clean_w = re.sub(r'[^\w]', '', w)
            if clean_w in ABBREVIATIONS:
                result[i] = ABBREVIATIONS[clean_w]
                break
    return " ".join(result)


def perturb_word_order(text):
    words = text.split()
    if len(words) < 3:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def perturb_word_drop(text):
    words = text.split()
    if len(words) < 3:
        return text
    candidates = [i for i, w in enumerate(words) if len(w) > 3]
    if not candidates:
        candidates = list(range(len(words)))
    drop_idx = random.choice(candidates)
    return " ".join(words[:drop_idx] + words[drop_idx + 1:])


PERTURBATION_TYPES = ["case", "encoding", "abbreviation", "word_order", "word_drop"]


def perturb_record(row):
    perturbed = row.copy()
    n_perturbations = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
    chosen = random.sample(PERTURBATION_TYPES, min(n_perturbations, len(PERTURBATION_TYPES)))
    fields_to_perturb = random.choice([["CAE_NAME"], ["WIN_NAME"], ["CAE_NAME", "WIN_NAME"]])

    for field in fields_to_perturb:
        col_clean = field + "_CLEAN"
        source_col = col_clean if col_clean in row.index and pd.notna(row[col_clean]) else field
        text = str(row[source_col]) if pd.notna(row[source_col]) else ""
        if not text or text == "nan":
            continue
        for ptype in chosen:
            if ptype == "case":
                text = perturb_case(text)
            elif ptype == "encoding":
                text = perturb_encoding(text)
            elif ptype == "abbreviation":
                text = perturb_abbreviation(text)
            elif ptype == "word_order":
                text = perturb_word_order(text)
            elif ptype == "word_drop":
                text = perturb_word_drop(text)
        perturbed[field] = text

    return perturbed, chosen


#Feature engineering

def safe_lev(a, b):
    if not a or not b:
        return 0.0
    return lev_ratio(str(a).lower(), str(b).lower())


def jaccard(a, b):
    if not a or not b:
        return 0.0
    set_a = set(str(a).lower().split())
    set_b = set(str(b).lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def jaro_winkler(a, b):
    try:
        import jellyfish
        return jellyfish.jaro_winkler_similarity(str(a).lower(), str(b).lower())
    except ImportError:
        return safe_lev(a, b)


def compute_features(row_a, row_b):
    features = {}

    cae_a = str(row_a.get("CAE_NAME", ""))
    cae_b = str(row_b.get("CAE_NAME", ""))
    features["cae_levenshtein"]  = safe_lev(cae_a, cae_b)
    features["cae_jaccard"]      = jaccard(cae_a, cae_b)
    features["cae_jaro_winkler"] = jaro_winkler(cae_a, cae_b)

    win_a = str(row_a.get("WIN_NAME", ""))
    win_b = str(row_b.get("WIN_NAME", ""))
    features["win_levenshtein"]  = safe_lev(win_a, win_b)
    features["win_jaccard"]      = jaccard(win_a, win_b)
    features["win_jaro_winkler"] = jaro_winkler(win_a, win_b)

    val_a = row_a.get("VALUE_EURO", np.nan)
    val_b = row_b.get("VALUE_EURO", np.nan)
    if pd.notna(val_a) and pd.notna(val_b) and max(float(val_a), float(val_b)) > 0:
        features["value_ratio"] = min(float(val_a), float(val_b)) / max(float(val_a), float(val_b))
    else:
        features["value_ratio"] = np.nan

    features["cpv_match"]     = int(str(row_a.get("CPV", "")) == str(row_b.get("CPV", "")))
    features["country_match"] = int(str(row_a.get("ISO_COUNTRY_CODE", "")) ==
                                    str(row_b.get("ISO_COUNTRY_CODE", "")))
    return features


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df[df["WIN_NAME"].notna() & (df["WIN_NAME"] != "")].copy()
    df = df.reset_index(drop=True)
    all_cpv_codes = df["CPV"].dropna().unique().tolist()

    # Generate duplicate pairs
    dup_records = []
    attempts = 0
    max_attempts = N_DUPLICATES * 10
    indices = list(range(len(df)))
    random.shuffle(indices)

    for idx in indices:
        if len(dup_records) >= N_DUPLICATES:
            break
        attempts += 1
        if attempts > max_attempts:
            break

        original = df.iloc[idx]
        perturbed, pert_types = perturb_record(original)

        sim = safe_lev(
            str(original.get("WIN_NAME", "")),
            str(perturbed.get("WIN_NAME", ""))
        )
        if sim > 0.98 or sim < 0.25:
            continue

        original_value = original.get("VALUE_EURO", np.nan)
        if pd.notna(original_value) and float(original_value) > 0:
            noise = np.random.uniform(VALUE_NOISE_MIN, VALUE_NOISE_MAX)
            noisy_value = float(original_value) * noise
        else:
            noisy_value = original_value

        original_cpv = original.get("CPV", "")
        if random.random() < CPV_MISMATCH_RATE and len(all_cpv_codes) > 1:
            alt_cpvs = [c for c in all_cpv_codes if c != original_cpv]
            varied_cpv = random.choice(alt_cpvs) if alt_cpvs else original_cpv
        else:
            varied_cpv = original_cpv

        perturbed_dict = perturbed.to_dict()
        perturbed_dict["VALUE_EURO"] = noisy_value
        perturbed_dict["CPV"] = varied_cpv

        features = compute_features(original.to_dict(), perturbed_dict)

        record = {
            "pair_id":      f"dup_{len(dup_records)}",
            "type":         "synthetic_duplicate",
            "perturbations": ",".join(pert_types),
            "cpv_varied":   int(varied_cpv != original_cpv),
            "value_noise":  round(noisy_value / float(original_value), 4)
                            if pd.notna(original_value) and float(original_value) > 0
                            else np.nan,
            "CAE_NAME_1":    original.get("CAE_NAME", ""),
            "WIN_NAME_1":    original.get("WIN_NAME", ""),
            "VALUE_EURO_1":  original.get("VALUE_EURO", np.nan),
            "CPV_1":         original_cpv,
            "COUNTRY_1":     original.get("ISO_COUNTRY_CODE", ""),
            "CAE_NAME_2":    perturbed.get("CAE_NAME", ""),
            "WIN_NAME_2":    perturbed.get("WIN_NAME", ""),
            "VALUE_EURO_2":  noisy_value,
            "CPV_2":         varied_cpv,
            "COUNTRY_2":     original.get("ISO_COUNTRY_CODE", ""),
            **features,
            "label": 1
        }
        dup_records.append(record)

    # Generate non-duplicate pairs
    non_dup_records = []
    used_pairs = set()

    while len(non_dup_records) < N_NON_DUPES:
        i, j = random.sample(range(len(df)), 2)
        pair_key = (min(i, j), max(i, j))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        row_a = df.iloc[i]
        row_b = df.iloc[j]

        # 70% same-country pairs (harder negatives)
        if row_a.get("ISO_COUNTRY_CODE") != row_b.get("ISO_COUNTRY_CODE"):
            if random.random() > 0.3:
                continue

        sim_win = safe_lev(str(row_a.get("WIN_NAME", "")), str(row_b.get("WIN_NAME", "")))
        if sim_win > 0.85:
            continue

        features = compute_features(row_a.to_dict(), row_b.to_dict())

        record = {
            "pair_id":      f"non_{len(non_dup_records)}",
            "type":         "real_non_duplicate",
            "perturbations":"none",
            "cpv_varied":   0,
            "value_noise":  np.nan,
            "CAE_NAME_1":   row_a.get("CAE_NAME", ""),
            "WIN_NAME_1":   row_a.get("WIN_NAME", ""),
            "VALUE_EURO_1": row_a.get("VALUE_EURO", np.nan),
            "CPV_1":        row_a.get("CPV", ""),
            "COUNTRY_1":    row_a.get("ISO_COUNTRY_CODE", ""),
            "CAE_NAME_2":   row_b.get("CAE_NAME", ""),
            "WIN_NAME_2":   row_b.get("WIN_NAME", ""),
            "VALUE_EURO_2": row_b.get("VALUE_EURO", np.nan),
            "CPV_2":        row_b.get("CPV", ""),
            "COUNTRY_2":    row_b.get("ISO_COUNTRY_CODE", ""),
            **features,
            "label": 0
        }
        non_dup_records.append(record)

    all_pairs = dup_records + non_dup_records
    random.shuffle(all_pairs)
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(OUTPUT_PAIRS, index=False)

    feature_cols = [
        "pair_id", "type", "perturbations", "cpv_varied",
        "cae_levenshtein", "cae_jaccard", "cae_jaro_winkler",
        "win_levenshtein", "win_jaccard", "win_jaro_winkler",
        "value_ratio", "cpv_match", "country_match",
        "label"
    ]
    gold_df = pairs_df[[c for c in feature_cols if c in pairs_df.columns]]
    gold_df.to_csv(OUTPUT_GOLD, index=False)

    print(f"Saved {OUTPUT_PAIRS} ({len(pairs_df):,} pairs)")
    print(f"Saved {OUTPUT_GOLD}")


if __name__ == "__main__":
    main()
