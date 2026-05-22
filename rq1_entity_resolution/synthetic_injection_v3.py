import pandas as pd
import numpy as np
import random
import re

from Levenshtein import ratio as levenshtein_ratio

random.seed(42)
np.random.seed(42)

INPUT_FILE   = "ted_clean_100k.csv"
OUTPUT_PAIRS = "synthetic_duplicate_pairs_v3.csv"
OUTPUT_GOLD  = "synthetic_gold_standard_v3.csv"

N_DUPLICATES     = 2000
N_NON_DUPLICATES = 2000

VALUE_NOISE_MIN = 0.85
VALUE_NOISE_MAX = 1.15
CPV_MISMATCH_RATE = 0.15

ABBREVIATIONS = {
    "spitalul": "sp.", "hospital": "hosp.", "ministerul": "min.",
    "universitatea": "univ.", "university": "univ.", "national": "nat.",
    "municipal": "mun.", "general": "gen.", "regional": "reg.",
    "clinic": "clin.", "department": "dept.", "services": "serv.",
    "international": "intl.", "medical": "med.",
}

ENCODING_MAP = {
    "ô": "Ã´", "ó": "Ã³", "ö": "Ã¶", "ü": "Ã¼",
    "ą": "Ä\x85", "ę": "Ä\x99", "ś": "Å\x9b", "ł": "Å\x82",
    "ș": "È\x99", "ț": "È\x9b", "â": "Ã¢", "î": "Ã®",
    "ă": "Äƒ", "č": "Ä\x8d", "š": "Å\xa1",
}

TRUNCATION_KEYWORDS = [
    "w ", "we ", "de ", "du ", "di ", "del ",
    "of ", "for ", "at ", "szpitala", "regional",
    "county", "district", "municipal",
]


# Perturbation functions

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
    for real_char, broken_char in ENCODING_MAP.items():
        if real_char in result and random.random() > 0.5:
            result = result.replace(real_char, broken_char, 1)
    return result


def perturb_abbreviation(text):
    words = text.lower().split()
    result = []
    changed = False
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in ABBREVIATIONS and random.random() > 0.4:
            result.append(ABBREVIATIONS[clean_word])
            changed = True
        else:
            result.append(word)
    if not changed:
        for i, word in enumerate(result):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in ABBREVIATIONS:
                result[i] = ABBREVIATIONS[clean_word]
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
    drop_index = random.choice(candidates)
    return " ".join(words[:drop_index] + words[drop_index + 1:])


def perturb_name_truncation(text):
    # Drops everything after a location or qualifier keyword.
    text_lower = text.lower()
    best_cut = None

    for keyword in TRUNCATION_KEYWORDS:
        position = text_lower.find(keyword, 5)
        if position != -1:
            prefix = text[:position].strip()
            if len(prefix.split()) >= 2:
                if best_cut is None or position < best_cut[0]:
                    best_cut = (position, prefix)

    if best_cut:
        return best_cut[1]

    # Fallback: keep only first half of words
    words = text.split()
    if len(words) >= 4:
        return " ".join(words[:len(words) // 2])
    return text


def perturb_punctuation_spacing(text):
    # Adds or removes spaces around dots in abbreviations
    mode = random.choice(["expand", "compact"])
    if mode == "expand":
        result = re.sub(r'\.(?=[^\s])', '. ', text)
    else:
        result = re.sub(r'\.\s+', '.', text)
    result = re.sub(r'  +', ' ', result).strip()
    if result == text or len(result) < 2:
        return text
    return result


ALL_PERTURBATION_TYPES = [
    "case", "encoding", "abbreviation",
    "word_order", "word_drop",
    "name_truncation", "punctuation_spacing",
]


def apply_perturbations(record):
    perturbed = record.copy()

    # Pick 1-3 perturbation types (40/40/20% weights)
    n = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
    chosen_types = random.sample(ALL_PERTURBATION_TYPES, n)

    for ptype in chosen_types:
        # name_truncation only applies to authority names
        if ptype == "name_truncation":
            fields = ["CAE_NAME"]
        else:
            fields = random.choice([
                ["CAE_NAME"], ["WIN_NAME"], ["CAE_NAME", "WIN_NAME"]
            ])

        for field in fields:
            text = str(record.get(field, ""))
            if not text or text == "nan":
                continue

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
            elif ptype == "name_truncation":
                text = perturb_name_truncation(text)
            elif ptype == "punctuation_spacing":
                text = perturb_punctuation_spacing(text)

            perturbed[field] = text

    return perturbed, chosen_types


#Feature computation

def compute_levenshtein(text_a, text_b):
    if not text_a or not text_b:
        return 0.0
    return levenshtein_ratio(str(text_a).lower(), str(text_b).lower())


def compute_jaccard(text_a, text_b):
    if not text_a or not text_b:
        return 0.0
    words_a = set(str(text_a).lower().split())
    words_b = set(str(text_b).lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def compute_jaro_winkler(text_a, text_b):
    try:
        import jellyfish
        return jellyfish.jaro_winkler_similarity(
            str(text_a).lower(), str(text_b).lower()
        )
    except ImportError:
        return compute_levenshtein(text_a, text_b)


def compute_all_features(record_a, record_b):
    features = {}

    cae_a = str(record_a.get("CAE_NAME", ""))
    cae_b = str(record_b.get("CAE_NAME", ""))
    features["cae_levenshtein"]  = compute_levenshtein(cae_a, cae_b)
    features["cae_jaccard"]      = compute_jaccard(cae_a, cae_b)
    features["cae_jaro_winkler"] = compute_jaro_winkler(cae_a, cae_b)

    win_a = str(record_a.get("WIN_NAME", ""))
    win_b = str(record_b.get("WIN_NAME", ""))
    features["win_levenshtein"]  = compute_levenshtein(win_a, win_b)
    features["win_jaccard"]      = compute_jaccard(win_a, win_b)
    features["win_jaro_winkler"] = compute_jaro_winkler(win_a, win_b)

    val_a = record_a.get("VALUE_EURO", np.nan)
    val_b = record_b.get("VALUE_EURO", np.nan)
    try:
        v1, v2 = float(val_a), float(val_b)
        if v1 > 0 and v2 > 0:
            features["value_ratio"] = min(v1, v2) / max(v1, v2)
        else:
            features["value_ratio"] = np.nan
    except (TypeError, ValueError):
        features["value_ratio"] = np.nan

    features["cpv_match"] = int(
        str(record_a.get("CPV", "")) == str(record_b.get("CPV", ""))
    )
    features["country_match"] = int(
        str(record_a.get("ISO_COUNTRY_CODE", "")) ==
        str(record_b.get("ISO_COUNTRY_CODE", ""))
    )
    return features


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df[df["WIN_NAME"].notna() & (df["WIN_NAME"] != "")].copy()
    df = df.reset_index(drop=True)
    all_cpv_codes = df["CPV"].dropna().unique().tolist()

    # Duplicate pairs
    duplicate_pairs = []
    indices = list(range(len(df)))
    random.shuffle(indices)

    for idx in indices:
        if len(duplicate_pairs) >= N_DUPLICATES:
            break

        original = df.iloc[idx]
        perturbed, pert_types = apply_perturbations(original)

        sim_win = compute_levenshtein(
            str(original.get("WIN_NAME", "")),
            str(perturbed.get("WIN_NAME", ""))
        )
        sim_cae = compute_levenshtein(
            str(original.get("CAE_NAME", "")),
            str(perturbed.get("CAE_NAME", ""))
        )
        if max(sim_win, sim_cae) > 0.99 or max(sim_win, sim_cae) < 0.20:
            continue

        original_value = original.get("VALUE_EURO", np.nan)
        if pd.notna(original_value) and float(original_value) > 0:
            noise_factor = np.random.uniform(VALUE_NOISE_MIN, VALUE_NOISE_MAX)
            noisy_value  = float(original_value) * noise_factor
        else:
            noisy_value = original_value

        original_cpv = original.get("CPV", "")
        if random.random() < CPV_MISMATCH_RATE:
            other_cpvs = [c for c in all_cpv_codes if c != original_cpv]
            varied_cpv = random.choice(other_cpvs) if other_cpvs else original_cpv
        else:
            varied_cpv = original_cpv

        perturbed_dict = perturbed.to_dict()
        perturbed_dict["VALUE_EURO"] = noisy_value
        perturbed_dict["CPV"]        = varied_cpv

        features = compute_all_features(original.to_dict(), perturbed_dict)

        pair = {
            "pair_id"      : f"dup_{len(duplicate_pairs)}",
            "type"         : "synthetic_duplicate",
            "perturbations": ",".join(pert_types),
            "CAE_NAME_1"   : original.get("CAE_NAME", ""),
            "WIN_NAME_1"   : original.get("WIN_NAME", ""),
            "VALUE_EURO_1" : original.get("VALUE_EURO", np.nan),
            "CPV_1"        : original_cpv,
            "COUNTRY_1"    : original.get("ISO_COUNTRY_CODE", ""),
            "CAE_NAME_2"   : perturbed.get("CAE_NAME", ""),
            "WIN_NAME_2"   : perturbed.get("WIN_NAME", ""),
            "VALUE_EURO_2" : noisy_value,
            "CPV_2"        : varied_cpv,
            "COUNTRY_2"    : original.get("ISO_COUNTRY_CODE", ""),
            **features,
            "label"        : 1,
        }
        duplicate_pairs.append(pair)

    # Non-duplicate pairs
    non_duplicate_pairs = []
    used_pairs = set()

    while len(non_duplicate_pairs) < N_NON_DUPLICATES:
        i, j = random.sample(range(len(df)), 2)
        pair_key = (min(i, j), max(i, j))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        record_a = df.iloc[i]
        record_b = df.iloc[j]

        # 70% same-country pairs (harder negatives)
        if record_a.get("ISO_COUNTRY_CODE") != record_b.get("ISO_COUNTRY_CODE"):
            if random.random() > 0.3:
                continue

        if compute_levenshtein(
            str(record_a.get("WIN_NAME", "")),
            str(record_b.get("WIN_NAME", ""))
        ) > 0.85:
            continue

        features = compute_all_features(record_a.to_dict(), record_b.to_dict())

        pair = {
            "pair_id"      : f"non_{len(non_duplicate_pairs)}",
            "type"         : "real_non_duplicate",
            "perturbations": "none",
            "CAE_NAME_1"   : record_a.get("CAE_NAME", ""),
            "WIN_NAME_1"   : record_a.get("WIN_NAME", ""),
            "VALUE_EURO_1" : record_a.get("VALUE_EURO", np.nan),
            "CPV_1"        : record_a.get("CPV", ""),
            "COUNTRY_1"    : record_a.get("ISO_COUNTRY_CODE", ""),
            "CAE_NAME_2"   : record_b.get("CAE_NAME", ""),
            "WIN_NAME_2"   : record_b.get("WIN_NAME", ""),
            "VALUE_EURO_2" : record_b.get("VALUE_EURO", np.nan),
            "CPV_2"        : record_b.get("CPV", ""),
            "COUNTRY_2"    : record_b.get("ISO_COUNTRY_CODE", ""),
            **features,
            "label"        : 0,
        }
        non_duplicate_pairs.append(pair)

    all_pairs = duplicate_pairs + non_duplicate_pairs
    random.shuffle(all_pairs)
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(OUTPUT_PAIRS, index=False)

    feature_columns = [
        "pair_id", "type", "perturbations",
        "cae_levenshtein", "cae_jaccard", "cae_jaro_winkler",
        "win_levenshtein", "win_jaccard", "win_jaro_winkler",
        "value_ratio", "cpv_match", "country_match", "label"
    ]
    gold_df = pairs_df[[c for c in feature_columns if c in pairs_df.columns]]
    gold_df.to_csv(OUTPUT_GOLD, index=False)

    print(f"Saved {OUTPUT_PAIRS} ({len(pairs_df):,} pairs)")
    print(f"Saved {OUTPUT_GOLD}")


if __name__ == "__main__":
    main()
