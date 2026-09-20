"""
Convert German reference_text to espeak IPA phonemes and preserve all original JSONL fields.
"""

import json
from pathlib import Path
from tqdm import tqdm
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_FILE = Path("path_to_original_file.jsonl")  # Update to your input file path if different
OUTPUT_FILE = Path("path_to_output_file.jsonl")
LANGUAGE = "de"
BATCH_SIZE = 256


def load_jsonl(file_path: Path):
    """Load JSONL data into a list of dictionaries."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")

    print(f"Loaded {len(records)} entries from {file_path}")
    return records


def add_espeak_ipa(records: list) -> list:
    """Generate ipa_espeak for each record based on reference_text."""
    phonemizer = EspeakBackend(
        language=LANGUAGE,
        preserve_punctuation=False,
        with_stress=True,
        tie=False,
        language_switch="remove-flags",
    )
    separator = Separator(word=" ")

    texts = [record.get("reference_text", "") for record in records]
    phonemized_results = []
    failed_count = 0

    # Process in batches for better performance
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Phonemizing"):
        batch_texts = texts[i: i + BATCH_SIZE]
        try:
            results = phonemizer.phonemize(
                batch_texts, separator=separator, strip=True
            )
            phonemized_results.extend(results)
        except Exception as e:
            # Fallback: process item by item if batch processing fails
            for text in batch_texts:
                try:
                    res = phonemizer.phonemize([text], separator=separator, strip=True)
                    phonemized_results.append(res[0] if res else "")
                except Exception:
                    phonemized_results.append("")
                    failed_count += 1

    # Attach phonemes back to original records
    for record, ipa in zip(records, phonemized_results):
        record["ipa_espeak"] = ipa

    if failed_count > 0:
        print(f"Warning: Failed phonemizations on {failed_count} entries.")

    return records


def save_jsonl(records: list, file_path: Path):
    """Save updated records to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(records)} entries to {file_path}")


def main():
    records = load_jsonl(INPUT_FILE)
    updated_records = add_espeak_ipa(records)
    save_jsonl(updated_records, OUTPUT_FILE)


if __name__ == "__main__":
    main()