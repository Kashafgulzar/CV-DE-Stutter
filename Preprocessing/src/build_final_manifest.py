"""
Build a final manifest tsv (same column format as Common Voice's validated.tsv,
so it still has 'sentence'/transcript etc.), restricted to:
  - the top 109 speakers in csv_usable_speakers.csv
  - only clips that passed the duration + quality check (via clip_quality_report.csv)

This replaces the earlier train.tsv-based approach: speakers now come from
validated.tsv (the full pool), not the pre-existing train split.

Usage
-----
python build_final_manifest.py \
    --validated_tsv validated.tsv \
    --clip_quality_report clip_quality_report.csv \
    --usable_speakers_csv csv_usable_speakers.csv \
    --output final_manifest.tsv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated_tsv", required=True, help="Common Voice validated.tsv (has transcripts)")
    parser.add_argument("--clip_quality_report", required=True, help="Has: path, client_id, duration_sec, dnsmos, passed")
    parser.add_argument("--usable_speakers_csv", required=True,
                         help="csv_usable_speakers.csv with client_id, usable_hours, num_usable_clips, avg_dnsmos")
    parser.add_argument("--output", default="final_manifest.tsv")
    parser.add_argument("--expected_num_speakers", type=int, default=109)
    args = parser.parse_args()

    # --- top speakers ---
    usable_speakers = pd.read_csv(args.usable_speakers_csv)
    top_speaker_ids = set(usable_speakers["client_id"])
    print(f"Speakers in usable_speakers_csv: {len(top_speaker_ids)}")
    if len(top_speaker_ids) != args.expected_num_speakers:
        print(f"WARNING: expected {args.expected_num_speakers} speakers, found {len(top_speaker_ids)}")

    # --- transcripts / metadata ---
    validated = pd.read_csv(args.validated_tsv, sep="\t")
    validated.columns = [c.strip().lower() for c in validated.columns]
    assert "sentence" in validated.columns, "validated.tsv must have a 'sentence' (transcript) column"

    # --- quality pass/fail lookup ---
    quality = pd.read_csv(args.clip_quality_report)
    required_cols = {"path", "client_id", "duration_sec", "dnsmos", "passed"}
    assert required_cols.issubset(quality.columns), f"clip_quality_report.csv must have columns: {required_cols}"

    # --- join validated transcripts with quality results, on path ---
    merged = validated.merge(
        quality[["path", "duration_sec", "dnsmos", "passed"]],
        on="path", how="inner",  # inner: only clips that were actually scored
    )

    # --- restrict to top speakers + passed clips ---
    final = merged[merged["client_id"].isin(top_speaker_ids) & (merged["passed"] == True)].copy()  # noqa: E712

    final.to_csv(args.output, sep="\t", index=False)

    print(f"\nFinal manifest: {len(final)} clips, {final['client_id'].nunique()} speakers")
    print(f"Total duration: {final['duration_sec'].sum() / 3600:.2f}h")
    print(f"Written to: {args.output}")

    # sanity check: does every top speaker actually have clips in the final manifest?
    missing_speakers = top_speaker_ids - set(final["client_id"].unique())
    if missing_speakers:
        print(f"\nWARNING: {len(missing_speakers)} speakers from usable_speakers_csv have "
              f"no passing clips in validated.tsv (unexpected -- check join keys): {list(missing_speakers)[:5]}")


if __name__ == "__main__":
    main()