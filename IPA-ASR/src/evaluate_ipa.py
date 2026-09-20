"""
evaluate_ipa.py

Runs inference with a fine-tuned omniASR_CTC checkpoint over an eval manifest,
scores predictions against a chosen ground-truth transcript variant (fluent or
dysfluent), and writes a JSON file with
per-utterance audio path / ground truth / prediction / metrics, plus a
corpus-level summary.

Metrics computed (all via jiwer, i.e. standard Levenshtein edit distance):
  - PER (Phone Error Rate): edit distance over the phone-token sequence with
    '|' word-boundary markers stripped. The standard metric for phoneme ASR.
  - WER (Word Error Rate): tokens regrouped into words at '|' boundaries, each
    word treated as one unit.
  - CER (Character Error Rate): edit distance over the raw (detokenized) IPA
    string - catches errors at finer granularity than PER when tokens span
    multiple characters (e.g. 'kʲ' as a bound diacritic pair).
  - TER (Token Error Rate): PER but WITHOUT stripping '|' boundaries first -
    a stricter variant where getting a word boundary wrong also counts as an
    error. Included for completeness / comparison to PER.

Usage:
    pip install jiwer --break-system-packages

    # sanity check first
    python evaluate_ipa.py --manifest_dir path/to/datadir --split eval \
        --gt_wrd path/to/datadir/eval.wrd \
        --model_card cv_de_ipa_finetuned_ctc --out_json /dev/null --dry_run

    # test 1: fluent eval
    python evaluate_ipa.py \
        --manifest_dir path/to/datadir --split eval \
        --gt_wrd path/to/datadir/eval.wrd \
        --model_card cv_de_ipa_finetuned_ctc \
        --out_json results/eval.json --batch_size 16

    # test 2: same model, dysfluent ground truth
    python evaluate_ipa.py \
        --manifest_dir path/to/datadir --split eval \
        --gt_wrd path/to/dysfluent-datadir/eval.wrd \
        --model_card cv_de_ipa_finetuned_ctc \
        --out_json results/eval_dysfluent.json --batch_size 16
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import re
import jiwer

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("evaluate_ipa")


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def read_tsv(path):
    """First line is the audio root dir, rest are `relpath\tnum_frames`."""
    lines = read_lines(path)
    audio_root = Path(lines[0])
    entries = []
    for line in lines[1:]:
        rel_path, length = line.split("\t")
        entries.append((audio_root / rel_path, int(length)))
    return entries


def strip_boundaries(tok_str: str) -> str:
    tokens = [t for t in tok_str.split() if t != "|"]
    return " ".join(tokens)


def regroup_into_words(tok_str: str) -> str:
    """'a b | c d |' -> 'ab cd' (also serves as the natural detokenized IPA string)."""
    words, current = [], []
    for t in tok_str.split():
        if t == "|":
            if current:
                words.append("".join(current))
                current = []
        else:
            current.append(t)
    if current:
        words.append("".join(current))
    return " ".join(words)


def compute_metrics(ref_tok: str, hyp_tok: str) -> dict:
    ref_tok, hyp_tok = ref_tok.strip(), hyp_tok.strip()

    ter = jiwer.wer(ref_tok, hyp_tok) if ref_tok else float("nan")

    ref_per, hyp_per = strip_boundaries(ref_tok), strip_boundaries(hyp_tok)
    per = jiwer.wer(ref_per, hyp_per) if ref_per else float("nan")

    ref_words, hyp_words = regroup_into_words(ref_tok), regroup_into_words(hyp_tok)
    wer = jiwer.wer(ref_words, hyp_words) if ref_words else float("nan")

    cer = jiwer.cer(ref_words, hyp_words) if ref_words else float("nan")

    return {"ter": ter, "per": per, "wer": wer, "cer": cer}

# --- Clean extra dysfluency markers not present in the fluent tokenizer ---
def clean_gt_line(line: str) -> str:
    # 1. Remove pause tokens (...) and (..)
    line = re.sub(r'\(\.\.\.\)', '', line)
    line = re.sub(r'\(\.\.\)', '', line)
    # 2. Fix double/duplicate pipes caused by removing standalone pauses (|  | -> |)
    line = re.sub(r'\|\s*\|', '|', line)  
    # 3. Collapse multiple spaces into a single space
    line = re.sub(r'\s+', ' ', line).strip() 
    # 4. Clean up any leading or trailing whitespace inside pipes
    line = re.sub(r'\s*\|\s*', ' | ', line).strip()
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", required=True, help="Dir with {split}.tsv (audio list)")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--gt_wrd", required=True,
                     help="Path to the .wrd file with ground-truth tokenized IPA to score "
                          "against - fluent or dysfluent - must be in the same line order "
                          "as {split}.tsv")
    ap.add_argument("--model_card", required=True,
                     help="Registered fairseq2 asset card name for the fine-tuned checkpoint")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true",
                     help="Only run inference on the first 3 utterances and print raw "
                          "predictions next to ground truth, then exit - use this to sanity "
                          "check the tokenizer format assumption before a full run")
    args = ap.parse_args()

    manifest_dir = Path(args.manifest_dir)
    tsv_path = manifest_dir / f"{args.split}.tsv"

    log.info(f"Reading manifest: {tsv_path}")
    entries = read_tsv(tsv_path)
    audio_paths = [str(p) for p, _ in entries]

    log.info(f"Reading ground truth: {args.gt_wrd}")
    gt_lines = read_lines(args.gt_wrd)
    assert len(gt_lines) == len(audio_paths), (
        f"Mismatch: {len(gt_lines)} ground-truth lines vs {len(audio_paths)} audio entries. "
        "Did you point --gt_wrd at a .wrd file built from the same split/order?"
    )

    # --- Clean extra dysfluency markers not present in the fluent tokenizer ---
    # gt_lines = [clean_gt_line(line) for line in gt_lines]
    # --------------------------------------------------------------------------

    log.info(f"Loading model card: {args.model_card}")
    pipeline = ASRInferencePipeline(model_card=args.model_card)

    if args.dry_run:
        log.info("DRY RUN: inferring first 3 utterances only")
        preds = pipeline.transcribe(audio_paths[:3], batch_size=3)
        for ap_, gt, pred in zip(audio_paths[:3], gt_lines[:3], preds):
            print(f"\naudio: {ap_}")
            print(f"  ground truth (.wrd): {gt!r}")
            print(f"  prediction (raw)   : {pred!r}")
        print("\nCompare the ground truth and prediction formatting above - if they don't "
              "look like the same tokenization scheme, fix compute_metrics()/the ref/hyp "
              "handling before running the full eval.")
        return

    log.info(f"Running inference on {len(audio_paths)} utterances (batch_size={args.batch_size})...")
    t0 = time.time()
    predictions = []
    for i in range(0, len(audio_paths), args.batch_size):
        batch = audio_paths[i:i + args.batch_size]
        preds = pipeline.transcribe(batch, batch_size=len(batch))
        predictions.extend(preds)
        if (i // args.batch_size) % 10 == 0:
            log.info(f"  {i + len(batch)}/{len(audio_paths)} done ({time.time() - t0:.1f}s elapsed)")
    log.info(f"Inference finished in {time.time() - t0:.1f}s")

    log.info("Computing metrics...")
    records = []
    for (audio_path, _), gt_tok, pred_tok in zip(entries, gt_lines, predictions):
        m = compute_metrics(gt_tok, pred_tok)
        records.append({
            "audio_path": str(audio_path),
            "ground_truth_ipa": regroup_into_words(gt_tok),
            "predicted_ipa": regroup_into_words(pred_tok),
            "ground_truth_tokens": gt_tok,
            "predicted_tokens": pred_tok,
            "ter": m["ter"],
            "per": m["per"],
            "wer": m["wer"],
            "cer": m["cer"],
        })

    corpus_per = jiwer.wer([strip_boundaries(g) for g in gt_lines],
                            [strip_boundaries(p) for p in predictions])
    corpus_wer = jiwer.wer([regroup_into_words(g) for g in gt_lines],
                            [regroup_into_words(p) for p in predictions])
    corpus_cer = jiwer.cer([regroup_into_words(g) for g in gt_lines],
                            [regroup_into_words(p) for p in predictions])

    summary = {
        "num_utterances": len(records),
        "corpus_per": corpus_per,
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "mean_utterance_per": sum(r["per"] for r in records) / len(records),
        "mean_utterance_wer": sum(r["wer"] for r in records) / len(records),
        "mean_utterance_cer": sum(r["cer"] for r in records) / len(records),
    }

    log.info(f"Corpus-level PER: {corpus_per:.4f}  WER: {corpus_wer:.4f}  CER: {corpus_cer:.4f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "utterances": records}, f, ensure_ascii=False, indent=2)

    log.info(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
