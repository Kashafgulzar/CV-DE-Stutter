"""
build_tokenizer.py

Builds a SentencePiece tokenizer over your IPA phone vocabulary in "word" mode,
so each whitespace-separated unit in your .wrd files (a phone symbol or the '|'
boundary marker) becomes exactly one vocab piece - no subword merging, no BPE.

IMPORTANT: verify the round-trip check at the end before trusting this in a full
training run. char_tokenizer's exact handling of pre-tokenized text vs. raw text
is not something I could fully confirm from the docs - if the decoded round-trip
doesn't match your input, stop and inspect before training.

Usage:
    pip install sentencepiece --break-system-packages

    python build_tokenizer.py \
        --wrd_files path/to/fluent/train.wrd path/to/fluent/eval.wrd \
        --out_dir path/to/Tokenizers \
        --vocab_size 100
"""
import argparse
from pathlib import Path

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrd_files", nargs="+", required=True,
                     help="All .wrd files whose vocab should be covered (train+eval, "
                          "so eval never has an unseen symbol)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=100,
                     help="Set generously above your actual unique-token count; "
                          "SentencePiece will use however many distinct pieces exist "
                          "in the corpus, up to this cap.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Concatenate the .wrd files into one training corpus for spm_train.
    corpus_path = out_dir / "_spm_train_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as out_f:
        for wf in args.wrd_files:
            with open(wf, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if line:
                        out_f.write(line + "\n")

    model_prefix = str(out_dir / "ipa_tokenizer")

    # model_type=word: treat each whitespace-separated unit (a phone, or '|') as an
    # atomic vocab piece. Do NOT use 'bpe' or 'unigram' here - those would merge or
    # split your phone tokens instead of preserving them exactly.
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type="word",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # user_defined_symbols=["|"],
    )

    model_path = f"{model_prefix}.model"
    print(f"\nTokenizer written to: {model_path}")

    # --- Round-trip verification ---
    sp = spm.SentencePieceProcessor(model_file=model_path)
    print(f"Vocab size: {sp.get_piece_size()}")

    print("\n--- Round-trip check on first 3 corpus lines ---")
    with open(corpus_path, "r", encoding="utf-8") as f:
        sample_lines = [next(f).strip() for _ in range(10000)]
    # with open(corpus_path, "r", encoding="utf-8") as f:
    #     sample_lines = [line.strip() for line in f]

    all_ok = True
    for line in sample_lines:
        ids = sp.encode(line, out_type=int)
        decoded = sp.decode(ids)
        pieces = sp.encode(line, out_type=str)
        # match = (decoded.replace(" ", "") == line.replace(" ", "").replace("|", ""))
        match = (decoded.replace(" ", "").replace("|", "") == line.replace(" ", "").replace("|", ""))
        print(f"  input : {line}")
        print(f"  pieces: {pieces}")
        print(f"  decoded: {decoded}")
        print(f"  match (content, ignoring boundary formatting): {match}\n")
        all_ok = all_ok and match

    if not all_ok:
        print("!!! WARNING: round-trip mismatch detected. Do NOT proceed to a full "
              "training run until you understand why - inspect the pieces/decoded "
              "output above against what AsrTask expects to encode/decode with this "
              "tokenizer_family (char_tokenizer). !!!")
    else:
        print("Round-trip looks consistent. Proceed to register the asset card.")

    corpus_path.unlink()  # cleanup temp file


if __name__ == "__main__":
    main()
