#!/usr/bin/env python3
"""
generate_dysfluent_ipa.py

Processes JSONL in batches with SQLite checkpointing.
Stops automatically when daily rate limit is reached.
Can be stopped and resumed at any time.

Usage:
    export GEMINI_API_KEY="your-key"
    python generate_dysfluent_ipa.py input.jsonl output.jsonl --prompt prompt.txt

Resume after crash/stop:
    python generate_dysfluent_ipa.py input.jsonl output.jsonl --prompt prompt.txt
    (automatically skips already-processed records)
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from google import genai

# ============================================================================
# CONFIG
# ============================================================================

TRANSCRIPT_FIELD = "ipa_correct"
BATCH_SIZE = 20
MAX_RETRIES = 5
RPM_LIMIT = 30
SLEEP_BETWEEN_CALLS = 60 / RPM_LIMIT + 0.5

# Gemini 3.1 Flash-Lite free tier:
# Set conservative default to avoid hard ban
DEFAULT_DAILY_LIMIT = 500

# ============================================================================
# PHONEME VOCAB (your exact list)
# ============================================================================

MULTI_PHONEMES = [
    'aɪ', 'aɪə', 'aʊ', 'dʑ', 'dʒ', 'dʲ', 'eə', 'eɪ', 'eː',
    'iə', 'iː', 'l̩', 'mʲ', 'n̩', 'oː', 'pf', 'ts', 'tɕ',
    'tʃ', 'tʲ', 'uː', 'yː', 'øː', 'ɑː', 'ɑ̃', 'ɔø', 'ɔɪ',
    'ɔː', 'əl', 'əʊ', 'ɛɪ', 'ɛː', 'ɜː', 'ɡʲ', 'ʊɐ',
    'ˈa', 'ˈaɪ', 'ˈaɪə', 'ˈaʊ', 'ˈaː', 'ˈeə', 'ˈeɪ', 'ˈeː',
    'ˈi', 'ˈiə', 'ˈiː', 'ˈn̩', 'ˈo', 'ˈoː', 'ˈu', 'ˈuː',
    'ˈyː', 'ˈøː', 'ˈœ', 'ˈœ̃', 'ˈɑ', 'ˈɑː', 'ˈɑ̃', 'ˈɒ',
    'ˈɔ', 'ˈɔø', 'ˈɔɪ', 'ˈɔː', 'ˈɔ̃', 'ˈə', 'ˈəʊ', 'ˈɛ',
    'ˈɛɪ', 'ˈɛː', 'ˈɜ', 'ˈɜː', 'ˈɨ', 'ˈɪ', 'ˈʊ', 'ˈʊɐ',
    'ˈʊə', 'ˈʌ', 'ˈʏ',
    'ˌa', 'ˌaɪ', 'ˌaʊ', 'ˌeə', 'ˌeɪ', 'ˌeː', 'ˌi', 'ˌiə',
    'ˌiː', 'ˌo', 'ˌoː', 'ˌu', 'ˌuː', 'ˌyː', 'ˌøː', 'ˌœ',
    'ˌɑ', 'ˌɑː', 'ˌɑ̃', 'ˌɒ', 'ˌɔ', 'ˌɔø', 'ˌɔː', 'ˌɔ̃',
    'ˌə', 'ˌəʊ', 'ˌɛ', 'ˌɛɪ', 'ˌɛː', 'ˌɜː', 'ˌɪ', 'ˌʊ',
    'ˌʊɐ', 'ˌʌ', 'ˌʏ',
]

MULTI_PHONEMES_SORTED = sorted(MULTI_PHONEMES, key=len, reverse=True)
SINGLE_PHONEMES = set('abdefhijklmnoprstuvwxzçðŋœɐɑɒɔɕəɛɜɡɨɪɲɹɾʃʊʌʏʒʔθː')
PHONEME_SET = set(MULTI_PHONEMES) | SINGLE_PHONEMES

# VOWELS must include diphthongs/triphthongs from MULTI_PHONEMES
VOWELS = set('aɛeɪiɔoʊuʏyøœəɐɑɒɔɜʌɨ')
for v in list(VOWELS):
    VOWELS.add(v + 'ː')

VOWEL_BASE_CHARS = set('aɛeɪiɔoʊuʏyøœəɐɑɒɔɜʌɨ')
for p in MULTI_PHONEMES:
    bare_p = p.replace('ˈ', '').replace('ˌ', '')
    if len(bare_p) > 1 and all(c in VOWEL_BASE_CHARS for c in bare_p):
        VOWELS.add(p)
        VOWELS.add(bare_p)

ELIGIBLE_PRO = VOWELS | {'m', 'n', 's', 'l', 'r', 'ŋ', 'ɲ', 'ɾ', 'ɹ'}
CLUSTERS = {'ʃt', 'ʃp', 'ʃk', 'ʃm', 'ʃn', 'ʃl', 'ʃr', 'bl', 'br', 'dr', 'fl', 'fr', 'gl', 'gr', 'kl', 'kr', 'pl', 'pr', 'tr', 'kn', 'gn', 'kv', 'ps', 'ks'}
FILLERS = ['ˈɛː', 'ˈɛːm', 'hˈəm']


# ============================================================================
# TOKENIZER & BUILDERS
# ============================================================================

def tokenize_ipa(word):
    tokens = []
    i = 0
    n = len(word)
    while i < n:
        matched = False
        for sym in MULTI_PHONEMES_SORTED:
            if word[i:].startswith(sym):
                tokens.append(sym)
                i += len(sym)
                matched = True
                break
        if matched:
            continue
        if word[i] in SINGLE_PHONEMES:
            tokens.append(word[i])
            i += 1
            continue
        tokens.append(word[i])
        i += 1
    return tokens


def bare(token):
    return token.replace('ˈ', '').replace('ˌ', '')


def is_phoneme_token(token):
    return token in PHONEME_SET


def build_srep(word, prefix1_end=None, prefix2_end=None):
    toks = tokenize_ipa(word)
    phonemes = [t for t in toks if is_phoneme_token(t)]
    n = len(phonemes)
    if n < 3:
        return word
    p1_end = prefix1_end if prefix1_end is not None else max(0, n // 3 - 1)
    p2_end = prefix2_end if prefix2_end is not None else max(p1_end + 1, 2 * n // 3 - 1)
    p2_end = min(p2_end, n - 2)
    p1_end = max(0, min(p1_end, p2_end - 1))
    prefix1 = ''.join(phonemes[:p1_end + 1])
    prefix2 = ''.join(phonemes[:p2_end + 1])
    return f"{prefix1} [sREP] {prefix2} [sREP] {word}"


def build_pro(word, split_after_phoneme=None):
    toks = tokenize_ipa(word)
    phonemes = [t for t in toks if is_phoneme_token(t)]

    if split_after_phoneme is not None:
        split_idx = split_after_phoneme
        if not (0 <= split_idx < len(phonemes) - 1):
            return word
    else:
        split_idx = -1
        for i, ph in enumerate(phonemes):
            if bare(ph) in ELIGIBLE_PRO:
                split_idx = i
                break
        if split_idx < 0:
            return word

    left, right = [], []
    ph_count = 0
    for t in toks:
        if is_phoneme_token(t):
            target = left if ph_count <= split_idx else right
            ph_count += 1
        else:
            target = left if ph_count <= split_idx else right
        target.append(t)
    return f"{''.join(left)} [PRO] {''.join(right)}"


def build_pau(word, split_after_phoneme=None):
    toks = tokenize_ipa(word)
    phonemes = [t for t in toks if is_phoneme_token(t)]
    if len(phonemes) < 2:
        return word

    split_idx = -1
    if split_after_phoneme is not None:
        if 0 <= split_after_phoneme < len(phonemes) - 1:
            split_idx = split_after_phoneme + 1  # convert to PAU internal semantics
        # else leave split_idx = -1 to trigger heuristic fallback

    if split_idx == -1:
        for i in range(1, len(phonemes)):
            if ('ˈ' in phonemes[i] or 'ˌ' in phonemes[i]) and bare(phonemes[i]) in VOWELS:
                split_idx = i
                break
        else:
            for i in range(1, len(phonemes) - 1):
                pair = bare(phonemes[i]) + bare(phonemes[i + 1])
                if pair in CLUSTERS:
                    split_idx = i
                    break
            else:
                split_idx = len(phonemes) // 2

    left, right = [], []
    ph_count = 0
    for t in toks:
        if is_phoneme_token(t):
            target = left if ph_count < split_idx else right
            ph_count += 1
        else:
            target = left if ph_count < split_idx else right
        target.append(t)
    return f"{''.join(left)} [PAU] {''.join(right)}"


def build_int(insert_before_word):
    return f"{FILLERS[insert_before_word % 3]} [INT]"


def build_wrep(words, indices):
    first = ' '.join([f"{words[i]} [wREP]" for i in indices])
    second = ' '.join([words[i] for i in indices])
    return f"{first} {second}"


# ============================================================================
# POST-PROCESSOR
# ============================================================================

def validate_and_apply(ipa_line, decisions):
    # print(decisions)
    # print(ipa_line)
    words = ipa_line.split()
    word_count = len(words)
    used = set()
    valid_decisions = []

    # --- Validation ---
    for dec in decisions:
        t = dec.get('type')
        valid = False
        if t == 'INT':
            ibw = dec.get('insert_before_word', -1)
            if 0 <= ibw <= word_count:
                valid = True
        elif t == 'SB':
            a, b = dec.get('between', [None, None])
            if a is not None and b is not None:
                if 0 <= a < word_count and b == a + 1 and b < word_count:
                    if a not in used and b not in used:
                        valid = True
                        used.update([a, b])
        elif t == 'wREP':
            idxs = dec.get('word_indices', [])
            if idxs and all(0 <= i < word_count for i in idxs):
                if all(i == idxs[0] + j for j, i in enumerate(idxs)):
                    if not any(i in used for i in idxs):
                        valid = True
                        used.update(idxs)
        # FIX 2: enforce minimum phoneme counts and validate LLM indices
        elif t == 'sREP':
            wi = dec.get('word_index', -1)
            if 0 <= wi < word_count and wi not in used:
                phs = [t for t in tokenize_ipa(words[wi]) if is_phoneme_token(t)]
                p1 = dec.get('prefix1_end_phoneme')
                p2 = dec.get('prefix2_end_phoneme')
                if len(phs) >= 3:
                    ok = True
                    if p1 is not None and not (0 <= p1 < len(phs)):
                        ok = False
                    if p2 is not None and not (p1 is not None and p1 < p2 < len(phs)):
                        ok = False
                    if ok:
                        valid = True
                        used.add(wi)
        elif t == 'PAU':
            wi = dec.get('word_index', -1)
            if 0 <= wi < word_count and wi not in used:
                phs = [t for t in tokenize_ipa(words[wi]) if is_phoneme_token(t)]
                sp = dec.get('split_after_phoneme')
                if len(phs) >= 2:
                    if sp is None or 0 <= sp < len(phs) - 1:
                        valid = True
                        used.add(wi)
        elif t == 'PRO':
            wi = dec.get('word_index', -1)
            if 0 <= wi < word_count and wi not in used:
                phs = [t for t in tokenize_ipa(words[wi]) if is_phoneme_token(t)]
                sp = dec.get('split_after_phoneme')
                if sp is None or 0 <= sp < len(phs) - 1:
                    valid = True
                    used.add(wi)
        if valid:
            valid_decisions.append(dec)

    # --- Build lookups by ORIGINAL word index ---
    int_by_pos = {}  # insert_before_word -> INT decision
    mod_by_word = {}  # word_index -> modification decision

    for dec in valid_decisions:
        t = dec['type']
        if t == 'INT':
            pos = dec['insert_before_word']
            int_by_pos[pos] = dec
        elif t == 'SB':
            a, b = dec['between']
            mod_by_word[a] = dec  # anchored at first word (a)
        elif t == 'wREP':
            idx0 = dec['word_indices'][0]
            mod_by_word[idx0] = dec
        elif t in ('sREP', 'PRO', 'PAU'):
            mod_by_word[dec['word_index']] = dec

    # --- Single left-to-right pass over ORIGINAL word indices ---
    out_parts = []
    w_idx = 0

    while w_idx <= word_count:
        # Insert any INT that belongs before original word w_idx
        if w_idx in int_by_pos:
            out_parts.append(build_int(w_idx))

        if w_idx >= word_count:
            break

        # Apply modification for original word w_idx (if any)
        if w_idx in mod_by_word:
            dec = mod_by_word[w_idx]
            t = dec['type']
            if t == 'SB':
                a, b = dec['between']
                out_parts.append(words[a])  # keep left word
                out_parts.append('[SB]')  # insert marker
                out_parts.append(words[b])  # keep right word
                w_idx += 2  # skip both (they're consumed)
            elif t == 'wREP':
                out_parts.append(build_wrep(words, dec['word_indices']))
                w_idx = dec['word_indices'][-1] + 1
            # FIX 3: pass LLM indices to builders
            elif t == 'sREP':
                out_parts.append(build_srep(words[w_idx],
                                            dec.get('prefix1_end_phoneme'),
                                            dec.get('prefix2_end_phoneme')))
                w_idx += 1
            elif t == 'PRO':
                out_parts.append(build_pro(words[w_idx],
                                           dec.get('split_after_phoneme')))
                w_idx += 1
            elif t == 'PAU':
                out_parts.append(build_pau(words[w_idx],
                                           dec.get('split_after_phoneme')))
                w_idx += 1
        else:
            # No modification — output original word unchanged
            out_parts.append(words[w_idx])
            w_idx += 1

    return ' '.join(out_parts)



# ============================================================================
# CHECKPOINT DATABASE
# ============================================================================

class CheckpointDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processed (
                global_idx INTEGER PRIMARY KEY,
                output_json TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_date TEXT NOT NULL,
                call_hour INTEGER NOT NULL,
                call_count INTEGER DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.conn.commit()

    def is_processed(self, global_idx: int) -> bool:
        cur = self.conn.execute("SELECT 1 FROM processed WHERE global_idx = ?", (global_idx,))
        return cur.fetchone() is not None

    def get_processed_count(self) -> int:
        cur = self.conn.execute("SELECT COUNT(*) FROM processed")
        return cur.fetchone()[0]

    def save_batch(self, results: list):
        for global_idx, output_json in results:
            self.conn.execute(
                "INSERT OR REPLACE INTO processed (global_idx, output_json) VALUES (?, ?)",
                (global_idx, output_json)
            )
        self.conn.commit()

    def get_all_outputs(self):
        cur = self.conn.execute("SELECT output_json FROM processed ORDER BY global_idx")
        return [row[0] for row in cur]

    def log_api_call(self):
        """Log one API call for daily rate limit tracking."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        hour = now.hour
        self.conn.execute(
            """INSERT INTO api_calls (call_date, call_hour, call_count)
               VALUES (?, ?, 1)
               ON CONFLICT DO UPDATE SET call_count = call_count + 1""",
            (date_str, hour)
        )
        self.conn.commit()

    def get_today_call_count(self) -> int:
        """Get total API calls made today."""
        today = datetime.now().strftime("%Y-%m-%d")
        cur = self.conn.execute(
            "SELECT SUM(call_count) FROM api_calls WHERE call_date = ?",
            (today,)
        )
        result = cur.fetchone()[0]
        return result or 0

    def close(self):
        self.conn.close()


# ============================================================================
# GEMINI API
# ============================================================================

def call_gemini(system_prompt: str, user_content: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=user_content,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            seed=1234,
        ),
    )
    return response.text


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def clean_transcript(ipa: str) -> str:
    return " ".join(ipa.split())


def format_batch(records: list) -> str:
    lines = []
    for pos, (global_idx, record) in enumerate(records, start=1):
        ipa = clean_transcript(record[TRANSCRIPT_FIELD])
        lines.append(f"{global_idx}. Transcript (IPA): {ipa}")
    return "\n".join(lines)


def parse_llm_output(text: str) -> list:
    text = text.replace("```json", "").replace("```", "").strip()
    decisions = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            decisions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return decisions


def process_batch(records: list, system_prompt: str, api_key: str) -> list:
    batch_text = format_batch(records)
    # print(batch_text)
    for attempt in range(MAX_RETRIES):
        try:
            raw_output = call_gemini(system_prompt, batch_text, api_key)
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = 2 ** attempt
                print(f"  Rate limited, retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

    decision_lines = parse_llm_output(raw_output)

    if len(decision_lines) != len(records):
        print(f"  WARNING: Expected {len(records)} decisions, got {len(decision_lines)}", file=sys.stderr)

    results = []
    for (global_idx, record), dec in zip(records, decision_lines):
        ipa = clean_transcript(record[TRANSCRIPT_FIELD])
        # print(ipa)
        dysfluent = validate_and_apply(ipa, dec.get('decisions', []))
        # print(dysfluent)
        output_record = dict(record)
        output_record['ipa_dysfluent'] = dysfluent
        results.append((global_idx, json.dumps(output_record, ensure_ascii=False)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", help="Input JSONL with ipa_correct field")
    parser.add_argument("output_jsonl", help="Output JSONL with ipa_dysfluent added")
    parser.add_argument("--prompt", default="prompt.txt", help="System prompt file")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--field", default="ipa_correct", help="IPA field name")
    parser.add_argument("--checkpoint", default="checkpoint.db", help="SQLite checkpoint DB")
    parser.add_argument("--daily-limit", type=int, default=DEFAULT_DAILY_LIMIT,
                        help="Max API calls per day (default: 1400)")
    parser.add_argument("--resume-only", action="store_true", help="Only write output from checkpoint")
    args = parser.parse_args()

    global TRANSCRIPT_FIELD
    TRANSCRIPT_FIELD = args.field
    api_key = "your_api_key"

    with open(args.prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    db = CheckpointDB(args.checkpoint)
    already_done = db.get_processed_count()
    print(f"Checkpoint: {already_done} records already processed")

    if args.resume_only:
        print(f"Writing output from checkpoint to {args.output_jsonl}...")
        with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
            for line in db.get_all_outputs():
                out_f.write(line + "\n")
        db.close()
        print("Done!")
        return

    all_records = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        # JSON array format
        data = json.loads(raw)
        for i, record in enumerate(data):
            all_records.append((i, record))

    total = len(all_records)
    remaining = [r for r in all_records if not db.is_processed(r[0])]
    print(f"Total: {total} | Done: {already_done} | Remaining: {len(remaining)}")

    try:
        batch_count = 0
        for start in range(0, len(remaining), args.batch_size):
            today_calls = db.get_today_call_count()
            if today_calls >= args.daily_limit:
                print(f"\n*** DAILY LIMIT REACHED ***")
                break

            batch = remaining[start:start + args.batch_size]
            if len(batch) < args.batch_size:
                print(f"  Skipping final incomplete batch ({len(batch)} records)")
                break

            global_start = batch[0][0]
            global_end = batch[-1][0]
            batch_count += 1
            print(f"  Batch {batch_count} (records {global_start}-{global_end}) | API calls today: {today_calls + 1}/{args.daily_limit}")

            results = process_batch(batch, system_prompt, api_key)
            db.save_batch(results)
            db.log_api_call()

            # WRITE OUTPUT AFTER EVERY BATCH
            with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
                for line in db.get_all_outputs():
                    out_f.write(line + "\n")

            done = db.get_processed_count()
            pct = done / total * 100
            remaining_batches = (total - done) // args.batch_size
            eta_hours = remaining_batches * SLEEP_BETWEEN_CALLS / 3600
            print(f"    Progress: {done}/{total} ({pct:.1f}%) | ETA: {eta_hours:.1f}h")

            time.sleep(SLEEP_BETWEEN_CALLS)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
    finally:
        # SAFETY NET: always write latest state before exiting
        print(f"Writing final output to {args.output_jsonl}...")
        with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
            for line in db.get_all_outputs():
                out_f.write(line + "\n")
        db.close()
        print(f"Done! Processed {db.get_processed_count()} records total.")


if __name__ == "__main__":
    main()