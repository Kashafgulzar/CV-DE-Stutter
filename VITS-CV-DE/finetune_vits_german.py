"""
Fine-tune a pretrained multi-speaker VITS-VCTK checkpoint (109 English
speakers) on German Common Voice data, by:

  1. Replacing the speaker embedding table (emb_g) entirely -- the English
     VCTK speaker identities are meaningless for German speakers, so these
     slots are re-initialized (mean-of-old-speakers + noise) and learned
     from scratch. The number of speakers is taken from the data, not
     hardcoded.

  2. Extending (not replacing) the phoneme/text embedding table -- German
     and English share a large chunk of IPA symbols (via espeak-ng), so
     overlapping symbols keep their pretrained embeddings, and only
     German-specific symbols not seen in English get fresh rows. The full
     pretrained VitsCharacters vocabulary (pad + punctuation + graphemes +
     phonemes + blank) is used as the "old" vocabulary -- not just
     config.characters.characters, which is only the English letters.

  3. Freezing the parts of the network that encode general "how to produce
     natural human speech waveforms" knowledge -- the posterior encoder and
     the decoder (HiFi-GAN-style generator) operate on spectrogram/waveform
     representations that are largely language-agnostic. Freezing them
     preserves audio-quality benefits from the English pretraining. This is
     recorded via config.model_args.freeze_* flags (applied every epoch),
     not just a one-off requires_grad pass.

  4. Leaving the text encoder, flow module, duration predictor, and the new
     speaker + extended text embeddings trainable -- these need to adapt to
     German phonotactics, prosody, and the new speaker identities.

Training data:
  - final_manifest.tsv (output of build_final_manifest.py): top speakers by
    usable_hours, passed clips, with transcripts.
  - Since Common Voice train/dev/test are speaker-disjoint, eval coverage
    for all speakers cannot come from the official splits. Instead we carve
    a per-speaker held-out set directly out of final_manifest.tsv.

Requires: pip install TTS  (Coqui-TTS)

Usage
-----
python finetune_vits_german.py \
    --pretrained_checkpoint /path/to/vits_vctk/model.pth \
    --pretrained_config /path/to/vits_vctk/config.json \
    --final_manifest_tsv build_final_manifest_outputs/final_manifest.tsv \
    --clips_dir /path/to/cv-corpus/de/clips \
    --output_path ./vits_de_finetune \
    --language de \
    --german_symbols_json german_symbols_full_dataset.json \
    --freeze_decoder --freeze_posterior_encoder \
    --epochs 50 --batch_size 16

Notes
-----
--german_symbols_json should be produced by running the FULL train+eval
(+ ideally test/inference) text through the exact cleaner + espeak-ng
phonemization pipeline you will train with (same cleaner, same backend,
same separator="", same Unicode normalization). If omitted, this script
falls back to phonemizing the *entire* training set here (not a 500-row
sample) using the same cleaner/backend configured below, but a precomputed
file generated the same way you'll do inference is strongly preferred so
train and inference environments can't drift apart.
"""
import os
import argparse
import json
from pathlib import Path
from types import MethodType
import csv

import pandas as pd
import torch
import torch.nn as nn
import time

from TTS.config import load_config, BaseDatasetConfig
from TTS.tts.models.vits import Vits, VitsCharacters
from TTS.tts.utils.text.phonemizers import ESpeak
from trainer import Trainer, TrainerArgs
from TTS.tts.datasets import load_tts_samples
import soundfile as sf
import TTS.tts.models.vits as _vits_mod

def _load_audio_soundfile(file_path):
    """Bypasses torchaudio->torchcodec->FFmpeg, which has unresolvable
    library-version conflicts"""
    data, sr = sf.read(file_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data).unsqueeze(0)
    return wav, sr

_vits_mod.load_audio = _load_audio_soundfile


# ---------------------------------------------------------------------------
# 1) Build manifests: N-speaker train set + speaker-matched eval set
# ---------------------------------------------------------------------------

def build_speaker_mapping(final_manifest_tsv: str) -> dict:
    """
    Maps German client_id -> local speaker index 0..N-1.

    NOTE: the number of speakers no longer needs to match the pretrained
    VCTK table size (109) -- the speaker embedding table is discarded and
    rebuilt from scratch, so it can have any number of rows. Only the
    embedding *dimension* has to match, since the frozen posterior
    encoder/decoder/flow expect that dimension.
    """
    df = pd.read_csv(final_manifest_tsv, sep="\t")
    assert "client_id" in df.columns, "final_manifest_tsv must have a client_id column"
    speaker_ids = sorted(df["client_id"].unique())
    print(f"Found {len(speaker_ids)} speakers in {final_manifest_tsv}")
    return {sid: idx for idx, sid in enumerate(speaker_ids)}


def per_speaker_holdout_split(final_manifest_tsv: str, held_out_frac: float = 0.1, seed: int = 0):
    """
    Carve a held-out set directly out of final_manifest.tsv: for each
    speaker, randomly hold out `held_out_frac` of their utterances for
    eval, keep the rest for train. Guarantees every speaker has both train
    and eval coverage despite train/dev/test being speaker-disjoint
    upstream.
    """
    df = pd.read_csv(final_manifest_tsv, sep="\t")

    train_parts, eval_parts = [], []
    for client_id, group in df.groupby("client_id"):
        if len(group) < 2:
            raise ValueError(
                f"Speaker {client_id} has only {len(group)} clip(s) -- "
                f"cannot carve a train/eval holdout split. Drop this speaker "
                f"upstream or lower the per-speaker inclusion threshold."
            )
        group = group.sample(frac=1.0, random_state=seed)  # shuffle per speaker
        n_eval = max(1, int(round(len(group) * held_out_frac)))  # at least 1 held-out clip/speaker
        eval_parts.append(group.iloc[:n_eval])
        train_parts.append(group.iloc[n_eval:])

    train_df = pd.concat(train_parts, ignore_index=True)
    eval_df = pd.concat(eval_parts, ignore_index=True)

    print(f"Per-speaker holdout split ({held_out_frac:.0%} eval): "
          f"train={len(train_df)} clips, eval={len(eval_df)} clips, "
          f"eval covers {eval_df['client_id'].nunique()}/{df['client_id'].nunique()} speakers")
    return train_df, eval_df


def write_coqui_csv(df: pd.DataFrame, clips_dir: str, out_path: str):
    """
    Reformat our manifest (client_id, path, ...) into the pipe-delimited
    format the custom formatter expects: audio_file|text|speaker_name
    """
    text_col = "sentence" if "sentence" in df.columns else "text"
    assert text_col in df.columns, "Manifest needs a transcript column ('sentence' or 'text')"
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            audio_path = str(row["path"])
            # Sanitize: line.split("|") is ambiguous if these characters leak in.
            text = (
                str(row[text_col])
                .replace("\r", " ")
                .replace("\n", " ")
                .replace("|", " ")
            )
            speaker = str(row["client_id"]).replace("|", "_")
            f.write(f"{audio_path}|{text}|{speaker}\n")
    return out_path


def cv_formatter(root_path, manifest_file, **kwargs):
    items = []
    with open(manifest_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed line {line_num} in {manifest_file}: {line!r}")
            audio_file, text, speaker_name = parts
            items.append({
                "text": text,
                "audio_file": os.path.join(root_path, audio_file),
                "speaker_name": speaker_name,
                "root_path": root_path,
                "language": "de",
            })
    return items

# ---------------------------------------------------------------------------
# 2) Speaker embedding replacement
# ---------------------------------------------------------------------------

def replace_speaker_embedding(model: Vits, old_speaker_weight: torch.Tensor, num_new_speakers: int):
    """
    Fresh nn.Embedding for the new German speakers. The old VCTK speaker
    identities are discarded, but we initialize around the mean of the old
    embeddings (plus small noise) rather than pure std=0.02 random init, so
    the initial conditioning vectors aren't wildly outside the distribution
    the frozen decoder/flow/posterior-encoder were trained to expect.
    """
    embed_dim = old_speaker_weight.size(1)
    new_emb = nn.Embedding(num_new_speakers, embed_dim)
    with torch.no_grad():
        mean = old_speaker_weight.mean(dim=0, keepdim=True)
        new_emb.weight.copy_(mean + 0.02 * torch.randn(num_new_speakers, embed_dim))
    model.emb_g = new_emb
    print(f"Replaced speaker embedding: {num_new_speakers} speakers x {embed_dim} dims "
          f"(initialized around mean of old VCTK speaker embeddings)")


# ---------------------------------------------------------------------------
# 3) Phoneme/text symbol discovery (full pipeline, not raw characters)
# ---------------------------------------------------------------------------

def load_german_symbol_set(path: str) -> set:
    """Load a precomputed full-dataset symbol inventory (recommended path)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values = raw.get("symbols", raw) if isinstance(raw, dict) else raw
    symbols = set(values)
    invalid = {s for s in symbols if not isinstance(s, str) or len(s) != 1}
    if invalid:
        raise ValueError(f"Expected single-codepoint symbols, got: {sorted(invalid)[:20]}")
    return symbols


def compute_german_symbol_set(texts, language: str, cleaner_fn=None) -> set:
    """
    Fallback: phonemize the given texts with espeak-ng using the SAME
    cleaner and backend configured for training, and collect the resulting
    symbol inventory. Prefer passing a precomputed --german_symbols_json
    that was built from the full dataset with this exact pipeline, so
    training and inference can't drift apart.
    """
    phonemizer = ESpeak(language=language, backend="espeak-ng")
    symbols = set()
    for text in texts:
        cleaned = cleaner_fn(text) if cleaner_fn else text
        try:
            phonemes = phonemizer.phonemize(cleaned, separator="")
            symbols.update(list(phonemes))
        except Exception:
            continue
    return symbols


# ---------------------------------------------------------------------------
# 4) Freezing strategy
# ---------------------------------------------------------------------------

def apply_freezing(model: Vits, freeze_decoder: bool, freeze_posterior_encoder: bool):
    """
    In addition to setting requires_grad here (needed immediately, e.g. for
    building the optimizer param groups below), the corresponding
    config.model_args.freeze_* flags are also set by the caller so the
    freezing strategy is recorded in the saved config and re-applied by
    VITS itself on every epoch/resume.
    """
    if freeze_decoder:
        for p in model.waveform_decoder.parameters():
            p.requires_grad = False
        print("Froze waveform decoder (HiFi-GAN generator)", flush=True)

    if freeze_posterior_encoder:
        for p in model.posterior_encoder.parameters():
            p.requires_grad = False
        print("Froze posterior encoder", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)", flush=True)


def build_differential_optimizers(model: Vits, config, lr_new: float, lr_backbone: float, lr_disc: float):
    """
    VITS trains with two optimizers (discriminator, generator) since it's a
    GAN. The discriminator gets a single uniform LR. The generator's
    trainable parameters are split:
      - emb_g (freshly initialized speaker embedding) -> lr_new.
      - Everything else, including model.text_encoder.emb -> lr_backbone.
    """
    disc_optimizer = torch.optim.AdamW(
        [p for p in model.disc.parameters() if p.requires_grad],
        lr=lr_disc,
        betas=config.optimizer_params.get("betas", (0.8, 0.99)),
        eps=config.optimizer_params.get("eps", 1e-9),
        weight_decay=config.optimizer_params.get("weight_decay", 0.0),
    )

    new_params = [p for p in model.emb_g.parameters() if p.requires_grad]
    # new_param_ids = {id(p) for p in model.emb_g.parameters()}
    new_param_ids = {id(p) for p in new_params}
    
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and id(p) not in new_param_ids
        and not n.startswith("disc.")  # discriminator handled by its own optimizer above
    ]

    gen_optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": new_params, "lr": lr_new},
        ],
        betas=config.optimizer_params.get("betas", (0.8, 0.99)),
        eps=config.optimizer_params.get("eps", 1e-9),
        weight_decay=config.optimizer_params.get("weight_decay", 0.0),
    )

    n_new = sum(p.numel() for p in new_params)
    n_backbone = sum(p.numel() for p in backbone_params)
    print(f"Generator optimizer: {n_new:,} new params (emb_g) @ lr={lr_new}, "
          f"{n_backbone:,} backbone params (incl. text embedding) @ lr={lr_backbone}")

    return [disc_optimizer, gen_optimizer]

class EarlyStopping:
    """Stops training if eval loss doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.epochs_since_improvement = 0

    def on_epoch_end(self, trainer):
        if not trainer.keep_avg_eval:
            return

        values = trainer.keep_avg_eval.avg_values
        eval_loss = values.get("avg_loss_1")
        if eval_loss is None:
            return

        # Sanity print so you can visually confirm on the next run that this
        # is tracking avg_loss_1 (~47-48 range), not the old blended metric.
        print(f"[EarlyStopping] tracking avg_loss_1={eval_loss:.4f}")

        if eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
            print(f"Early stopping: no improvement for {self.epochs_since_improvement}/{self.patience} epochs "
                  f"(best eval loss: {self.best_loss:.4f})")

        if self.epochs_since_improvement >= self.patience:
            print(f"Early stopping triggered -- eval loss hasn't improved in {self.patience} epochs.")
            raise KeyboardInterrupt  # Trainer catches this and saves a checkpoint before exiting


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VITS-VCTK on German Common Voice")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--pretrained_config", required=True)
    parser.add_argument("--final_manifest_tsv", required=True,
                         help="Output of build_final_manifest.py -- speakers, passed clips, with transcripts")
    parser.add_argument("--held_out_frac", type=float, default=0.1,
                         help="Fraction of each speaker's utterances held out for eval")
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--output_path", default="./vits_de_finetune")
    parser.add_argument("--language", default="de")
    parser.add_argument("--german_symbols_json", default=None,
                         help="Precomputed full-dataset phonemized symbol inventory (recommended). "
                              "If omitted, symbols are computed here from the full training set.")
    parser.add_argument("--freeze_decoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_decoder", dest="freeze_decoder", action="store_false")
    parser.add_argument("--freeze_posterior_encoder", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--save_step", type=int, default=1000,
                         help="Checkpoint frequency -- keep this low on preemptible queues so a kill "
                              "doesn't lose much progress")
    parser.add_argument("--lr_new", type=float, default=1e-4,
                         help="LR for freshly-initialized params (speaker embedding)")
    parser.add_argument("--lr_backbone", type=float, default=2.5e-5,
                         help="LR for pretrained-but-trainable params (text encoder incl. its embedding, "
                              "flow, duration predictor)")
    parser.add_argument("--lr_disc", type=float, default=1e-4, help="LR for the discriminator")
    parser.add_argument("--resume_weights_from", default=None,
                         help="Path to a previously fine-tuned German checkpoint "
                             "(e.g. best_model.pth) to restore weights from. "
                             "Starts a FRESH optimizer, epoch counter, and "
                             "early-stopping patience in a new --output_path.")
    parser.add_argument("--phoneme_cache_path", default=None,
                     help="Reuse an existing phoneme cache dir from a prior run "
                          "instead of recomputing.")
    parser.add_argument("--reuse_speakers_json", default=None,
                     help="Path to a prior run's speakers.json to use verbatim, "
                          "verified against --final_manifest_tsv.")
    parser.add_argument("--reuse_split_from", default=None,
                     help="Directory containing a prior run's train.csv/eval.csv "
                          "to reuse verbatim instead of recomputing the holdout split.")
    
    args = parser.parse_args()

    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: speaker mapping from final_manifest_tsv (any size, no longer forced to 109) ---
    speaker_map_recomputed = build_speaker_mapping(args.final_manifest_tsv)

    if args.reuse_speakers_json:
        speaker_map = json.loads(Path(args.reuse_speakers_json).read_text(encoding="utf-8"))
        if speaker_map != speaker_map_recomputed:
            raise ValueError(
                f"--reuse_speakers_json does not match what --final_manifest_tsv "
                f"would produce. This means the manifest has changed since the "
                f"checkpoint you're restoring was trained -- restoring emb_g "
                f"weights under this mapping would silently swap speaker identities. "
                f"Diff: reused has {len(speaker_map)} speakers, recomputed has "
                f"{len(speaker_map_recomputed)} speakers."
            )
        print(f"Reusing speaker map from {args.reuse_speakers_json} "
              f"(verified identical to --final_manifest_tsv derivation)")
    else:
        speaker_map = speaker_map_recomputed

    num_speakers = len(speaker_map)
    speakers_json = out_dir / "speakers.json"
    speakers_json.write_text(json.dumps(speaker_map, ensure_ascii=False, indent=2))

    # --- Step 2: per-speaker held-out split from final_manifest.tsv ---
    if args.reuse_split_from:
        train_txt = str(Path(args.reuse_split_from) / "train.csv")
        eval_txt = str(Path(args.reuse_split_from) / "eval.csv")
        assert Path(train_txt).exists() and Path(eval_txt).exists(), \
            f"train.csv/eval.csv not found in {args.reuse_split_from}"
        print(f"Reusing existing split from {args.reuse_split_from}")
    else:
        train_df, eval_df = per_speaker_holdout_split(args.final_manifest_tsv, held_out_frac=args.held_out_frac)
        train_txt = write_coqui_csv(train_df, args.clips_dir, str(out_dir / "train.csv"))
        eval_txt = write_coqui_csv(eval_df, args.clips_dir, str(out_dir / "eval.csv"))


    # --- Step 3: load pretrained config ---
    config = load_config(args.pretrained_config)
    config.output_path = str(out_dir)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.eval_batch_size = args.eval_batch_size
    config.save_step = args.save_step
    config.save_n_checkpoints = 3
    config.run_eval = True
    config.mixed_precision = True
    config.precision = "fp16"
    config.num_loader_workers = 8
    config.num_eval_loader_workers = 2
    config.use_speaker_weighted_sampler = True
    config.save_best_after = 1000
    config.allow_tf32 = True
    config.max_audio_len = config.audio.sample_rate * 12 

    config.lr = args.lr_backbone
    config.lr_gen = args.lr_new
    config.lr_disc = args.lr_disc
    config.lr_scheduler = None
    config.lr_scheduler_params = {}
    config.lr_scheduler_gen = None
    config.lr_scheduler_gen_params = {}
    config.lr_scheduler_disc = None 
    config.lr_scheduler_disc_params = {}


    # Phonemization/text pipeline must explicitly match the German data.
    config.phoneme_language = args.language
    config.use_phonemes = True
    config.text_cleaner = "basic_german_cleaners"
    config.phoneme_cache_path = args.phoneme_cache_path or str(out_dir / "phoneme_cache")

    # --- freezing strategy recorded in config (applied every epoch by VITS itself) ---
    config.model_args.freeze_waveform_decoder = args.freeze_decoder
    config.model_args.freeze_PE = args.freeze_posterior_encoder
    config.model_args.init_discriminator = True


    # --- Step 4: derive the COMPLETE old vocabulary (pad + punctuation +
    #     graphemes + phonemes + blank), not just config.characters.characters
    #     (which for VCTK is only the 52 English letters). ---
    old_characters, _ = VitsCharacters.init_from_config(config)
    old_vocab = list(old_characters.vocab)
    print(f"old_vocab: {old_vocab}")
    assert len(old_vocab) == config.model_args.num_chars, (
        f"old_vocab size ({len(old_vocab)}) doesn't match "
        f"config.model_args.num_chars ({config.model_args.num_chars}) -- "
        f"pretrained config/checkpoint mismatch."
    )

    # --- Step 5: pull the PRETRAINED text + speaker embedding tensors
    #     directly from the checkpoint, before any model is initialized,
    #     so nothing gets overwritten by a freshly-initialized model first. ---
    if not args.resume_weights_from:
        checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu")
        pretrained_state = checkpoint.get("model", checkpoint)

        old_text_weight = pretrained_state["text_encoder.emb.weight"].detach().cpu()
        assert old_text_weight.size(0) == len(old_vocab)

        old_speaker_weight = pretrained_state["emb_g.weight"].detach().cpu()

    # --- Step 6: discover/load the German symbol inventory and extend the
    #     vocabulary DETERMINISTICALLY, appending only genuinely new symbols
    #     to config.characters.phonemes (not .characters, and not as a list). ---
    # Hand-verified against THIS pretrained checkpoint's vocab (see verification
    # notes -- reproduced with `espeak-ng -v de --ipa`, diffed against
    # VitsCharacters.init_from_config(config), confirmed deterministic/sorted).
    # If you ever swap pretrained checkpoints, this must be re-verified: it is
    # NOT re-derived from old_vocab at runtime, so a stale list here would
    # silently misalign embedding rows with no error raised.
    HARDCODED_GERMAN_SYMBOLS = {' ', '!', "'", '(', ')', ',', '-', '.', '1', ':', ';', '?',
                                 'A', 'B', 'C', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q',
                                 'R', 'S', 'T', 'V', 'W', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                                 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's',
                                 't', 'u', 'v', 'w', 'x', 'y', 'z', '¡', '«', '»', 'ç', 'ð',
                                 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ', 'ɒ', 'ɔ', 'ɕ', 'ə', 'ɛ', 'ɜ', 'ɡ',
                                 'ɨ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ', 'ʏ', 'ʑ', 'ʒ', 'ʔ',
                                 'ʲ', 'ˈ', 'ˌ', 'ː', '̃', '̩', 'θ', '—', '\u201c', '\u201d', '…'}

    if args.german_symbols_json:
        german_symbols = load_german_symbol_set(args.german_symbols_json)
        print(f"Loaded {len(german_symbols)} German symbols from {args.german_symbols_json}")
    else:
        german_symbols = HARDCODED_GERMAN_SYMBOLS
        print(f"Using hardcoded German symbol set ({len(german_symbols)} symbols)")

    new_symbols = sorted(s for s in german_symbols if s not in old_vocab)  # sorted -> deterministic IDs
    config.characters.phonemes = (config.characters.phonemes or "") + "".join(new_symbols)
    print(f"config with new phonemes: {config.characters.phonemes}")
    print(f"Text vocab: {len(old_vocab)} pretrained symbols kept, "
          f"{len(new_symbols)} new German symbols appended to phonemes "
          f"(total vocab: {len(old_vocab) + len(new_symbols)})")

    # CRITICAL: config.model_args.num_chars is a plain int field (default=100) that
    # Vits.__init__ reads directly to size the text embedding table
    # (self.text_encoder = TextEncoder(self.args.num_chars, ...), where
    # self.args IS config.model_args). It is NOT auto-derived from
    # config.characters/the tokenizer vocab length. If left stale at the
    # pretrained checkpoint's old count, the model would build
    # model.text_encoder.emb at the OLD size even though the tokenizer's
    # vocab (and the row-copy logic below) assume the NEW extended size --
    # silently misaligning the blank-token row, then crashing later during
    # training with an embedding index-out-of-range error the first time a
    # new German symbol ID is actually used. Must be set explicitly here,
    # before Vits.init_from_config() constructs the text encoder.
    config.model_args.num_chars = len(old_vocab) + len(new_symbols)

    # --- Step 7: speaker config -- set BOTH top-level and model_args copies,
    #     since model_args takes precedence and an unset copy would leave the
    #     old VCTK speakers_file path (p225, ...) in effect. ---
    config.speakers_file = str(speakers_json)
    config.model_args.speakers_file = str(speakers_json)
    config.num_speakers = num_speakers
    config.model_args.num_speakers = num_speakers
    config.use_speaker_embedding = True
    config.model_args.use_speaker_embedding = True

    # Coqui expects a speaker NAME (name_to_id lookup), not an integer id.
    speaker_names = list(speaker_map.keys())
    preview_speakers = speaker_names[:3]  # or hand-pick specific client_ids

    sentences = [
        "Guten Tag, wie geht es Ihnen heute?",
        "Heute scheint die Sonne, aber morgen wird es wahrscheinlich regnen.",
        "Ich hätte gern fünf Brötchen und zwei Tassen Kaffee, bitte.",
        "Kannst du das Fenster schließen?",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "Um zwölf Uhr fährt der Zug von München nach Nürnberg.",
        "Fröhliche Schüler üben täglich schwierige deutsche Wörter.",
        "Das Mädchen erzählt eine spannende Geschichte über den Winter.",
        "Warum bist du gestern so spät nach Hause gekommen?",
        "Herzlichen Glückwunsch zum Geburtstag! Ich wünsche dir alles Gute.",
    ]
    config.test_sentences = [[text, speaker, None, "de"] for speaker in preview_speakers for text in sentences]

    # --- Step 8: dataset config -- BaseDatasetConfig, not a bare dict
    #     (load_tts_samples expects fields like dataset_name / ignored_speakers). ---
    config.datasets = [
        BaseDatasetConfig(
            formatter=None,
            dataset_name="german_common_voice",
            path=args.clips_dir,
            meta_file_train=str(train_txt),
            meta_file_val=str(eval_txt),
            language=args.language,
        )
    ]

    print(f"config: {config}")

    # --- detect a previous (possibly preempted) run to resume from ---
    existing_checkpoints = sorted(out_dir.glob("checkpoint_*.pth")) + \
        sorted(out_dir.glob("best_model*.pth"), key=lambda p: p.stat().st_mtime)
    is_resume = len(existing_checkpoints) > 0
    if is_resume:
        print(f"Found {len(existing_checkpoints)} existing checkpoint(s) in {out_dir} -- "
              f"resuming instead of starting from the English pretrained checkpoint.")

    # --- Step 9: NOW initialize the model, after config.characters /
    #     config.model_args reflect the final vocabulary + speaker count, so
    #     model.tokenizer and model.text_encoder.emb are built at the right
    #     sizes from the start. ---
    model = Vits.init_from_config(config)

    new_vocab = list(model.tokenizer.characters.vocab)
    old_body_len = len(old_vocab) - 1  # VitsCharacters places <BLNK> last
    assert new_vocab[:old_body_len] == old_vocab[:old_body_len], \
        "Old vocabulary prefix changed unexpectedly -- refusing to misalign embedding rows."
    assert new_vocab[-1] == old_vocab[-1], "Blank token position assumption violated."
    assert len(new_vocab) == len(old_vocab) + len(new_symbols)
    # Safety net for exactly the num_chars staleness bug described above: confirms
    # the embedding table Vits actually built matches the tokenizer vocab length,
    # not just that we intended it to.
    assert model.text_encoder.emb.weight.shape[0] == len(new_vocab), (
        f"model.text_encoder.emb has {model.text_encoder.emb.weight.shape[0]} rows but "
        f"tokenizer vocab has {len(new_vocab)} entries -- config.model_args.num_chars "
        f"was not applied before model construction."
    )
    if not is_resume:
        if args.resume_weights_from:
            # --- Clean restore: load a full previously fine-tuned German
            #     checkpoint. Requires the SAME final_manifest_tsv (same
            #     speaker count/order) and SAME pretrained_config/German
            #     symbols as the run that produced this checkpoint, since
            #     num_speakers/vocab size must match exactly. ---
            print(f"Restoring weights from {args.resume_weights_from} "
                  f"(fresh optimizer, epoch counter, and early-stopping state)")
            resume_ckpt = torch.load(args.resume_weights_from, map_location="cpu")
            resume_state = resume_ckpt.get("model", resume_ckpt)
            missing, unexpected = model.load_state_dict(resume_state, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint/model mismatch on resume. "
                    f"Missing={missing}, unexpected={unexpected}. "
                    f"This usually means num_speakers or the vocab differs "
                    f"from the run that produced this checkpoint."
                )
            print("Restored full model state (speaker embedding, text "
                  "embedding, and all trainable weights).")
        else:
            # Load everything except the two embeddings whose shapes changed.
            filtered_state = {
                k: v for k, v in pretrained_state.items()
                if k not in {"emb_g.weight", "text_encoder.emb.weight"}
            }
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)

            allowed_missing = {"emb_g.weight", "text_encoder.emb.weight"}
            extra_missing = set(missing) - allowed_missing
            unexpected_disc = {k for k in extra_missing if not k.startswith("disc.")}
            if unexpected or unexpected_disc:
                raise RuntimeError(
                    f"Checkpoint/model mismatch. Unexpectedly missing={sorted(unexpected_disc)}, "
                    f"unexpected keys={sorted(unexpected)}"
                )
            print(f"Loaded pretrained weights (excluding speaker + text embeddings). "
                  f"Missing (expected): {sorted(extra_missing & allowed_missing)}, "
                  f"unexpected: {len(unexpected)}")

            replace_speaker_embedding(model, old_speaker_weight, num_new_speakers=num_speakers)

            with torch.no_grad():
                model.text_encoder.emb.weight[:old_body_len].copy_(old_text_weight[:old_body_len])
                model.text_encoder.emb.weight[-1].copy_(old_text_weight[-1])
            print(f"Text embedding: transferred {old_body_len + 1} pretrained rows ", flush=True)
            print(f"incl. blank, {len(new_symbols)} new German rows left randomly initialized.", flush=True)
    
    else:
        # Resuming: Trainer restores model + optimizer + scheduler state from
        # the latest checkpoint in out_dir once trainer.fit() starts.
        print("Skipping English pretrained weight load -- will be restored from local checkpoint instead.")

    # --- Step 12: freeze the language-agnostic acoustic backbone (requires_grad,
    #     used immediately below for optimizer groups; config flags set above
    #     make VITS re-apply this every epoch/resume). ---
    apply_freezing(model, args.freeze_decoder, args.freeze_posterior_encoder)

    # --- Step 13: validate the final tokenizer covers everything before
    #     spending any compute on training. ---
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        formatter=cv_formatter,
    )

    final_vocab = set(model.tokenizer.characters.vocab)
    missing_symbols = german_symbols - final_vocab
    if missing_symbols:
        raise ValueError(
            f"German symbols missing from tokenizer vocabulary: "
            f"{sorted(missing_symbols)}"
        )
    print(
        f"Vocabulary check passed: all {len(german_symbols)} German symbols "
        f"are present.",
        flush=True,
    )

    # --- Step 14: differential-LR optimizers, wired in via model.get_optimizer()
    #     since Trainer() does not accept an optimizer= kwarg. ---
    def _get_optimizer(self):
        return build_differential_optimizers(
            self, config, lr_new=args.lr_new, lr_backbone=args.lr_backbone, lr_disc=args.lr_disc
        )
    model.get_optimizer = MethodType(_get_optimizer, model)

    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    trainer_args = TrainerArgs(continue_path=str(out_dir)) if is_resume else TrainerArgs()
    trainer_args.use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    trainer_args.rank = int(os.environ.get("LOCAL_RANK", "0"))

    trainer = Trainer(
        trainer_args,
        config,
        output_path=str(out_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        callbacks={"on_epoch_end": early_stopping.on_epoch_end},
        parse_command_line_args=False,  # this script has its own argparse args
    )

    if is_resume:
        # Trainer.__init__ restored optimizer state (including the OLD saved
        # lr per param group) from the checkpoint when continue_path was set.
        # That happens AFTER our custom get_optimizer() already set the
        # correct lr from --lr_new/--lr_backbone/--lr_disc, silently
        # overwriting it. Force this run's LR back onto the now-restored
        # (momentum-preserving) optimizer objects here.
        disc_opt, gen_opt = trainer.optimizer
        for group in disc_opt.param_groups:
            group["lr"] = args.lr_disc
        for i, group in enumerate(gen_opt.param_groups):
            # index 0 = backbone_params, index 1 = new_params (see build_differential_optimizers)
            group["lr"] = args.lr_backbone if i == 0 else args.lr_new
        print(f"[Resume] Overrode restored optimizer LR: "
                f"disc={args.lr_disc}, backbone={args.lr_backbone}, new={args.lr_new}")
        # Also verify epochs wasn't silently reset by the continue_path config reload
        print(f"[Resume] trainer.config.epochs = {trainer.config.epochs} "
                f"(should be {args.epochs} -- if not, config.json in the resumed "
                f"folder is stale relative to --epochs on this command line)")
    
    trainer.fit()
    # --- Final export: bundle everything needed for later inference ---
    final_dir = out_dir / "final_model"
    final_dir.mkdir(exist_ok=True)

    best_candidates = sorted(out_dir.glob("best_model*.pth"), key=lambda p: p.stat().st_mtime)
    if best_candidates:
        import shutil
        shutil.copy(best_candidates[-1], final_dir / "best_model.pth")
    else:
        torch.save({"model": model.state_dict(), "config": config.to_dict()}, final_dir / "model.pth")
        print("WARNING: no best_model*.pth found -- saved current model state instead. "
              "Check config.run_eval is True if you expected a best-model checkpoint.")

    config.save_json(str(final_dir / "config.json"))
    with open(str(final_dir / "symbols.json"), "w", encoding="utf-8") as f:
        json.dump(new_vocab, f, ensure_ascii=False, indent=2)
    (final_dir / "speakers.json").write_text(json.dumps(speaker_map, ensure_ascii=False, indent=2))

    print(f"\nFinal inference-ready model bundled at: {final_dir}")
    print("  best_model.pth / model.pth -- weights")
    print("  config.json                -- includes German phoneme_language + extended phonemes")
    print("  symbols.json               -- FULL final vocabulary (pad+punct+graphemes+phonemes+blank)")
    print("  speakers.json              -- client_id -> embedding index mapping")



if __name__ == "__main__":
    main()