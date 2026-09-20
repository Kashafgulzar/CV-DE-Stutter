#!/usr/bin/env python3
"""
Generate dysfluent speech from IPA transcripts annotated with inline markers
    [PRO]  -> prolong the phoneme immediately BEFORE the marker
    [PAU]  -> insert a short intra-word pause at that point
    [SB]   -> insert a longer "silent block" (inter-word pause) at that point
Any other bracket tag (e.g. [INT], [wREP], [sREP]) is treated as already
resolved in the text (REP-type dysfluencies are assumed to already be
spelled out as literal repeated phonemes in `ipa_dysfluent`) and is simply
stripped out before synthesis rather than acted on.

METHODOLOGY (adapted from Berkeley-Speech-Group/LLM-Dys, data_simulation/VITS,
and the HASS paper's description of the same pipeline):

  Prolongation:
    Encode the text once, predict the duration matrix `w_ceil` once, add
    extra frames (0.17-0.8s, configurable) at the token index of every
    phoneme flagged by a [PRO] marker, then build the alignment path and
    decode. The prolongation is baked into the model's own duration/
    attention mechanism, not created by repeating waveform samples.

    Both [PRO] and non-[PRO] lines are synthesized through the SAME manual
    encode -> duration -> align -> flow -> decode pipeline (rather than one
    path calling Coqui's black-box `Vits.inference()` and the other doing
    this manually), specifically so the sampling noise scale is guaranteed
    identical whether or not a line has a [PRO] marker. Letting the two
    paths use different noise scales would make "has a prolongation" and
    "sounds different for unrelated reasons" the same acoustic signal,
    which is exactly the kind of confound a disfluency-detection model
    could learn to exploit instead of the actual prolongation.

    OPTIONAL, OFF BY DEFAULT (`--pro_blend_next_strength`, default 0.0):
    pure duration-extension can decode as silence/breath noise on some
    checkpoints for phonemes whose predicted variance is very low, because
    the extended span is otherwise many near-identical frames outside
    anything seen in training. If you hit that, this option ramps the
    prolonged phoneme's identity gradually (linearly, across the extended
    span) toward the following phoneme's identity, so it reads as a
    natural glide into the next sound rather than a flat repeated frame.
    At 0.0 this does nothing and you get the byte-faithful reference
    behaviour described above.

  Pause / Silent block:
    Locate the time boundary adjacent to a [PAU]/[SB] marker from the
    duration matrix (sum of frame durations up to that token, converted to
    seconds via hop_length/sample_rate), then splice true silence into the
    already-synthesized waveform at that timestamp with a short cosine
    crossfade on both sides so the splice doesn't click. Silence duration
    is drawn uniformly at random: 0.3-1.5s for [PAU], 0.8-3.5s for [SB].

Requires: coqui-tts (`pip install coqui-tts`), torch, soundfile, numpy, tqdm.

Usage: python synthesize_dysfluent_audio.py \
    --final_dir "path/to/VITS-CV-DE" \
    --jsonl "path/to/dysfluent_transcript/jsonl" \
    --output_dir "CV_Dysfluent" \
    --device "cuda" \
    --seed 1234 \
    --manifest cv_dys_output_manifest.jsonl \
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from phoneme_inventory import ELIGIBLE, build_char_symbol_map

logger = logging.getLogger("dysfluent_synth")

# --------------------------------------------------------------------------- #
# Marker parsing
# --------------------------------------------------------------------------- #

# Matches ANY bracket tag, anywhere in the string (not anchored to a full
# token) -- markers in real transcripts appear both as standalone
# space-delimited tokens (e.g. " [SB] ") and embedded mid-word with no
# surrounding whitespace at all (e.g. "b[PAU]ˈɪslaŋ", "bəkˈan[PRO]t").
BRACKET_TAG_RE = re.compile(r"\[([A-Za-z]+)\]")

_HANDLED_MARKERS = ("PRO", "SB", "PAU")


@dataclass
class Marker:
    kind: str          # "PRO" | "SB" | "PAU"
    anchor_char: int    # index into clean_text of the last real char before the marker (-1 if none)


def parse_dysfluent_ipa(text: str) -> tuple[str, list[Marker]]:
    """
    Strip [PRO]/[SB]/[PAU] markers out of an ipa_dysfluent string, returning
    the marker-free phoneme text plus a list of Marker objects recording,
    for each marker, the character index (in the FINAL, whitespace-
    normalized clean_text) of the last real (non-whitespace) character that
    preceded it -- i.e. the phoneme to prolong, or the point after which to
    splice silence.

    This runs as a single forward pass: clean_text is built up character by
    character, whitespace is collapsed to a single space AS WE GO (any run
    of whitespace -- possibly spanning one or more removed tags -- becomes
    exactly one space if it contained any whitespace at all, or zero spaces
    if it didn't, e.g. for a marker embedded mid-word), and each marker's
    anchor is recorded against the length of clean_text at the exact moment
    the marker is encountered. Because clean_text is never modified
    retroactively, an anchor recorded this way is always correct against
    the string this function ultimately returns -- there is no separate
    "collapse whitespace" pass afterward that could invalidate indices
    computed earlier (which is what a two-pass version of this function
    would risk: any tag surrounded by spaces on both sides, e.g. [SB],
    leaves a double space when removed, and collapsing that after the fact
    shifts every anchor that comes after it).

    Any OTHER bracket tag (e.g. [INT], [wREP], [sREP]) is dropped with a
    warning rather than acted on, and contributes no characters and no
    anchor -- it's fully transparent to the anchors around it.
    """
    clean_chars: list[str] = []
    markers: list[Marker] = []
    last_real_idx = -1        # final index in clean_chars of the last real char emitted
    seen_any_real = False     # suppresses leading whitespace
    pending_space = False     # whitespace seen since the last real char, not yet emitted

    i = 0
    n = len(text)
    while i < n:
        m = BRACKET_TAG_RE.match(text, i)
        if m:
            tag = m.group(1)
            if tag in _HANDLED_MARKERS:
                markers.append(Marker(kind=tag, anchor_char=last_real_idx))
            else:
                # Expected, not an error: REP-type tags (and any other tag
                # this script doesn't act on) mark a point in the transcript
                # but carry no separate acoustic instruction of their own --
                # the repeated phonemes are already spelled out as literal
                # text. Logged at DEBUG since this fires on every such tag,
                # which is most lines in a stuttering dataset.
                logger.debug("Dropping unhandled bracket tag [%s] from transcript segment near index %d", tag, i)
            i = m.end()
            continue

        ch = text[i]
        if ch.isspace():
            if seen_any_real:
                pending_space = True
            i += 1
            continue

        if pending_space:
            clean_chars.append(' ')
            pending_space = False
        clean_chars.append(ch)
        last_real_idx = len(clean_chars) - 1
        seen_any_real = True
        i += 1

    clean_text = "".join(clean_chars)
    return clean_text, markers


# --------------------------------------------------------------------------- #
# Tokenization with character -> final-token-index bookkeeping
# --------------------------------------------------------------------------- #

def encode_with_char_map(tokenizer, clean_text: str) -> tuple[list[int], dict[int, int]]:
    """
    Mirrors TTSTokenizer.text_to_ids() but (a) skips phonemization, since
    `clean_text` is already IPA, and (b) records a mapping from each
    successfully-encoded character's index in `clean_text` to its index in
    the *pre*-blank-intersperse id sequence, so we can later locate exactly
    which token in the model's duration matrix corresponds to a given
    marker anchor.

    Deliberately does NOT run tokenizer.text_cleaner: that cleaner is
    designed for raw grapheme text, and running it here would risk
    changing the string (length or content) AFTER marker anchors were
    already computed against clean_text, silently reintroducing the same
    class of index-misalignment bug that parse_dysfluent_ipa's single-pass
    design was written to avoid. If your checkpoint's config does specify
    a cleaner, you'll get a one-time warning below -- verify offline
    whether it does anything necessary for your phoneme set; if it does,
    anchors would need to be recomputed after cleaning rather than before.
    """
    if tokenizer.text_cleaner is not None:
        logger.warning(
            "Tokenizer has a text_cleaner configured (%s) but it is being "
            "intentionally bypassed for phoneme input -- see the docstring "
            "of encode_with_char_map if you suspect this cleaner does "
            "something your phoneme set actually needs.",
            getattr(tokenizer.text_cleaner, "__name__", tokenizer.text_cleaner),
        )

    ids: list[int] = []
    char_to_pretoken: dict[int, int] = {}
    for ci, ch in enumerate(clean_text):
        try:
            tid = tokenizer.characters.char_to_id(ch)
        except KeyError:
            continue
        char_to_pretoken[ci] = len(ids)
        ids.append(tid)

    return ids, char_to_pretoken


def pretoken_to_final_index(pre_idx: int, tokenizer) -> int:
    """Map a pre-blank/bos index to its final index in the id sequence fed to the model."""
    idx = pre_idx
    if tokenizer.add_blank:
        idx = 2 * idx + 1
    if tokenizer.use_eos_bos:
        idx += 1
    return idx


def finalize_ids(pre_ids: list[int], tokenizer) -> list[int]:
    ids = pre_ids
    if tokenizer.add_blank:
        ids = tokenizer.intersperse_blank_char(ids, True)
    if tokenizer.use_eos_bos:
        ids = tokenizer.pad_with_bos_eos(ids)
    return ids


def build_final_idx_to_char(char_to_pretoken: dict[int, int], tokenizer) -> dict[int, int]:
    """Forward map: final token index -> the clean_text character index it
    represents. Tokens with no entry are blanks/pad/bos/eos."""
    out = {}
    for char_idx, pre_idx in char_to_pretoken.items():
        out[pretoken_to_final_index(pre_idx, tokenizer)] = char_idx
    return out


def resolve_anchor_token_index(
    anchor_char: int,
    char_to_pretoken: dict[int, int],
    clean_text: str,
    char_symbol_map: dict[int, str],
    tokenizer,
    require_continuant: bool = False,
) -> int | None:
    """
    Find the final token index corresponding to a marker anchor. If the
    exact anchor character was dropped (out-of-vocab), or -- when
    require_continuant is True -- is part of a stop/affricate/tap/trill/
    glide symbol that can't be meaningfully held, search backward for the
    nearest preceding character that qualifies.

    The backward search NEVER crosses a whitespace character: the moment
    it encounters one, it stops and returns None (the caller then logs and
    skips this marker) rather than silently re-anchoring the marker onto a
    different word entirely.

    require_continuant classification is symbol-based (via
    char_symbol_map, built from the shared phoneme_inventory tokenizer),
    not single-character-based, so a multi-character symbol like the
    affricate "tʃ" is correctly excluded as a whole unit rather than
    accidentally passing because its last character happens to look like a
    fricative.
    """
    if anchor_char < 0:
        return None
    ci = anchor_char
    while ci >= 0:
        if clean_text[ci].isspace():
            return None
        if ci in char_to_pretoken:
            if not require_continuant:
                return pretoken_to_final_index(char_to_pretoken[ci], tokenizer)
            symbol = char_symbol_map.get(ci)
            if symbol is not None and symbol in ELIGIBLE:
                return pretoken_to_final_index(char_to_pretoken[ci], tokenizer)
        ci -= 1
    return None


# --------------------------------------------------------------------------- #
# Model bundle loading
# --------------------------------------------------------------------------- #

@dataclass
class ModelBundle:
    model: "object"
    config: "object"
    tokenizer: "object"
    speaker_manager: "object"
    device: torch.device
    sample_rate: int
    hop_length: int


def load_bundle(final_dir: str, device_str: str = "auto") -> ModelBundle:
    from TTS.config import load_config
    from TTS.tts.models.vits import Vits

    final_dir = Path(final_dir)
    config_path = final_dir / "config.json"
    speakers_path = final_dir / "speakers.json"
    ckpt_path = final_dir / "checkpoint.pth"
    if not ckpt_path.exists():
        ckpt_path = final_dir / "model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pth or model.pth found in {final_dir}")

    logger.info("Loading config from %s", config_path)
    config = load_config(str(config_path))

    if speakers_path.exists():
        config.speakers_file = str(speakers_path)
    else:
        logger.warning("speakers.json not found at %s; multi-speaker lookup by client_id will fail", speakers_path)

    logger.info("Building model from config (this also builds the tokenizer + speaker manager)")
    model = Vits.init_from_config(config)

    logger.info("Loading weights from %s", ckpt_path)
    model.load_checkpoint(config, str(ckpt_path), eval=True)

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model.to(device)
    model.eval()

    sample_rate = config.audio["sample_rate"]
    hop_length = config.audio["hop_length"]

    return ModelBundle(
        model=model,
        config=config,
        tokenizer=model.tokenizer,
        speaker_manager=model.speaker_manager,
        device=device,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )


def resolve_speaker_id(bundle: ModelBundle, client_id: str) -> int:
    sm = bundle.speaker_manager
    if sm is not None and sm.name_to_id and client_id in sm.name_to_id:
        return sm.name_to_id[client_id]
    raise KeyError(f"client_id '{client_id}' not found in speakers.json / speaker manager")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG actually used during synthesis. random.seed() alone
    (the previous behaviour) only reproduces the duration-extension length
    and pause lengths, since it doesn't touch PyTorch's generator -- but
    the acoustically important randomness (z_p sampling via
    torch.randn_like, and the stochastic duration predictor's own
    torch.randn-based sampling) both come from torch's RNG, not Python's.

    Note: even with torch seeded, exact bit-for-bit reproducibility on GPU
    additionally requires deterministic=True, which forces deterministic
    (often slower) CUDA/cuDNN kernels via torch.use_deterministic_algorithms.
    CPU inference is deterministic here without that flag.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Core synthesis
# --------------------------------------------------------------------------- #

@dataclass
class DysfluencyRanges:
    pro_min_sec: float = 0.17
    pro_max_sec: float = 0.80
    pau_min_sec: float = 0.30   # intra-word pause  ([PAU])
    pau_max_sec: float = 1.50
    sb_min_sec: float = 0.80    # silent block / inter-word pause ([SB])
    sb_max_sec: float = 3.50
    crossfade_ms: float = 30.0

    # None = use the checkpoint's own configured default
    # (model.inference_noise_scale / model.inference_noise_scale_dp).
    # Applied identically to PRO and non-PRO lines -- see module docstring
    # for why that match matters.
    inference_noise_scale: float | None = None
    inference_noise_scale_dp: float | None = None

    # OFF by default (0.0 = pure reference behaviour: duration extension
    # only, no identity blending). See module docstring for when/why to
    # raise this.
    pro_blend_next_strength: float = 0.0


def frames_for_seconds(seconds: float, sample_rate: int, hop_length: int) -> int:
    return max(1, round(seconds * sample_rate / hop_length))


def run_vits_inference(
    bundle: ModelBundle,
    ids: list[int],
    speaker_id: int,
    pro_token_indices: list[int],
    ranges: DysfluencyRanges,
    final_idx_to_char: dict[int, int] | None = None,
    clean_text: str | None = None,
):
    """
    Manual VITS inference: encode -> predict duration once -> optionally
    extend duration at pro_token_indices -> build alignment path -> flow ->
    decode. Used for BOTH [PRO] and non-[PRO] lines (pro_token_indices=[]
    for the latter) rather than routing non-PRO lines through Coqui's
    black-box Vits.inference(), specifically so noise-scale handling and
    truncation checking are identical on both paths.
    """
    from TTS.tts.utils.helpers import generate_path, sequence_mask

    model = bundle.model
    device = bundle.device

    x = torch.LongTensor(ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(ids)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)

    noise_scale = ranges.inference_noise_scale
    if noise_scale is None:
        noise_scale = getattr(model, "inference_noise_scale", 1.0)
    noise_scale_dp = ranges.inference_noise_scale_dp
    if noise_scale_dp is None:
        noise_scale_dp = getattr(model, "inference_noise_scale_dp", 1.0)

    with torch.no_grad():
        g = None
        if model.args.use_speaker_embedding and sid is not None:
            g = model.emb_g(sid).unsqueeze(-1)

        x_enc, m_p, logs_p, x_mask = model.text_encoder(x, x_lengths, lang_emb=None)

        dp_g = g if getattr(model.args, "condition_dp_on_speaker", True) else None
        if model.args.use_sdp:
            logw = model.duration_predictor(x_enc, x_mask, g=dp_g, reverse=True, noise_scale=noise_scale_dp)
        else:
            logw = model.duration_predictor(x_enc, x_mask, g=dp_g)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w)

        for tok_idx in pro_token_indices:
            extra_sec = random.uniform(ranges.pro_min_sec, ranges.pro_max_sec)
            extra_frames = frames_for_seconds(extra_sec, bundle.sample_rate, bundle.hop_length)
            w_ceil[0, 0, tok_idx] += extra_frames

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)
        attn_mask = x_mask * y_mask.transpose(1, 2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p_exp = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_exp = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        if pro_token_indices and ranges.pro_blend_next_strength > 0:
            cum = torch.cumsum(w_ceil[0, 0], dim=0)
            T_dec = m_p_exp.shape[-1]
            num_tokens = w_ceil.shape[-1]

            for tok_idx in pro_token_indices:
                end = int(cum[tok_idx].item())
                start = end - int(w_ceil[0, 0, tok_idx].item())
                start, end = max(0, min(start, T_dec)), max(0, min(end, T_dec))
                span_len = end - start
                if span_len <= 1:
                    continue

                # Find the next token that maps to a genuine (non-space)
                # phoneme character, skipping blanks/pad/bos/eos (which
                # have no entry in final_idx_to_char) AND skipping any
                # token that maps to a literal space character -- never
                # blend toward a word-boundary token. If none is found
                # before the end of the utterance, skip blending for this
                # marker rather than guessing.
                next_tok = None
                if final_idx_to_char is not None and clean_text is not None:
                    for cand in range(tok_idx + 1, num_tokens):
                        char_idx = final_idx_to_char.get(cand)
                        if char_idx is not None and not clean_text[char_idx].isspace():
                            next_tok = cand
                            break
                if next_tok is None:
                    continue

                target_mp = m_p[:, :, next_tok].unsqueeze(-1)
                # Ramp from 0% blend at the start of the extended span to
                # pro_blend_next_strength by the end, rather than a
                # constant blend across the whole span -- reads as a
                # gradual coarticulatory glide into the next sound.
                ramp = torch.linspace(
                    0.0, ranges.pro_blend_next_strength, span_len, device=device, dtype=m_p_exp.dtype
                ).view(1, 1, -1)
                m_p_exp[:, :, start:end] = (1 - ramp) * m_p_exp[:, :, start:end] + ramp * target_mp

        z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale
        z = model.flow(z_p, y_mask, g=g, reverse=True)
        z, _, _, y_mask = model.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        max_len = getattr(model, "max_inference_len", None)
        truncated = max_len is not None and z.shape[-1] > max_len
        if truncated:
            logger.warning(
                "Decode length %d exceeds model.max_inference_len=%d; output "
                "will be truncated. Consider shortening this transcript or "
                "reducing pro_max_sec.",
                z.shape[-1], max_len,
            )
        o = model.waveform_decoder((z * y_mask)[:, :, :max_len], g=g)

    return {"model_outputs": o, "durations": w_ceil, "truncated": truncated}


def synthesize_dysfluent_utterance(
    bundle: ModelBundle,
    ipa_dysfluent: str,
    speaker_id: int,
    ranges: DysfluencyRanges,
    seed: int | None = None,
    deterministic: bool = False,
    require_continuant_for_pro: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Synthesize one ipa_dysfluent transcript, realizing [PRO] as genuine
    duration-matrix prolongation and [PAU]/[SB] as post-hoc silence
    splices. Returns (waveform, truncated).
    """
    if seed is not None:
        set_seed(seed, deterministic=deterministic)

    tokenizer = bundle.tokenizer
    clean_text, markers = parse_dysfluent_ipa(ipa_dysfluent)
    pre_ids, char_to_pretoken = encode_with_char_map(tokenizer, clean_text)
    if not pre_ids:
        raise ValueError(f"No valid phonemes found after cleaning: {ipa_dysfluent!r}")
    final_ids = finalize_ids(pre_ids, tokenizer)
    final_idx_to_char = build_final_idx_to_char(char_to_pretoken, tokenizer)
    char_symbol_map = build_char_symbol_map(clean_text)

    logger.debug("clean_text=%r", clean_text)

    resolved: list[tuple[int, str]] = []
    for m in markers:
        tok_idx = resolve_anchor_token_index(
            m.anchor_char, char_to_pretoken, clean_text, char_symbol_map, tokenizer,
            require_continuant=(m.kind == "PRO" and require_continuant_for_pro),
        )
        if tok_idx is None:
            logger.warning("Could not resolve anchor for marker %s in %r; skipping it", m.kind, ipa_dysfluent)
            continue
        resolved.append((tok_idx, m.kind))

    pro_markers = [t for t, k in resolved if k == "PRO"]
    pause_markers = [(t, k) for t, k in resolved if k in ("PAU", "SB")]

    valid_pro = [t for t in pro_markers if t < len(final_ids)]
    if len(valid_pro) != len(pro_markers):
        logger.warning("Some PRO token indices were out of range and were skipped: %r", pro_markers)

    for idx in valid_pro:
        token_id = final_ids[idx]
        logger.debug(
            "[PRO] final_idx=%d token_id=%d char=%s", idx, token_id, tokenizer.characters.id_to_char(token_id)
        )

    outputs = run_vits_inference(
        bundle, final_ids, speaker_id, pro_token_indices=valid_pro, ranges=ranges,
        final_idx_to_char=final_idx_to_char, clean_text=clean_text,
    )
    w_ceil = outputs["durations"]
    truncated = outputs["truncated"]

    waveform = outputs["model_outputs"].squeeze().detach().cpu().float().numpy()

    # --- Splice in [PAU]/[SB] silence at the timestamps implied by w_ceil ---
    if pause_markers:
        timed = []
        for tok_idx, kind in pause_markers:
            tok_idx = min(tok_idx, w_ceil.shape[-1] - 1)
            frames_before = float(w_ceil[0, 0, : tok_idx + 1].sum().item())
            t_sec = frames_before * bundle.hop_length / bundle.sample_rate
            timed.append((t_sec, kind))
        timed.sort(key=lambda x: x[0])

        offset_sec = 0.0
        for t_sec, kind in timed:
            if kind == "SB":
                dur = random.uniform(ranges.sb_min_sec, ranges.sb_max_sec)
            else:  # PAU
                dur = random.uniform(ranges.pau_min_sec, ranges.pau_max_sec)
            insert_at = t_sec + offset_sec
            waveform = insert_silence(
                waveform, bundle.sample_rate, insert_at, dur, crossfade_ms=ranges.crossfade_ms
            )
            offset_sec += dur

    return waveform.astype(np.float32), truncated


def insert_silence(
    waveform: np.ndarray, sample_rate: int, position_sec: float, duration_sec: float, crossfade_ms: float = 30.0
) -> np.ndarray:
    """
    Splice `duration_sec` of true silence into `waveform` at `position_sec`,
    with a short cosine fade-out/fade-in on either side of the splice so it
    doesn't click.
    """
    pos = int(round(position_sec * sample_rate))
    pos = max(0, min(pos, len(waveform)))
    fade_len = max(1, int(round(crossfade_ms * sample_rate / 1000.0)))
    fade_len = min(fade_len, pos, len(waveform) - pos)

    silence = np.zeros(int(round(duration_sec * sample_rate)), dtype=waveform.dtype)

    if fade_len > 0:
        pre = waveform[pos - fade_len: pos].copy()
        post = waveform[pos: pos + fade_len].copy()
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=waveform.dtype)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=waveform.dtype)
        pre *= fade_out
        post *= fade_in
        left = np.concatenate([waveform[: pos - fade_len], pre])
        right = np.concatenate([post, waveform[pos + fade_len:]])
    else:
        left = waveform[:pos]
        right = waveform[pos:]

    return np.concatenate([left, silence, right])


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #

def process_jsonl(
    final_dir: str,
    jsonl_path: str,
    output_dir: str,
    device: str = "auto",
    ranges: DysfluencyRanges | None = None,
    limit: int | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    manifest_path: str | None = None,
    require_continuant_for_pro: bool = False,
):
    ranges = ranges or DysfluencyRanges()
    bundle = load_bundle(final_dir, device)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if limit is not None:
        lines = lines[:limit]

    manifest = []
    n_ok, n_fail, n_truncated = 0, 0, 0

    for i, item in enumerate(tqdm(lines, desc="Synthesizing")):
        client_id = item["client_id"]
        rel_path = item["path"]
        ipa_dysfluent = item["ipa_dysfluent"]

        out_path = out_root / rel_path
        if out_path.suffix.lower() != ".wav":
            out_path = out_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            speaker_id = resolve_speaker_id(bundle, client_id)
            item_seed = None if seed is None else seed + i
            waveform, truncated = synthesize_dysfluent_utterance(
                bundle, ipa_dysfluent, speaker_id, ranges, seed=item_seed,
                deterministic=deterministic, require_continuant_for_pro=require_continuant_for_pro,
            )
            sf.write(str(out_path), waveform, bundle.sample_rate)
            n_ok += 1
            if truncated:
                n_truncated += 1
            manifest.append(
                {
                    "client_id": client_id,
                    "path": str(out_path.relative_to(out_root)),
                    "reference_text": item.get("reference_text"),
                    "ipa_correct": item.get("ipa_correct"),
                    "ipa_dysfluent": ipa_dysfluent,
                    "orig_duration_sec": item.get("duration_sec"),
                    "synth_duration_sec": round(len(waveform) / bundle.sample_rate, 4),
                    "truncated": truncated,
                    "status": "ok",
                }
            )
        except Exception as e:  # noqa: BLE001 - keep batch running on per-item failure
            n_fail += 1
            logger.exception("Failed on item %d (client_id=%s, path=%s): %s", i, client_id, rel_path, e)
            manifest.append(
                {
                    "client_id": client_id,
                    "path": rel_path,
                    "status": "failed",
                    "error": str(e),
                }
            )

    manifest_path = manifest_path or str(out_root / "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "Done. %d succeeded (%d truncated), %d failed. Manifest: %s",
        n_ok, n_truncated, n_fail, manifest_path,
    )
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final_dir", required=True, help="Directory with best_model.pth/model.pth, config.json, symbols.json, speakers.json")
    p.add_argument("--jsonl", required=True, help="Path to the dysfluent-transcript jsonl")
    p.add_argument("--output_dir", required=True, help="Where to write generated wavs (mirrors 'path' from jsonl)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--limit", type=int, default=None, help="Only process first N rows (debugging)")
    p.add_argument("--seed", type=int, default=None, help="Base random seed for reproducibility (per-item seed = seed + row index)")
    p.add_argument(
        "--deterministic", action="store_true",
        help="Force deterministic CUDA/cuDNN kernels for bit-exact reproducibility on GPU. "
             "Slower; irrelevant on CPU, which is already deterministic once --seed is set.",
    )
    p.add_argument("--manifest", default=None, help="Output manifest jsonl path (default: <output_dir>/manifest.jsonl)")

    p.add_argument("--pro_min_sec", type=float, default=0.17)
    p.add_argument("--pro_max_sec", type=float, default=0.80)
    p.add_argument("--pau_min_sec", type=float, default=0.30, help="intra-word [PAU] min duration")
    p.add_argument("--pau_max_sec", type=float, default=1.50, help="intra-word [PAU] max duration")
    p.add_argument("--sb_min_sec", type=float, default=0.80, help="[SB] silent-block min duration")
    p.add_argument("--sb_max_sec", type=float, default=3.50, help="[SB] silent-block max duration")
    p.add_argument("--crossfade_ms", type=float, default=30.0)
    p.add_argument(
        "--inference_noise_scale", type=float, default=None,
        help="z_p sampling noise scale, applied identically to PRO and non-PRO lines. "
             "Default: use the checkpoint's own configured value.",
    )
    p.add_argument(
        "--inference_noise_scale_dp", type=float, default=None,
        help="Stochastic duration predictor noise scale, applied identically to PRO and "
             "non-PRO lines. Default: use the checkpoint's own configured value.",
    )
    p.add_argument(
        "--pro_require_continuant", action="store_true",
        help="If set, [PRO] never anchors on a stop/affricate/tap/trill/glide -- it searches "
             "backward (within the same word only) for the nearest audibly-holdable phoneme "
             "instead. Off by default (see NOTE in synthesize_dysfluent_utterance).",
    )
    p.add_argument(
        "--pro_blend_next_strength", type=float, default=0.0,
        help="OFF by default (0.0 = pure reference behaviour: duration extension only). If "
             "pure duration-stretching decodes as silence/breath noise on your checkpoint for "
             "some phonemes, try 0.1-0.2: this ramps the prolonged phoneme's identity toward "
             "the following phoneme's identity across the extended span. See module docstring.",
    )
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_arg_parser().parse_args()

    ranges = DysfluencyRanges(
        pro_min_sec=args.pro_min_sec,
        pro_max_sec=args.pro_max_sec,
        pau_min_sec=args.pau_min_sec,
        pau_max_sec=args.pau_max_sec,
        sb_min_sec=args.sb_min_sec,
        sb_max_sec=args.sb_max_sec,
        crossfade_ms=args.crossfade_ms,
        inference_noise_scale=args.inference_noise_scale,
        inference_noise_scale_dp=args.inference_noise_scale_dp,
        pro_blend_next_strength=args.pro_blend_next_strength,
    )

    process_jsonl(
        final_dir=args.final_dir,
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        device=args.device,
        ranges=ranges,
        limit=args.limit,
        seed=args.seed,
        deterministic=args.deterministic,
        manifest_path=args.manifest,
        require_continuant_for_pro=args.pro_require_continuant,
    )


if __name__ == "__main__":
    main()