"""
synthesize_fluent_audio.py

Generates clean, fluent speech audio from IPA transcripts using the exact same
VITS low-level inference architecture as the dysfluent synthesis pipeline.
Bypasses high-level text cleaners to ensure faithful character-to-ID mapping.

Usage: python synthesize_fluent_audio.py \
    --jsonl "path/to/cv/jsonl" \
    --final_dir "path/to/VITS-CV-DE" \
    --output_dir "Synthetic_Fluent_CV" \
    --seed 1234 \
    --device "cuda" \
    --inference_noise_scale 0.50 \
    --inference_noise_scale_dp 0.60
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

logger = logging.getLogger("fluent_synth")


# --------------------------------------------------------------------------- #
# Tokenization & Character Mapping
# --------------------------------------------------------------------------- #

def encode_ipa_text(tokenizer: Any, clean_text: str) -> list[int]:
    """
    Directly maps IPA characters to model IDs using the tokenizer's internal 
    char_to_id mapping, matching the dysfluent synthesis character mapping logic.
    Bypasses standard TTS text cleaners that mangle IPA sequences.
    """
    ids: list[int] = []
    for ch in clean_text:
        try:
            tid = tokenizer.characters.char_to_id(ch)
            ids.append(tid)
        except KeyError:
            continue

    if getattr(tokenizer, "add_blank", False):
        ids = tokenizer.intersperse_blank_char(ids, True)
    if getattr(tokenizer, "use_eos_bos", False):
        ids = tokenizer.pad_with_bos_eos(ids)
    return ids


# --------------------------------------------------------------------------- #
# Model Loading & Reproducibility
# --------------------------------------------------------------------------- #

@dataclass
class ModelBundle:
    model: Any
    config: Any
    tokenizer: Any
    speaker_manager: Any
    device: torch.device
    sample_rate: int


def load_bundle(final_dir: str, device_str: str = "auto") -> ModelBundle:
    """Loads VITS checkpoint, config, and speaker mappings from final_dir."""
    from TTS.config import load_config
    from TTS.tts.models.vits import Vits

    final_root = Path(final_dir)
    config_path = final_root / "config.json"
    speakers_path = final_root / "speakers.json"
    
    ckpt_path = final_root / "checkpoint.pth"
    if not ckpt_path.exists():
        ckpt_path = final_root / "best_model.pth"
    if not ckpt_path.exists():
        ckpt_path = final_root / "model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No valid checkpoint (.pth) found in {final_root}")

    logger.info("Loading config from %s", config_path)
    config = load_config(str(config_path))

    if speakers_path.exists():
        config.speakers_file = str(speakers_path)

    logger.info("Initializing VITS model from checkpoint: %s", ckpt_path)
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, str(ckpt_path), eval=True)

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        config=config,
        tokenizer=model.tokenizer,
        speaker_manager=model.speaker_manager,
        device=device,
        sample_rate=config.audio["sample_rate"],
    )


def resolve_speaker_id(bundle: ModelBundle, client_id: str) -> int:
    """Resolves client_id string to internal speaker ID integer."""
    sm = bundle.speaker_manager
    if sm is not None and getattr(sm, "name_to_id", None) and client_id in sm.name_to_id:
        return sm.name_to_id[client_id]
    raise KeyError(f"client_id '{client_id}' not found in speaker manager mapping.")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Sets random seeds for reproducibility across CPU/CUDA execution."""
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
# Core VITS Inference Routine
# --------------------------------------------------------------------------- #

def run_vits_fluent_inference(
    bundle: ModelBundle,
    ids: list[int],
    speaker_id: int,
    noise_scale: float | None = None,
    noise_scale_dp: float | None = None,
    length_scale: float = 1.0,
) -> np.ndarray:
    """
    Runs low-level VITS forward pass directly over token IDs.
    Calculates durations using duration predictor without manual path manipulation.
    """
    from TTS.tts.utils.helpers import generate_path, sequence_mask

    model = bundle.model
    device = bundle.device

    x = torch.LongTensor(ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(ids)]).to(device)
    sid = torch.LongTensor([speaker_id]).to(device)

    if noise_scale is None:
        noise_scale = getattr(model, "inference_noise_scale", 1.0)
    if noise_scale_dp is None:
        noise_scale_dp = getattr(model, "inference_noise_scale_dp", 1.0)

    with torch.no_grad():
        g = None
        if getattr(model.args, "use_speaker_embedding", False) and sid is not None:
            g = model.emb_g(sid).unsqueeze(-1)

        x_enc, m_p, logs_p, x_mask = model.text_encoder(x, x_lengths, lang_emb=None)

        dp_g = g if getattr(model.args, "condition_dp_on_speaker", True) else None
        if getattr(model.args, "use_sdp", False):
            logw = model.duration_predictor(x_enc, x_mask, g=dp_g, reverse=True, noise_scale=noise_scale_dp)
        else:
            logw = model.duration_predictor(x_enc, x_mask, g=dp_g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)
        attn_mask = x_mask * y_mask.transpose(1, 2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p_exp = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_exp = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale
        z = model.flow(z_p, y_mask, g=g, reverse=True)
        z, _, _, y_mask = model.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        max_len = getattr(model, "max_inference_len", None)
        o = model.waveform_decoder((z * y_mask)[:, :, :max_len], g=g)

    return o.squeeze().detach().cpu().float().numpy()


# --------------------------------------------------------------------------- #
# Batch Processing & Manifest Pipeline
# --------------------------------------------------------------------------- #

def process_fluent_jsonl(
    final_dir: str,
    jsonl_path: str,
    output_dir: str,
    text_field: str = "ipa_correct",
    device: str = "auto",
    limit: int | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    noise_scale: float | None = None,
    noise_scale_dp: float | None = None,
    length_scale: float = 1.0,
) -> None:
    """Iterates over input JSONL and generates fluent audio."""
    bundle = load_bundle(final_dir, device)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if limit is not None:
        lines = lines[:limit]

    n_ok, n_fail = 0, 0

    for i, item in enumerate(tqdm(lines, desc="Synthesizing Fluent Audio")):
        client_id = item["client_id"]
        rel_path = item["path"]
        ipa_text = item.get(text_field)

        if not ipa_text:
            logger.warning("Item %d missing target text field '%s'. Skipping.", i, text_field)
            n_fail += 1
            continue

        out_path = out_root / rel_path
        if out_path.suffix.lower() != ".wav":
            out_path = out_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            speaker_id = resolve_speaker_id(bundle, client_id)
            if seed is not None:
                set_seed(seed + i, deterministic=deterministic)

            ids = encode_ipa_text(bundle.tokenizer, ipa_text)
            waveform = run_vits_fluent_inference(
                bundle,
                ids,
                speaker_id,
                noise_scale=noise_scale,
                noise_scale_dp=noise_scale_dp,
                length_scale=length_scale,
            )

            sf.write(str(out_path), waveform, bundle.sample_rate)
            n_ok += 1

        except Exception as e:
            n_fail += 1
            logger.exception("Failed on item %d (client_id=%s, path=%s): %s", i, client_id, rel_path, e)

    logger.info("Done. %d succeeded, %d failed.", n_ok, n_fail)


# --------------------------------------------------------------------------- #
# CLI Interface
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synthesizes fluent reference audio from IPA transcripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--final_dir", required=True, help="Directory containing model checkpoints, config.json, etc.")
    p.add_argument("--jsonl", required=True, help="Path to input jsonl file with IPA transcripts")
    p.add_argument("--output_dir", required=True, help="Directory where generated wav files will be saved")
    p.add_argument("--text_field", default="ipa_correct", help="Field key in JSONL containing clean IPA (default: ipa_correct)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    p.add_argument("--seed", type=int, default=None, help="Base seed (per-item seed = seed + row index)")
    p.add_argument("--deterministic", action="store_true", help="Force deterministic execution algorithms")
    p.add_argument("--inference_noise_scale", type=float, default=None, help="Override z_p sampling noise scale")
    p.add_argument("--inference_noise_scale_dp", type=float, default=None, help="Override duration predictor noise scale")
    p.add_argument("--length_scale", type=float, default=1.0, help="Global speaking rate modifier (1.0 = normal speed)")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_arg_parser().parse_args()

    process_fluent_jsonl(
        final_dir=args.final_dir,
        jsonl_path=args.jsonl,
        output_dir=args.output_dir,
        text_field=args.text_field,
        device=args.device,
        limit=args.limit,
        seed=args.seed,
        deterministic=args.deterministic,
        noise_scale=args.inference_noise_scale,
        noise_scale_dp=args.inference_noise_scale_dp,
        length_scale=args.length_scale,
    )


if __name__ == "__main__":
    main()