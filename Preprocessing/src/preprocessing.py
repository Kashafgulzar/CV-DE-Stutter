"""
Filter Common Voice speakers by minimum total hours, then filter out
poor-quality clips using DNSMOS Pro (NISQA checkpoint recommended for
German -- it's the only one of the three released checkpoints trained
on German-language MOS ratings), with GPU + batched inference for speed.

Setup
-----
git clone https://github.com/fcumlin/DNSMOSPro.git
# use the NISQA checkpoint for German:
#   DNSMOSPro/runs/NISQA/model_best.pt
pip install torch librosa soundfile pandas numpy

Usage
-----
python preprocessing.py \
    --speaker_hours cv_data_time_summary.csv \
    --clips_manifest validated.tsv \
    --clips_dir /path/to/cv-corpus/de/clips \
    --dnsmos_checkpoint DNSMOSPro/runs/NISQA/model_best.pt \
    --output usable_speakers.csv \
    --min_hours 1.0 \
    --min_mos 3.0 \
    --device cuda:0 \
    --batch_size 32
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

DNSMOS_SR = 16000
DEFAULT_DURATION_BOUNDS = (1.0, 59.0)


# ---------------------------------------------------------------------------
# Audio loading + feature extraction
# ---------------------------------------------------------------------------

def load_audio(path: Path):
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        return None, None


def probe_duration(path: Path) -> float:
    """Cheap duration lookup from file header only -- no full decode."""
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return -1.0  # sentinel for unreadable; handled downstream


def to_16k_mono(signal: np.ndarray, sr: int) -> np.ndarray:
    if sr != DNSMOS_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=DNSMOS_SR)
    return signal


def log_magnitude_stft(samples: np.ndarray, win_length=320, hop_length=160, n_fft=320) -> np.ndarray:
    """Feature extraction matching what the released DNSMOS Pro checkpoints expect."""
    spec = librosa.stft(y=samples, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
    spec = np.abs(spec).T  # (time, freq)
    spec = np.clip(spec, 1e-7, 1e7)
    spec = np.log10(spec)
    return spec


# ---------------------------------------------------------------------------
# Batched DNSMOS Pro scorer
# ---------------------------------------------------------------------------

class BatchedDNSMOSProScorer:
    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() or "cpu" in device else "cpu")
        self.model = torch.jit.load(checkpoint_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def score_batch(self, specs: list[np.ndarray]) -> list[float]:
        """
        specs: list of (time, freq) log-magnitude spectrograms, possibly different
        lengths along the time axis. We zero-pad to the max length in the batch
        so they can be stacked into a single tensor for one forward pass.
        """
        max_t = max(s.shape[0] for s in specs)
        freq_bins = specs[0].shape[1]

        batch = np.zeros((len(specs), 1, max_t, freq_bins), dtype=np.float32)
        for i, s in enumerate(specs):
            batch[i, 0, : s.shape[0], :] = s

        batch_t = torch.from_numpy(batch).to(self.device)

        with torch.no_grad():
            prediction = self.model(batch_t)
            mean = prediction[:, 0]

        return mean.detach().cpu().numpy().tolist()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter speakers by hours + batched DNSMOS Pro quality (GPU)")
    parser.add_argument("--speaker_hours", required=True)
    parser.add_argument("--clips_manifest", required=True)
    parser.add_argument("--clips_dir", required=True)
    parser.add_argument("--dnsmos_checkpoint", required=True,
                         help="Path to DNSMOS Pro model_best.pt -- use the NISQA checkpoint for German")
    parser.add_argument("--output", default="usable_speakers.csv")
    parser.add_argument("--min_hours", type=float, default=1.0)
    parser.add_argument("--min_mos", type=float, default=3.0)
    parser.add_argument("--min_duration", type=float, default=DEFAULT_DURATION_BOUNDS[0])
    parser.add_argument("--max_duration", type=float, default=DEFAULT_DURATION_BOUNDS[1])
    parser.add_argument("--min_retained_hours", type=float, default=None)
    parser.add_argument("--device", default="cuda:0", help="'cuda:0' or 'cpu' (falls back to cpu if no GPU found)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.min_retained_hours is None:
        args.min_retained_hours = args.min_hours

    clips_dir = Path(args.clips_dir)

    # --- Step 1: speakers with more than min_hours ---
    speaker_hours = pd.read_csv(args.speaker_hours)
    speaker_hours.columns = [c.strip().lower() for c in speaker_hours.columns]
    assert {"client_id", "total_hours"}.issubset(speaker_hours.columns), \
        "speaker_hours.csv must have columns: client_id, total_hours"

    eligible_speakers = set(
        speaker_hours.loc[speaker_hours["total_hours"] > args.min_hours, "client_id"]
    )
    print(f"Speakers with > {args.min_hours}h: {len(eligible_speakers)}")

    # --- Step 2: clips manifest restricted to eligible speakers ---
    sep = "\t" if args.clips_manifest.endswith(".tsv") else ","
    manifest = pd.read_csv(args.clips_manifest, sep=sep)
    manifest.columns = [c.strip().lower() for c in manifest.columns]
    assert {"client_id", "path"}.issubset(manifest.columns), \
        "clips_manifest must have columns: client_id, path"

    manifest = manifest[manifest["client_id"].isin(eligible_speakers)].reset_index(drop=True)
    print(f"Clips to score for eligible speakers: {len(manifest)}")

    # --- Step 2b: probe durations cheaply (header only, no decode) and sort ---
    # This groups similar-length clips into the same batch later, so padding
    # to the max length in a batch wastes far less compute than random order.
    print("Probing clip durations for sorting...")
    manifest["probed_duration"] = manifest["path"].apply(lambda p: probe_duration(clips_dir / p))

    unreadable = manifest[manifest["probed_duration"] < 0]
    for _, row in unreadable.iterrows():
        results_placeholder = None  # filled in below once `results` exists
    manifest = manifest.sort_values("probed_duration", kind="stable").reset_index(drop=True)

    # --- Step 3: batched DNSMOS Pro scoring ---
    scorer = BatchedDNSMOSProScorer(args.dnsmos_checkpoint, device=args.device)
    print(f"Scoring on device: {scorer.device}")

    results = []
    batch_specs, batch_meta = [], []

    def flush_batch():
        if not batch_specs:
            return
        scores = scorer.score_batch(batch_specs)
        for meta, mos in zip(batch_meta, scores):
            client_id, path, duration = meta
            passed = args.min_duration <= duration <= args.max_duration and mos >= args.min_mos
            reason = "ok" if passed else ("low_dnsmos" if mos < args.min_mos else "bad_duration")
            results.append({
                "client_id": client_id, "path": path, "duration_sec": duration,
                "dnsmos": mos, "passed": passed, "reason": reason,
            })
        batch_specs.clear()
        batch_meta.clear()

    n_processed = 0
    for _, row in manifest.iterrows():
        clip_path = clips_dir / row["path"]
        signal, sr = load_audio(clip_path)

        if signal is None or sr is None or signal.size == 0:
            results.append({
                "client_id": row["client_id"], "path": row["path"], "duration_sec": 0.0,
                "dnsmos": 0.0, "passed": False, "reason": "unreadable_or_empty",
            })
            continue

        duration = signal.size / sr
        # skip obviously bad-duration clips before wasting a GPU forward pass
        if duration < args.min_duration or duration > args.max_duration:
            results.append({
                "client_id": row["client_id"], "path": row["path"], "duration_sec": duration,
                "dnsmos": 0.0, "passed": False, "reason": "bad_duration",
            })
            continue

        signal = to_16k_mono(signal, sr)
        spec = log_magnitude_stft(signal)

        batch_specs.append(spec)
        batch_meta.append((row["client_id"], row["path"], duration))

        if len(batch_specs) >= args.batch_size:
            flush_batch()

        n_processed += 1
        if n_processed % 2000 == 0:
            print(f"  processed {n_processed}/{len(manifest)} clips...")

    flush_batch()  # remaining partial batch

    clip_results = pd.DataFrame(results)
    clip_results.to_csv("clip_quality_report.csv", index=False)

    passed_clips = clip_results[clip_results["passed"]]
    print(f"Clips passing DNSMOS Pro >= {args.min_mos}: {len(passed_clips)} / {len(clip_results)}")

    # --- Step 4: recompute retained hours per speaker ---
    retained = (
        passed_clips.groupby("client_id")["duration_sec"]
        .sum()
        .div(3600.0)
        .rename("usable_hours")
        .reset_index()
    )
    clip_counts = passed_clips.groupby("client_id").size().rename("num_usable_clips").reset_index()
    avg_mos = passed_clips.groupby("client_id")["dnsmos"].mean().rename("avg_dnsmos").reset_index()

    retained = retained.merge(clip_counts, on="client_id").merge(avg_mos, on="client_id")

    final = retained[retained["usable_hours"] > args.min_retained_hours].sort_values(
        "usable_hours", ascending=False
    )

    final.to_csv(args.output, index=False)
    print(f"\nFinal usable speakers: {len(final)}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()