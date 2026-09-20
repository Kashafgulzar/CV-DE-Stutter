import os
import pandas as pd
import torchaudio
from tqdm import tqdm


# --------- CONFIG ---------
final_manifest_tsv = "/path/to/final_manifest.tsv"

audio_dir = "/path/to/cv-corpus-13.0-2023-03-09/de/clips"   # directory containing original audio files
output_dir = "./Resampled_CV"

target_sr = 22050

os.makedirs(output_dir, exist_ok=True)


# --------- LOAD MANIFEST ---------
df = pd.read_csv(final_manifest_tsv, sep="\t")

print("Columns:", df.columns)

# assuming your TSV has a column called "path"
assert "path" in df.columns, "TSV must contain a 'path' column"


# --------- RESAMPLER CACHE ---------
resamplers = {}


def resample_audio(audio_path, save_path):

    waveform, sr = torchaudio.load(audio_path)

    # convert stereo -> mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # resample only if needed
    if sr != target_sr:

        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )

        waveform = resamplers[sr](waveform)

    # save wav
    torchaudio.save(
        save_path,
        waveform,
        target_sr,
        encoding="PCM_S",
        bits_per_sample=16
    )


# --------- PROCESS FILES ---------
for _, row in tqdm(df.iterrows(), total=len(df)):

    relative_path = row["path"]

    input_audio = os.path.join(
        audio_dir,
        relative_path
    )

    if not os.path.exists(input_audio):
        print(f"Missing: {input_audio}")
        continue


    # keep same filename
    filename = os.path.splitext(os.path.basename(relative_path))[0] + ".wav"

    output_audio = os.path.join(
        output_dir,
        filename
    )


    resample_audio(
        input_audio,
        output_audio
    )


print("Finished resampling!")