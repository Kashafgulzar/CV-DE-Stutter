# CV-DE-Stutter

**The first German dysfluency corpus with word-level, multi-type, verbatim IPA annotation**, along with the full pipeline used to build it.
This repository accompanies an ICASSP submission currently under double-blind review. It contains:
- The **CV-DE-Stutter** dataset (dysfluent IPA transcripts + scripts for synthesized speech)
- Code to reproduce the full pipeline from raw Common Voice audio to trained ASR models
- Configs, tokenizers, and evaluation scripts for the IPA-based ASR models

## Overview

We address the scarcity of fine-grained German stuttering data by:
1. Inserting six co-occurring dysfluency types (word repetitions, sound repetitions, interjections, intra-word pauses, silent blocks, prolongations) into clean eSpeak-ng IPA transcripts from 109 Common Voice German speakers, using an LLM.
2. Finetuning VITS-VCTK into **VITS-CV-DE**, a speaker-adapted German TTS model, to synthesize fluent and stuttered speech in each speaker's own voice.
3. Finetuning a verbatim German IPA ASR model (**OmniASR**) for evaluation.

See the paper for full methodology, evaluation results, and dataset statistics.

## Repository Structure

| Directory | Contents |
|---|---|
| `Preprocessing/` | Common Voice audio filtering, quality control, manifest creation |
| `CV-DE-Stutter/` | Dysfluent IPA generation and parallel fluent/stuttered audio synthesis |
| `VITS-CV-DE/` | Speaker-adapted German TTS finetuning |
| `IPA-ASR/` | OmniASR tokenizers, configs, training data, and evaluation |
| `Data/` | Final released dataset (transcripts, labels, manifest) |

## Pipeline

The stages below mirror the paper's pipeline figure and can be run in order to reproduce the dataset and models from scratch.

**1. Preprocess source audio** (`Preprocessing/`)
Resample Common Voice German audio to 16kHz, filter by clip duration and NISQA-predicted MOS quality, and select the 109 top-contributing speakers to build the source manifest.

**2. Generate dysfluent IPA transcripts** (`CV-DE-Stutter/Text_generation/`)
Convert reference transcripts to clean IPA via eSpeak-ng, then prompt an LLM to insert co-occurring dysfluencies at selected phoneme positions, producing paired dysfluent IPA transcripts and word-level labels.

**3. Finetune the speaker-adapted TTS model** (`VITS-CV-DE/`)
Finetune VITS-VCTK on the 109 CV German speakers, replacing the English speaker embeddings and extending the phoneme inventory to German while keeping the language-agnostic decoder frozen.

**4. Synthesize parallel speech** (`CV-DE-Stutter/Audio_generation/`)
Use the finetuned VITS-CV-DE to synthesize both fluent (CV-DE-Synthetic) and stuttered (CV-DE-Stutter) speech from the same speaker embeddings, yielding three parallel recording types per utterance: real fluent, synthetic fluent, and synthetic stuttered.

**5. Train and evaluate the verbatim IPA ASR** (`IPA-ASR/`)
Build IPA tokenizers, finetune two OmniASR variants (fluent-only vs. fluent + stuttered), and evaluate PER/WER/CER.

## Data

| File | Description |
|---|---|
| `Data/cv_dys.jsonl.gz` | Dysfluent IPA transcripts with word-level, multi-type dysfluency labels |
| `Data/final_manifest.tsv` | Manifest linking audio clips, speakers, and transcript |
| `IPA-ASR/Data/Fluent/` | Train/eval splits used for OmniASR-Fluent |
| `IPA-ASR/Data/Fluent_Stuttered/` | Train/eval splits used for OmniASR-Stutter |
| `IPA-ASR/Data/Stuttered/` | Evaluation-only dysfluent split |

## Models

- **VITS-CV-DE**: Speaker-adapted German TTS, finetuned from VITS-VCTK, used to synthesize the CV-DE-Synthetic and CV-DE-Stutter audio.
- **OmniASR-Fluent**: Verbatim German IPA ASR finetuned on real fluent speech (CV-DE) only.
- **OmniASR-Stutter**: Verbatim German IPA ASR finetuned on real fluent speech plus synthetic stuttered speech (CV-DE-Stutter).

## Status

This repository is released for anonymous peer review. Citation details, author information, and license will be added upon acceptance / de-anonymization.

A demo page with synthetic stuttered speech samples, along with the VITS-CV-DE model checkpoint, will be released on Hugging Face after acceptance.
