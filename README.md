# AutoCensor

AutoCensor detects profanity in spoken audio with WhisperX, then replaces those words with silence. It supports local audio files, local video files, YouTube URLs, and a Gradio web UI for interactive use.

## Features

- Censors profanity in audio and video files
- Preserves the original video stream when processing video
- Supports local file input or YouTube download via `yt-dlp`
- Includes a Gradio UI for browser-based use
- Reuses loaded WhisperX models to speed up repeated runs

## Requirements

System dependencies:

- `ffmpeg`
- `yt-dlp` for URL downloads
- Python 3.10+

Python dependencies are listed in [`requirements.txt`], but you will also need working installs of `torch` and `whisperx`. This repository vendors a `whisperX/` copy for reference, but the runtime imports the installed `whisperx` Python package.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch whisperx
```

If you plan to run on GPU, install the CUDA-compatible PyTorch build for your system before installing or using WhisperX.

## Usage

### Censor a local file

```bash
python main.py --input_file path/to/input.mp3
```

This writes an output file in the current directory named like `input_censored.mp3`.

### Censor a local video and choose the output path

```bash
python main.py --input_file path/to/input.mp4 path/to/output.mp4
```

### Download and censor a YouTube video

```bash
python main.py "https://www.youtube.com/watch?v=..." output.mp4
```

### Launch the Gradio UI

```bash
python main.py --gradio --gradio_port 7860
```

Then open `http://localhost:7860` in your browser.

## CLI options

```text
python main.py [youtube_url] [output_file] [options]
```

Common options:

- `--input_file`: local audio or video file to process
- `--device`: inference device, such as `cuda`, `cuda:0`, or `cpu`
- `--cuda_index`: CUDA device index when using `cuda`
- `--batch_size`: WhisperX transcription batch size
- `--compute_type`: one of `float16`, `int8`, or `int8_float16`
- `--pad_ms`: extra silence added before and after each detected profane word
- `--gradio`: launch the web UI instead of the CLI flow
- `--gradio_port`: port used by the Gradio server

## How it works

1. Audio is loaded directly, or extracted from video with `ffmpeg`.
2. WhisperX transcribes and word-aligns the speech.
3. Each detected profane word is expanded by the configured padding window.
4. Matching ranges are replaced with silence using `pydub`.
5. For video inputs, the censored audio is muxed back into the original video container.

## Notes

- Profanity detection is dictionary-based through `better-profanity`, so some words may be missed and some false positives are possible.
- The default WhisperX model is `large-v3-turbo`.
- The default model cache directory in the code is `/bulk/whisper_models`. Update [`censor.py`] if that path does not exist on your machine.
- The Gradio workflow writes outputs to a temporary directory and returns the processed file for download.

