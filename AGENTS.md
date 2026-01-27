# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the primary CLI/Gradio entrypoint; it drives download, transcription, and censoring.
- `autocensor/` contains the Python package namespace (currently minimal scaffolding).
- `whisperX/` is a vendored dependency with its own build metadata and docs; avoid editing unless updating the upstream snapshot.
- `tmp/` is a local scratch area (ignored by Git).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates a virtualenv for local work.
- `pip install -r requirements.txt` installs runtime dependencies.
- `python main.py --input_file path/to/audio.mp3` censors a local file.
- `python main.py https://www.youtube.com/watch?v=... output.mp4` downloads and censors a YouTube video.
- `python main.py --gradio --gradio_port 7860` launches the Gradio UI.

System tools required: `ffmpeg` for audio/video processing and `yt-dlp` for URL downloads. Whisper models are downloaded via `whisperx` (see `main.py` for the cache path).

## Coding Style & Naming Conventions
- Use 4-space indentation and PEP 8 naming: `snake_case` for functions/variables, `PascalCase` for classes.
- Prefer clear, explicit names for media paths (e.g., `input_path`, `output_path`).
- Keep side effects isolated (e.g., file I/O, subprocess calls) and guard user input early.

## Testing Guidelines
- No test suite is currently defined. If you add tests, keep them in `tests/` with `test_*.py` names and document how to run them (e.g., `pytest`).
- For manual checks, verify both `--input_file` and YouTube URL flows and the `--gradio` UI.

## Commit & Pull Request Guidelines
- Commit messages in history are short, sentence-style summaries (e.g., "Added requirements"). Follow that pattern.
- PRs should include: a clear summary, the exact command(s) used to verify behavior, and example input/output filenames when touching media handling.

## Configuration & Security Notes
- Avoid committing large media files or model weights; use local paths or temporary directories.
- Validate any new CLI inputs that affect filesystem paths or subprocess calls.
