#!/home/john/Programs/miniconda3/envs/autocensor/bin/python3

import argparse
import mimetypes
import subprocess
import tempfile
from pathlib import Path

import gradio as gr
import whisperx
from better_profanity import profanity
from pydub import AudioSegment


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def is_video_file(path: Path) -> bool:
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return True
    mime, _ = mimetypes.guess_type(path.as_posix())
    return bool(mime and mime.startswith("video/"))


def extract_audio(video_path: Path, wav_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path.as_posix(),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            wav_path.as_posix(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path.as_posix(),
            "-i",
            audio_path.as_posix(),
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            output_path.as_posix(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def censor_audio_segments(
    audio_path: Path,
    output_path: Path,
    device: str,
    batch_size: int,
    compute_type: str,
    pad_ms: int,
) -> None:
    model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type, download_root="/bulk/whisper_models")
    audio = whisperx.load_audio(audio_path.as_posix())
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)

    censor_times = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            if profanity.contains_profanity(word.get("word", "")):
                start_ms = int(word["start"] * 1000) - pad_ms
                end_ms = int(word["end"] * 1000) + pad_ms
                if start_ms < 0:
                    start_ms = 0
                censor_times.append({"start": start_ms, "end": end_ms})

    print(f"Detected {len(censor_times)} swear words in the audio file")

    audio_segment = AudioSegment.from_file(audio_path.as_posix())
    audio_length_ms = len(audio_segment)
    for censor in censor_times:
        start_ms = censor["start"]
        end_ms = min(censor["end"], audio_length_ms)
        if end_ms <= start_ms:
            continue
        silence = AudioSegment.silent(duration=end_ms - start_ms)
        audio_segment = audio_segment[:start_ms] + silence + audio_segment[end_ms:]
        print(f"Censored audio at time {(start_ms//60000)%60:02d}:{(start_ms//1000)%60:02d}")

    output_format = output_path.suffix.lstrip(".") or "mp3"
    audio_segment.export(output_path.as_posix(), format=output_format)


def default_output_path(input_path: Path, is_video: bool) -> Path:
    suffix = input_path.suffix or (".mp4" if is_video else ".mp3")
    return input_path.with_name(f"{input_path.stem}_censored{suffix}")


def censor_media_file(
    input_path: Path,
    output_path: Path,
    device: str,
    batch_size: int,
    compute_type: str,
    pad_ms: int,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        treat_as_video = is_video_file(input_path)
        if treat_as_video:
            extracted_audio = tmp_dir_path / "extracted_audio.wav"
            censored_audio = tmp_dir_path / "censored_audio.wav"
            extract_audio(input_path, extracted_audio)
            censor_audio_segments(
                extracted_audio,
                censored_audio,
                device,
                batch_size,
                compute_type,
                pad_ms,
            )
            mux_audio(input_path, censored_audio, output_path)
        else:
            censor_audio_segments(
                input_path,
                output_path,
                device,
                batch_size,
                compute_type,
                pad_ms,
            )


def download_youtube(url: str, output_dir: Path) -> Path:
    output_template = output_dir / "youtube_download.%(ext)s"
    subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bv*+ba/best",
            "-o",
            output_template.as_posix(),
            url,
        ],
        check=True,
    )
    matches = sorted(output_dir.glob("youtube_download.*"))
    if not matches:
        raise FileNotFoundError("yt-dlp did not produce an output file")
    return matches[0]


def build_gradio_interface() -> gr.Blocks:
    def run_censor(
        input_file: str,
        input_url: str,
        output_name: str,
        device: str,
        batch_size: int,
        compute_type: str,
        pad_ms: int,
    ) -> str:
        if not input_file and not input_url:
            raise gr.Error("Upload a file or paste a URL.")
        if input_file and input_url:
            raise gr.Error("Provide only one input: file or URL.")

        output_dir = Path(tempfile.mkdtemp(prefix="autocensor_"))
        if input_url:
            input_path = download_youtube(input_url.strip(), output_dir)
        else:
            input_path = Path(input_file).expanduser().resolve()
            if not input_path.exists():
                raise gr.Error("Uploaded file could not be found on disk.")

        suffix = input_path.suffix or (".mp4" if is_video_file(input_path) else ".mp3")
        if output_name and output_name.strip():
            output_name_clean = Path(output_name.strip()).name
            if not Path(output_name_clean).suffix:
                output_name_clean = f"{output_name_clean}{suffix}"
            output_path = output_dir / output_name_clean
        else:
            output_path = output_dir / f"{input_path.stem}_censored{suffix}"

        censor_media_file(
            input_path,
            output_path,
            device,
            batch_size,
            compute_type,
            pad_ms,
        )
        return output_path.as_posix()

    with gr.Blocks(title="AutoCensor") as demo:
        gr.Markdown("# AutoCensor\nUpload a video to censor profanity and download the result.")
        with gr.Row():
            input_file = gr.File(
                label="Video or audio file",
                type="filepath",
            )
        with gr.Row():
            input_url = gr.Textbox(
                label="Video URL (YouTube supported)",
                placeholder="https://www.youtube.com/watch?v=...",
            )
        with gr.Row():
            output_name = gr.Textbox(
                label="Output file name (optional)",
                placeholder="my_censored_video.mp4",
            )
        with gr.Row():
            device = gr.Dropdown(
                choices=["cuda", "cpu"],
                value="cuda",
                label="Device",
            )
            batch_size = gr.Slider(
                minimum=1,
                maximum=64,
                step=1,
                value=16,
                label="Batch size",
            )
            compute_type = gr.Dropdown(
                choices=["float16", "int8", "int8_float16"],
                value="float16",
                label="Compute type",
            )
            pad_ms = gr.Slider(
                minimum=0,
                maximum=1000,
                step=50,
                value=200,
                label="Padding (ms)",
            )
        output_file = gr.File(label="Censored output", type="filepath")
        run_button = gr.Button("Censor and prepare download")
        run_button.click(
            run_censor,
            inputs=[input_file, input_url, output_name, device, batch_size, compute_type, pad_ms],
            outputs=output_file,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Censor swear words in an audio or video file")
    parser.add_argument(
        "youtube_url",
        type=str,
        nargs="?",
        help="YouTube URL to download and censor (default input type)",
    )
    parser.add_argument(
        "output_file",
        type=str,
        nargs="?",
        help="Path to save the censored output",
    )
    parser.add_argument("--input_file", type=str, nargs="?", help="Path to the audio or video file to censor")
    parser.add_argument("--device", type=str, default="cuda", help="Device for whisperx (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for whisperx")
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="Compute type for whisperx (float16 or int8)",
    )
    parser.add_argument(
        "--pad_ms",
        type=int,
        default=200,
        help="Milliseconds to pad before and after each detected word",
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch a Gradio web UI for LAN use",
    )
    parser.add_argument(
        "--gradio_port",
        type=int,
        default=7860,
        help="Port for the Gradio server",
    )
    args = parser.parse_args()

    if args.gradio:
        demo = build_gradio_interface()
        demo.launch(server_name="0.0.0.0", server_port=args.gradio_port)
        return

    if args.input_file and args.youtube_url and args.output_file is None:
        args.output_file = args.youtube_url
        args.youtube_url = None

    if not args.input_file and not args.youtube_url:
        raise ValueError("Provide either input_file or youtube_url")
    if args.input_file and args.youtube_url:
        raise ValueError("Provide only one of input_file or youtube_url")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        if args.input_file:
            input_path = Path(args.input_file).expanduser().resolve()
            if not input_path.exists():
                raise FileNotFoundError(f"Input file does not exist: {input_path}")
        else:
            input_path = download_youtube(args.youtube_url, tmp_dir_path)

        treat_as_video = is_video_file(input_path)
        output_path = (
            Path(args.output_file).expanduser().resolve()
            if args.output_file
            else default_output_path(input_path, treat_as_video)
        )

        censor_media_file(
            input_path,
            output_path,
            args.device,
            args.batch_size,
            args.compute_type,
            args.pad_ms,
        )
        if treat_as_video:
            print(f"Censored video file has been saved to {output_path}")
        else:
            print(f"Censored audio file has been saved to {output_path}")


if __name__ == "__main__":
    main()
