import argparse
import whisperx
from pydub import AudioSegment
from better_profanity import profanity

parser = argparse.ArgumentParser(description="Censor swear words in an audio file")
parser.add_argument("audio_file", type=str, help="Path to the audio file to censor")
parser.add_argument("--output_file", type=str, default="censored_audio.mp3", help="Path to save the censored audio file")
args = parser.parse_args()


device = "cuda" 
audio_file = args.audio_file
output_file = args.output_file
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

segments = result["segments"]

for segment in segments:
    print(segment.keys())


# Initialize an empty list to hold start and end times of swear words
censor_times = []

# Iterate through each segment in the "segments" list
for segment in result["segments"]:
    # Then iterate through each word in the "words" list of each segment
    for word_info in segment["words"]:
        if profanity.contains_profanity(word_info["word"]):
            # Convert start and end times from seconds to milliseconds
            start_ms = int(word_info["start"] * 1000)
            end_ms = int(word_info["end"] * 1000)
            # If the word is a swear word, append its start and end times to the censor_times list
            censor_times.append({"start": start_ms, "end": end_ms})

print(f"Detected {len(censor_times)} swear words in the audio file")

# Load the audio file
audio = AudioSegment.from_file(audio_file)

# Replace each specified segment with silence
for censor in censor_times:
    start_ms = censor["start"]
    end_ms = censor["end"]
    duration_ms = end_ms - start_ms
    silence = AudioSegment.silent(duration=duration_ms)
    audio = audio.overlay(silence, position=start_ms)

# Save the modified audio file
audio.export(output_file, format="mp3")

print(f"Censored audio file has been saved to {output_file}")