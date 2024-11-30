#running code
#code which has generated transcripts for 50 rapaport podcasta .mp3 files and saved into output folder
import torch
from transformers import pipeline
from pydub import AudioSegment
import os

# Initialize Whisper pipeline for transcription
device = 0 if torch.cuda.is_available() else -1
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    device=device
)

# Paths
audio_folder = "/mnt/c/Users/mayur/Desktop/Dal - Fall 2024/CSCI 6518 - Deep Speech/Project/Data Set Analysis/Data-Set/Audio/Rapaport Podcasts/Audio"
output_folder = "/mnt/c/Users/mayur/Desktop/Dal - Fall 2024/CSCI 6518 - Deep Speech/Project/Data Set Analysis/Data-Set/Audio/Rapaport Podcasts/Transcripts"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to process audio and transcribe directly
def transcribe_audio(audio_path, chunk_length_ms=30000):
    audio = AudioSegment.from_file(audio_path)
    full_transcription = ""
    
    # Process each chunk one at a time
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        temp_file = f"temp_chunk.wav"
        chunk.export(temp_file, format="wav")
        
        # Transcribe the chunk
        result = transcriber(temp_file)
        full_transcription += result["text"] + " "
        
        # Remove the temporary file
        os.remove(temp_file)
    
    return full_transcription

# Process each audio file in the folder
for k, filename in enumerate(os.listdir(audio_folder)):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        audio_path = os.path.join(audio_folder, filename)
        
        # Transcribe the audio file
        try:
            transcription = transcribe_audio(audio_path)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            
            # Save transcription to text file
            with open(output_path, "w") as f:
                f.write(transcription)
            
            # Print status
            print(f"{k + 1} - File Transcription done for {filename}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Rapaport Podcasts Transcriptions completed and saved to text files.")
