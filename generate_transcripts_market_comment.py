#running code
#code which has generated transcripts for 26 youtube market comment .mp4 files and store into output folder
import torch
from transformers import pipeline
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os

# Initialize Whisper pipeline for transcription
device = 0 if torch.cuda.is_available() else -1
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    device=device
)

# Paths
video_folder = "/mnt/c/Users/mayur/Desktop/Dal - Fall 2024/CSCI 6518 - Deep Speech/Project/Data Set Analysis/Data-Set/Audio/Youtube market Analysis/Video"  # Folder with video files
output_folder = "/mnt/c/Users/mayur/Desktop/Dal - Fall 2024/CSCI 6518 - Deep Speech/Project/Data Set Analysis/Data-Set/Audio/Youtube market Analysis/Transcripts"  # Folder to save text files

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to extract audio from video file and split into chunks
def transcribe_video(video_path, chunk_length_ms=30000):
    # Extract audio from video
    with VideoFileClip(video_path) as video:
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)
    
    # Load audio and prepare for transcription
    audio = AudioSegment.from_file(audio_path)
    full_transcription = ""
    
    # Process each chunk one at a time
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        temp_chunk_file = "temp_chunk.wav"
        chunk.export(temp_chunk_file, format="wav")
        
        # Transcribe the chunk
        result = transcriber(temp_chunk_file)
        full_transcription += result["text"] + " "
        
        # Remove the temporary chunk file
        os.remove(temp_chunk_file)
    
    # Remove the temporary full audio file
    os.remove(audio_path)
    
    return full_transcription

# Process each video file in the folder
for k, filename in enumerate(os.listdir(video_folder)):
    if filename.endswith(".mp4"):
        video_path = os.path.join(video_folder, filename)
        
        # Transcribe the video file
        try:
            transcription = transcribe_video(video_path)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            
            # Save transcription to text file
            with open(output_path, "w") as f:
                f.write(transcription)
            
            # Print status
            print(f"{k + 1} - File Transcription done for {filename}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Youtube market comment Transcriptions completed and saved to text files.")
