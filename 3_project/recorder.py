import sys
import os
import pyaudio
import wave
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from ASR_trans import get_pred
from scipy.signal import resample
from ignore_warning import ignore_warning, hide_print
import warnings
warnings.filterwarnings("ignore")

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 512
RECORD_SECONDS = 10
MIC_INDEX = 11


def record_audio():
    with ignore_warning():
        audio = pyaudio.PyAudio()
    
    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, input_device_index=MIC_INDEX,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    print("Finished recording")

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return b''.join(frames)

def save_audio(frames, filename):
    with ignore_warning():
        audio = pyaudio.PyAudio()
    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(frames)
    waveFile.close()

def process_audio(input_filename, output_filename, target_rate=16000):
    # Load the audio file
    data, rate = sf.read(input_filename, dtype='int16')
    
    # Resample the audio data to the target rate
    resampled_data = librosa.resample(data.astype(np.float32), orig_sr=rate, target_sr=target_rate)
    
    # Apply dynamic range compression
    compressed_data = librosa.effects.preemphasis(resampled_data, coef=0.97)

    # Apply noise reduce
    compressed_data = nr.reduce_noise(y=compressed_data, sr=target_rate)
    
    # Normalize the audio to maximize peak value
    peak = np.max(np.abs(compressed_data))
    if peak > 0:
        compressed_data = compressed_data * (0.99 / peak)  # Using 0.99 to avoid clipping
    
    # Convert back to int16 range
    compressed_data = np.int16(compressed_data * 32767)
    
    # Save the resampled, filtered, and normalized audio data to a new file
    sf.write(output_filename, compressed_data, target_rate)


def main():
    while True:
        input("Press Enter to start recording...")
        
        # Record audio
        frames = record_audio()
        print("Audio recorded")
        
        # Save the recorded audio
        original_filename = "recording.wav"
        save_audio(frames, original_filename)
        print(f"Original audio saved as {original_filename}")
        
        # Resample the audio and save with a new filename
        resampled_filename = "resampled_recording.flac"
        process_audio(original_filename, resampled_filename)
        print(f"Resampled audio saved as {resampled_filename}")

        # Predict the words of the audio
        print("Predicting words...")
        with hide_print():
            pred_words=get_pred(resampled_filename)
        print(f"Predicted words: {pred_words}")

if __name__ == "__main__":
    main()

