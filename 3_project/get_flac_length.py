import os
import soundfile as sf
# Get the length of the .flac file


def get_flac_length(flac_file_path):
    info = sf.info(flac_file_path)
    return info.frames / info.samplerate


if __name__ == "__main__":
    flac_file_path = "04.flac"
    print(get_flac_length(flac_file_path))
