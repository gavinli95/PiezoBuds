import os
from pydub import AudioSegment

def get_audio_length(file_path):
    audio = AudioSegment.from_wav(file_path)

    return len(audio) / 1000.0

def main(folder_path):
    total_length = 0

    for i in range(70):  # Since your folders are named 0 to 69
        subfolder_path = os.path.join(folder_path, str(i))
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(subfolder_path, file_name)
                total_length += get_audio_length(file_path)

    return total_length


if __name__ == "__main__":

    folder_path = '/mnt/ssd/gen/piezo_authentication/data/'
    print("Total length of audio files:", main(folder_path), "seconds")