import glob
import os
import librosa
import numpy as np
import soundfile as sf
from librosa import lpc
import numpy as np
# from aubio import source, pitch
# from torchvggish import vggish, vggish_input
from speech_split import split_audio_to_utterances, apply_vad
from matplotlib import pyplot as plt
from scipy.signal import get_window
from scipy.fftpack import fft
import librosa.display
from scipy.signal import butter, filtfilt
import noisereduce as nr
import matplotlib.pyplot as plt


def compute_mfcc(data, sr, n_mfcc=13, n_mels=40, fmin=0, fmax=None):
    """
    Compute the MFCCs for a given audio data.

    Parameters:
    - data (numpy.ndarray): The audio signal from which to compute features. Should be one-dimensional (1D).
    - sr (int): The sampling rate of the audio signal.
    - n_mfcc (int, optional): Number of MFCCs to return.
    - n_mels (int, optional): Number of Mel bands to generate.
    - fmin (float, optional): Lowest frequency (in Hz).
    - fmax (float, optional): Highest frequency (in Hz). If `None`, use fmax = sr / 2.0

    Returns:
    - numpy.ndarray: MFCC sequence.
    """
    
    # Ensure the audio signal is one-dimensional
    if len(data.shape) > 1:
        raise ValueError("The input data should be one-dimensional (1D).")

    # Compute the MFCCs using librosa
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, fmin=fmin, fmax=fmax)
    
    return mfccs


def plot_stft_colormap(stft_matrix, filename):
    """
    Visualize an STFT matrix (in dB) using a colormap and save it to a file.

    Parameters:
    - stft_matrix (numpy.ndarray): The STFT matrix (in dB) to visualize. Typically, rows represent frequencies and columns represent time frames.
    - filename (str): The path and filename where the image will be saved.

    Returns:
    - None (saves a plot to a file).
    """

    # Since the STFT is already in dB scale, no need to compute magnitude spectrum in log scale again.

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(stft_matrix, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bin')
    plt.title('STFT Magnitude Spectrum (dB)')
    plt.tight_layout()
    
    # Save the plot to the specified filename
    plt.savefig(filename)

    # Close the plot to free up resources
    plt.close()




def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def generate_white_noise(num_samples):
    noise = np.random.normal(0, 1, num_samples)
    return noise


def split_stft_matrix(stft_matrix, num_frames):
    """
    Split a matrix of STFT into a list of STFT matrices of given frame length.

    Parameters:
        stft_matrix (numpy.ndarray): Input STFT matrix with shape (num_freq_bins, num_frames).
        num_frames (int): Number of frames in each STFT matrix.

    Returns:
        List[numpy.ndarray]: A list of STFT matrices, where each matrix has shape (num_freq_bins, num_frames).
    """
    num_freq_bins, total_frames = stft_matrix.shape

    # Compute the number of STFT matrices we can create
    num_splits = total_frames // num_frames

    # Initialize an empty list to store the split STFT matrices
    stft_matrices = []

    # Split the STFT matrix into sub-matrices of the specified frame length
    for i in range(num_splits):
        start_frame = i * num_frames
        end_frame = start_frame + num_frames
        stft_matrix_split = stft_matrix[:, start_frame:end_frame]
        stft_matrices.append(stft_matrix_split)

    return stft_matrices


def get_mel_spectrogram(audio, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=256, fmin=10, fmax=8000):
    """
    Calculate the mel spectrogram of an audio signal.

    Parameters:
        audio (numpy array): 1D array representing the audio signal.
        sample_rate (int, optional): The sample rate of the audio (default is 22050).
        n_fft (int, optional): The number of FFT points (default is 2048).
        hop_length (int, optional): The number of samples between successive frames (default is 512).
        n_mels (int, optional): The number of mel frequency bins (default is 128).

    Returns:
        numpy array: The mel spectrogram of the audio signal.
    """
    # Calculate the mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                                     n_mels=n_mels, fmin=fmin, fmax=fmax)

    lists = split_stft_matrix(mel_spectrogram, num_frames=50)

    return lists


def get_stft(audio, sample_rate=44100, n_fft=2048, hop_length=512, n_frames=128):
    """
    Calculate the mel spectrogram of an audio signal.

    Parameters:
        audio (numpy array): 1D array representing the audio signal.
        sample_rate (int, optional): The sample rate of the audio (default is 22050).
        n_fft (int, optional): The number of FFT points (default is 2048).
        hop_length (int, optional): The number of samples between successive frames (default is 512).
        n_mels (int, optional): The number of mel frequency bins (default is 128).

    Returns:
        numpy array: The mel spectrogram of the audio signal.
    """
    # Calculate the mel spectrogram using librosa
    mel_spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                                   win_length=n_fft, window='hann')

    lists = split_stft_matrix(mel_spectrogram, num_frames=n_frames)

    return lists


def find_wav_files(directory):
    wav_files = []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".wav")]:
            wav_files.append(os.path.join(dirpath, filename))

    return wav_files


voxceleb = './voxceleb1/wav/'
speakers_list = [d for d in os.listdir(voxceleb) if os.path.isdir(os.path.join(voxceleb, d))]
print(len(speakers_list))

vad_level = 2
frame_duration = 30  # Duration of each frame in ms
min_speech_duration = 1800
# model = vggish()

labels = []
file_dir_name = './processed_data/voxceleb/res_256/'
os.makedirs(file_dir_name, exist_ok=True)

for i in range(len(speakers_list)):
    speaker = speakers_list[i]
    print(speaker)
    speaker_root = voxceleb + speaker + '/'
    wavs = find_wav_files(speaker_root)
    print(len(wavs))


    audios = []
    for wav in wavs:
        audio, sr = sf.read(wav)

        speech_labels = apply_vad(audio, sr, vad_level, frame_duration)
        speech_utterances = split_audio_to_utterances(audio, sr, speech_labels, frame_duration, min_speech_duration)
        for utterance in speech_utterances:
            audio_piece = audio[utterance[0]: utterance[1]]
            bins_audio = get_stft(audio_piece, 16000, 512, 256, 128)
            audios += bins_audio

    os.makedirs(file_dir_name+str(i)+'/', exist_ok=True)
    for j in range(len(audios)):
        audios[j] = 10 * np.log10(np.abs(audios[j]) ** 2 + 1e-6)
        audios[j] = (audios[j] - np.min(audios[j])) / (np.max(audios[j]) - np.min(audios[j]))

        np.save(file_dir_name+str(i)+'/'+'{}_{}_audio.npy'.format(i, j), audios[j][: 256, :])
        print('user {} utterance {}'.format(i, j))
    # audios = np.array(audios)
    # audios = audios[:, :256, :]
    # print(audios.shape)
    # file_dir_name = './processed_data/res_512/'
    #file_dir_name = './processed_data/res_512a_512p/'
    
    
    # for j in range(audios.shape(0)):
    #     np.save(file_dir_name+'{}_{}_audio.npy'.format(i, j), audios[j])
    # print(len(speech_utterances))