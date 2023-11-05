import os
import librosa
import numpy as np
from speech_split import split_audio_to_utterances, apply_vad
from matplotlib import pyplot as plt
import scipy.signal as signal
from scipy.fftpack import fft
import librosa.display
import scipy
from scipy.signal import butter, filtfilt
import noisereduce as nr
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav_writer


def slice_audio(audio_data, sample_rate, window_size_ms):
    """
    Slices the audio data into clips of the specified window size. If the last clip
    is shorter than the window size, it is padded with the preceding audio to make
    it full length.
    
    :param audio_data: 1D NumPy array of audio data.
    :param sample_rate: Integer, the sample rate of the audio data.
    :param window_size_ms: Integer, the window size in milliseconds.
    :return: List of 1D NumPy arrays, each representing a clip of the specified window size.
    """
    # Calculate the window size in samples
    window_size_samples = int((window_size_ms / 1000) * sample_rate)
    
    # Calculate the number of clips to be created
    num_clips = int(np.ceil(len(audio_data) / window_size_samples))
    
    # Slice the audio and collect clips
    audio_clips = []
    for i in range(num_clips):
        start_index = i * window_size_samples
        end_index = start_index + window_size_samples
        if end_index > len(audio_data):
            # If we're beyond the end of the audio, pad the last clip
            last_clip = audio_data[start_index:]
            padding_needed = window_size_samples - len(last_clip)
            # Pad with the preceding audio
            padded_clip = np.concatenate((audio_data[-(padding_needed + len(last_clip)):-len(last_clip)], last_clip))
            audio_clips.append(padded_clip)
        else:
            # Otherwise, just append the clip
            audio_clips.append(audio_data[start_index:end_index])
    
    return audio_clips


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
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
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


if __name__ == "__main__":
    sr, noise = scipy.io.wavfile.read('./noise.wav')
    raw_noise = librosa.resample(noise, orig_sr=sr, target_sr=16000)

    root = './data/{}/{}.wav'
    vad_level = 2
    frame_duration = 30  # Duration of each frame in ms
    min_speech_duration = 1000
    per_clip_duration = 500 # clip in ms
    # model = vggish()

    dir_wav = './processed_data/wav_clips/'
    data_set = 'piezobuds/'

    labels = []

    for i in range(0, 51):
        dir_wav_piezo = dir_wav + data_set + 'piezo/' + '{}/'.format(i)
        dir_wav_audio = dir_wav + data_set + 'audio/' + '{}/'.format(i)             
        os.makedirs(dir_wav_audio, exist_ok=True)
        os.makedirs(dir_wav_piezo, exist_ok=True)
        print(i)
        piezos = []
        audios = []
        noises = []

        fft_piezos = []
        fft_audios = []

        file_path = root.format(i, i)
        sr, data = scipy.io.wavfile.read(file_path)
        if sr == 24000:
            piezo = data[:, 0]
            piezo = piezo[24000:len(piezo) - 3000]
            audio = data[24000 + 3000:, 1]
            piezo = butter_highpass_filter(piezo, 10, sr, order=5)
            piezo = nr.reduce_noise(piezo, sr)
        else:
            piezo = data[:, 0]
            piezo = piezo[16000:len(piezo) - 1500]
            audio = data[16000 + 1500:, 1]
            piezo = butter_highpass_filter(piezo, 10, sr, order=5)
            piezo = nr.reduce_noise(piezo, sr, y_noise=raw_noise)
            # piezo = piezo[24000:len(piezo) - 3000]
            # audio = data[24000 + 3000:, 1]
        piezo = piezo / np.max(np.abs(piezo))
        audio = audio / np.max(np.abs(audio))
        piezo = librosa.resample(piezo, orig_sr=sr, target_sr=16000)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

        speech_labels = apply_vad(audio, sr, vad_level, frame_duration)
        speech_utterances = split_audio_to_utterances(audio, sr, speech_labels, frame_duration, min_speech_duration)

        # longest = max(speech_utterances, key=lambda x: x[1] - x[0])
        # longest_length = longest[1] - longest[0]
        # print(longest_length)

        n_fft = 512
        hop_len = 256
        t_len = 16
        j = 0
        for utterance in speech_utterances:
            labels.append(i)

            piezo_piece = piezo[utterance[0]: utterance[1]]
            audio_piece = audio[utterance[0]: utterance[1]]

            piezo_clips = slice_audio(piezo_piece, sr, per_clip_duration)
            audio_clips = slice_audio(audio_piece, sr, per_clip_duration)
            for x in range(len(piezo_clips)):
                wav_writer.write(dir_wav_piezo + '{}.wav'.format(j), sr, piezo_clips[x])
                wav_writer.write(dir_wav_audio + '{}.wav'.format(j), sr, audio_clips[x])
                j = j + 1

            noise = generate_white_noise(len(audio_piece))

            bins_piezo = get_stft(piezo_piece, 16000, n_fft, hop_len, t_len)
            bins_audio = get_stft(audio_piece, 16000, n_fft, hop_len, t_len)
            piezos += bins_piezo
            audios += bins_audio

        for j in range(len(piezos)):
            piezos[j] = 10*np.log10(np.abs(piezos[j])**2)
            audios[j] = 10*np.log10(np.abs(audios[j])**2)
            audios_max = np.max(audios[j])
            audios_min = np.min(audios[j])
            audios[j] = (audios[j] - audios_min) / (audios_max - audios_min)
            piezos[j] = (piezos[j] - audios_min) / (audios_max - audios_min)

            # piezos[j] = librosa.amplitude_to_db(np.abs(piezos[j]))
            # audios[j] = librosa.amplitude_to_db(np.abs(audios[j]))
            
        piezos = np.array(piezos)
        audios = np.array(audios)
        piezos = piezos[:, :256, :]
        audios = audios[:, :256, :]
        n = piezos.shape[0]
        # os.makedirs("./stft_img/", exist_ok=True)
        # for j in range(n):
        #     plot_stft_colormap(piezos[j], "./stft_img/{}_{}_{}.png".format(i, j, 0))
        #     plot_stft_colormap(audios[j], "./stft_img/{}_{}_{}.png".format(i, j, 1))
        print(piezos.shape)
        print(audios.shape)
        # file_dir_name = './processed_data/res_512/'
        #file_dir_name = './processed_data/res_512a_512p/'
        file_dir_name = './processed_data/power_spectra/res_' + str(n_fft // 2) + '_hop_' + str(hop_len) + '_t_' + str(t_len) +'/'
        os.makedirs(file_dir_name, exist_ok=True)
        np.save(file_dir_name+'{}_piezo.npy'.format(i), piezos)
        np.save(file_dir_name+'{}_audio.npy'.format(i), audios)
        # np.save(file_dir_name+'{}_noise.npy'.format(i), noises)


        # print(len(speech_utterances))