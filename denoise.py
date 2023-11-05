import soundfile as sf
import numpy as np
import noisereduce as nr
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav_writer
import scipy.signal as signal
from scipy.signal import butter, filtfilt, savgol_filter


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


path = './dataset/{}/{}.wav'
noise_sample_start = 0
noise_sample_end = 10000

for i in range(0, 10):
    filepath = path.format(i, i)
    data, sr = sf.read(filepath)

    piezo = data[:, 0]
    piezo = butter_highpass_filter(piezo, 10, sr, 5)
    piezo = piezo / np.max(np.abs(piezo))
    audio = data[:, 1]

    noise_sample = piezo[noise_sample_start: noise_sample_end]
    piezo = nr.reduce_noise(piezo, sr, y_noise=noise_sample, prop_decrease=1.0)

    audio = audio[sr:]
    piezo = piezo[sr:]
    piezo = piezo/np.max(np.abs(piezo))

    sf.write('./piezo/{}.wav'.format(i), piezo, sr)
    sf.write('./audio/{}.wav'.format(i), audio, sr)



