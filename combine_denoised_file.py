import glob
import os
import librosa
import numpy as np
import soundfile as sf

from hparam import hparam as hp
import torchaudio
import torchopenl3


sr = 44100
target_sr = 48000
resampler = torchaudio.transforms.Resample(sr, target_sr)


def extract_features():
    piezo_root = './piezo/{}.wav'
    audio_root = './audio/{}.wav'

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * target_sr
    print(utter_min_len)
    for i in range(10):
        os.mkdir('./dataset_processed/{}'.format(i))
        piezo, _ = torchaudio.load(piezo_root.format(i))
        audio, _ = torchaudio.load(audio_root.format(i))

        piezo = resampler(piezo)
        audio = resampler(audio)

        data = np.vstack([piezo, audio]).T
        sf.write('./dataset_processed/{}/{}.wav'.format(i, i), data, target_sr)


extract_features()
