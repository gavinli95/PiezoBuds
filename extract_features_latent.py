import glob
import os
import librosa
import numpy as np
import soundfile as sf

from hparam import hparam as hp
import torch
import torchaudio
from torchvggish import vggish, vggish_input
from speech_split import split_audio_to_utterances, apply_vad
from matplotlib import pyplot as plt


root = './dataset_processed/{}/{}.wav'
vad_level = 2
frame_duration = 30  # Duration of each frame in ms
min_speech_duration = 1800
model = vggish()

for i in range(10):
    file_path = root.format(i, i)
    data, sr = sf.read(file_path)
    piezo = data[:, 0]
    piezo = piezo[:len(piezo) - 4555]
    audio = data[4555:, 1]

    speech_labels = apply_vad(audio, sr, vad_level, frame_duration)
    speech_utterances = split_audio_to_utterances(audio, sr, speech_labels, frame_duration, min_speech_duration)

    latent_feature_numpy_piezo = []
    latent_feature_numpy_audio = []

    for utterance in speech_utterances:
        piezo_piece = piezo[utterance[0]: utterance[1]]
        audio_piece = audio[utterance[0]: utterance[1]]

        vgg_piezo = vggish_input.waveform_to_examples(piezo_piece, sr)
        vgg_audio = vggish_input.waveform_to_examples(audio_piece, sr)
        with torch.no_grad():
            latent_features_piezo = model.forward(vgg_piezo)
            latent_features_audio = model.forward(vgg_audio)
            tmp = latent_features_piezo.detach().cpu().numpy()
            if tmp.ndim == 1:
                tmp = np.reshape(tmp, (1, 128))
            latent_feature_numpy_piezo = latent_feature_numpy_piezo + tmp.tolist()
            tmp = latent_features_audio.detach().cpu().numpy()
            if tmp.ndim == 1:
                tmp = np.reshape(tmp, (1, 128))
            latent_feature_numpy_audio = latent_feature_numpy_audio + tmp.tolist()

    latent_feature_numpy_piezo = np.array(latent_feature_numpy_piezo)
    latent_feature_numpy_audio = np.array(latent_feature_numpy_audio)
    np.save('./latent_feature/{}_piezo.npy'.format(i), latent_feature_numpy_piezo)
    np.save('./latent_feature/{}_audio.npy'.format(i), latent_feature_numpy_audio)

    print(len(speech_utterances))

