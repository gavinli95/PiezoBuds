import os
import sys
import time
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import *
from my_models import *
from torch.utils.tensorboard import SummaryWriter
from UniqueDraw import UniqueDraw
from utils import *
import torchvision
from mobile_net_v3 import *
from SincNet import SincConv_fast
import torchaudio
import scipy.io.wavfile as wav_writer
from extract_feature import *

def save_sinc_output(sincnet_output, store_dir, sample_rate, spectrogram, A2D):
    '''
    Store the output of the SincNet
    '''
    if store_dir[-1] != '/':
        store_dir += '/'
    os.makedirs(store_dir, exist_ok=True)

    _, n_channel, T_len = sincnet_output.shape
    for i in range(n_channel):
        print('Processing output {}'.format(i))
        audio_script = sincnet_output[: , i, :]
        audio_script.squeeze()
        audio_script_n = audio_script.cpu().data.numpy()
        wav_writer.write(store_dir+'{}.wav'.format(i), sample_rate, audio_script_n)
        # save the stft image
        stft_matrix = A2D(spectrogram(audio_script).abs())
        plot_stft_colormap(stft_matrix=stft_matrix.cpu().data.numpy(), filename=store_dir+'{}.jpg'.format(i))
        print('Finished.')


def transform_audio_2_stft(input_audio, spec, a2d, store_dir, clip_id):
    '''
    Tranform the original audio script to STFT matrix
    '''
    # stft_matrix = a2d(spec(input_audio).abs())
    print('Saving clip {}...'.format(clip_id))

    stft_matrix = torch.stft(input_audio, n_fft=512, hop_length=256, win_length=512, return_complex=True)
    stft_matrix = 20*torch.log10(torch.abs(stft_matrix))
    plot_stft_colormap(stft_matrix=stft_matrix.cpu().data.numpy(), filename=store_dir+'{}_torchstft.jpg'.format(clip_id))

    stft_matrix = a2d(spec(input_audio).abs())
    plot_stft_colormap(stft_matrix=stft_matrix.cpu().data.numpy(), filename=store_dir+'{}_torchaudio.jpg'.format(clip_id))

    stft_matrix = librosa.stft(input_audio.data.numpy(), n_fft=512, hop_length=256, win_length=512, window="hann")
    stft_matrix = librosa.amplitude_to_db(np.abs(stft_matrix))
    plot_stft_colormap(stft_matrix, filename=store_dir+'{}_librosa.jpg'.format(clip_id))


    print('Saving finished!')


def load_sinc_model_output(model_pth, input_audio, device, sinc_out_channel, sinc_kernel_size, user_id, time_frame, sample_rate, spec, a2d):
    '''
    Load the SincNet model and generate the output given the input
    '''
    # load the models
    sinc_p = SincConv_fast(out_channels=sinc_out_channel, kernel_size=sinc_kernel_size).to(device)
    sinc_a = SincConv_fast(out_channels=sinc_out_channel, kernel_size=sinc_kernel_size).to(device)

    sinc_p.load_state_dict(torch.load(model_pth + 'sinc_p.pth'))
    sinc_a.load_state_dict(torch.load(model_pth + 'sinc_a.pth'))

    sinc_p.eval()
    sinc_a.eval()

    # load the input audio lists
    piezo_script = np.array(input_audio[0])
    audio_script = np.array(input_audio[1])

    piezo_clips = torch.from_numpy(piezo_script).float().to(device)
    audio_clips = torch.from_numpy(audio_script).float().to(device)

    # reshape to 1, 1, T_len
    piezo_clips.contiguous()
    audio_clips.contiguous()
    piezo_clips = piezo_clips.view(1, 1, -1)
    audio_clips = audio_clips.view(1, 1, -1)

    # generate the output
    # 1, sinc_out_channels, new_T_len
    piezo_scripts = sinc_p(piezo_clips)
    audio_scripts = sinc_a(audio_clips)
    
    # save the output to designated file path
    store_dir = './sicnnet_output/{}/user_{}/'.format(time_frame, user_id)
    os.makedirs(store_dir+'piezo/', exist_ok=True)
    save_sinc_output(piezo_scripts, store_dir+'piezo/', sample_rate, spec, a2d)
    os.makedirs(store_dir+'audio/', exist_ok=True)
    save_sinc_output(audio_scripts, store_dir+'audio/', sample_rate, spec, a2d)


if __name__ == "__main__":
    model_pth = ""
    input_audio_pth = "./processed_data/wav_clips/piezobuds/"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sinc_out_channel = 4
    sinc_kernel_size = 2
    user_id = 1
    sample_rate = 16000
    time_frame = model_pth.split('/')[-1].strip()
    # load the audio
    _, audio_script = wav_writer.read(input_audio_pth+'audio/{}/0.wav'.format(user_id))
    _, piezo_script = wav_writer.read(input_audio_pth+'piezo/{}/0.wav'.format(user_id))
    spectrogram = torchaudio.transforms.Spectrogram(
                                                    n_fft=512,
                                                    win_length=512,
                                                    hop_length=256,
                                                    # window_fn=512,
                                                    power=None,  # For power spectrogram, use 2. For complex spectrogram, use None.
                                                    # batch_first=True,
                                                    # sample_rate=16000
                                                )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)
    # load_sinc_model_output(model_pth=model_pth, input_audio=(audio_script, piezo_script), device=device, sinc_out_channel=sinc_out_channel,
    #                        sinc_kernel_size=sinc_kernel_size, user_id=user_id, time_frame=time_frame, sample_rate=sample_rate, spec=spectrogram, a2d=amplitude_to_db)
    store_dir = './clip_stft/'
    os.makedirs(store_dir, exist_ok=True)
    for i in range(51):
        print('Starting processing user {}'.format(i))
        store_dir_per_user = store_dir+'{}/'.format(i)
        os.makedirs(store_dir_per_user, exist_ok=True)
        for j in range(10):
            # load the audio script
            print("Audio")
            _, audio_script = wav_writer.read(input_audio_pth + 'audio/{}/{}.wav'.format(i, j))
            audio_script = torch.from_numpy(audio_script).float()
            transform_audio_2_stft(input_audio=audio_script, spec=spectrogram, a2d=amplitude_to_db, store_dir=store_dir_per_user, clip_id=str(j)+'_a')
            print("Piezo")
            _, piezo_script = wav_writer.read(input_audio_pth + 'piezo/{}/{}.wav'.format(i, j))
            piezo_script = torch.from_numpy(piezo_script).float()
            transform_audio_2_stft(input_audio=piezo_script, spec=spectrogram, a2d=amplitude_to_db, store_dir=store_dir_per_user, clip_id=str(j)+'_p')