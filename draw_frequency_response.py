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
from ECAPA_TDNN import *
from RealNVP import *
from GLOW import Glow
from math import log, sqrt, pi
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def apply_window(fft_magnitude, window_size):
    batch_size, num_of_utterance, frequency_bins = fft_magnitude.shape
    bins = frequency_bins // window_size
    fft_magnitude = fft_magnitude.contiguous()
    fft_magnitude = fft_magnitude.view(batch_size, num_of_utterance, bins, window_size)
    fft_magnitude = torch.mean(fft_magnitude, dim=-1)
    fft_magnitude = fft_magnitude.view(batch_size, num_of_utterance, -1)
    return fft_magnitude


def draw_pca(tensor, save_path=None):
    '''
    The shape of tensor is (batch_size, num_of_utterance, features)
    '''
    n_user, n_uttr, _ = tensor.shape
    scaler = StandardScaler()
    tensor = tensor.view(n_user * n_uttr, -1)
    data = tensor.detach().cpu().numpy()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=3)  # Reduce to 3 components
    pca.fit(scaled_data)
    reduced_data = pca.transform(scaled_data)
    reduced_data = reduced_data.reshape(n_user, n_uttr, -1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    # Get the colormap
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n_user))
    
    # Plot each user with a different color
    for i in range(n_user):
        ax.scatter(reduced_data[i, :, 0], reduced_data[i, :, 1], reduced_data[i, :, 2], alpha=0.5, label=f'User {i}', c=colors[i])

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('PCA of tensor by user')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    plt.show()
    plt.close()


import numpy as np
import matplotlib.pyplot as plt

def plot_user_averages(data, path=None, error_type='stderr'):
    """
    Plots the average sample and the error for each user.
    
    Parameters:
    - data: numpy array of shape (num_of_user, num_of_samples, sample_length)
    - error_type: 'stddev' for standard deviation or 'stderr' for standard error
    """
    num_of_user, num_of_samples, sample_length = data.shape

    # Calculate the mean for each user's samples
    means = np.mean(data, axis=1)
    
    # Calculate the standard deviation or standard error for each user's samples
    if error_type == 'stddev':
        errors = np.std(data, axis=1)
    elif error_type == 'stderr':
        errors = np.std(data, axis=1) / np.sqrt(num_of_samples)
    else:
        raise ValueError("error_type must be 'stddev' or 'stderr'")

    # Create an x-axis value for each sample point
    x_values = np.arange(sample_length)

    # Plotting
    plt.figure(figsize=(10, 7))  # Set the figure size as desired

    for i in range(num_of_user):
        plt.errorbar(x_values, means[i], yerr=errors[i], label=f'User {i+1}', fmt='-', capsize=5)

    plt.xlabel('Sample Point')
    plt.ylabel('Average Value')
    plt.title('Average Sample and Error for Each User')
    plt.legend()
    if path:
        plt.savefig(path)
        print(f'Plot saved to {path}')
    # plt.grid(True)
    plt.show()
    plt.close()


def normalize_tensor(tensor1, tensor2, dim=-1):
    # Compute the mean and standard deviation along the specified dimension
    max = torch.max(tensor2, dim=dim, keepdim=True).values
    min = torch.min(tensor2, dim=dim, keepdim=True).values
    # Normalize the tensor
    normalized_tensor1 = (tensor1 - min) / (max - min + 1e-5)  # Adding a small value to avoid division by zero
    normalized_tensor2 = (tensor2 - min) / (max - min + 1e-5)  # Adding a small value to avoid division by zero
    
    return normalized_tensor1, normalized_tensor2


def verify_frequency_response(device, data_set, batch_size=10,
                         n_fft=512, hop_length=256, win_length=512, window_fn = torch.hann_window, power=None,
                         num_epochs=20, train_ratio=0.8, fig_store_path=None):

    data_size = len(data_set)
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=False)

    sample_rate = 16000  # Sample rate is 16kHz
    target_high_freq = 1500   # Target frequency to clip at is 2500Hz
    target_low_freq = 500   # Target frequency to clip at is 2500Hz

    # Calculate the frequency resolution (freq per bin)
    freq_resolution = sample_rate / 8000
    # Calculate the number of bins needed to reach the target_freq
    num_bins_high = int(target_high_freq / freq_resolution)
    num_bins_low = int(target_low_freq / freq_resolution)


    for epoch in range(num_epochs):
        for batch_id, (piezo_clips, audio_clips, ids) in enumerate(dataloader):
            id = ids[:, 0].numpy()
            piezo_fft = torch.fft.fft(piezo_clips, dim=-1)
            audio_fft = torch.fft.fft(audio_clips, dim=-1)
            magnitude_piezo = torch.abs(piezo_fft)
            magnitude_audio = torch.abs(audio_fft)

            magnitude_piezo = magnitude_piezo[:, :, num_bins_low : num_bins_high]
            magnitude_audio = magnitude_audio[:, :, num_bins_low : num_bins_high]

            magnitude_piezo, magnitude_audio = normalize_tensor(magnitude_piezo, magnitude_audio)

            bins_piezo = apply_window(magnitude_piezo, window_size=10)
            bins_audio = apply_window(magnitude_audio, window_size=10)
            bins_frequency_response = bins_piezo / bins_audio
            plot_user_averages(bins_frequency_response.numpy(), fig_store_path+'{}_{}.png'.format(epoch, batch_id))
            





if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips_400ms/piezobuds/' # folder where stores the data for training and test
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------

    n_user = 69
    batch_size = 3

    n_fft = 512  # Size of FFT, affects the frequency granularity
    hop_length = 256  # Typically n_fft // 4 (is None, then hop_length = n_fft // 2 by default)
    win_length = n_fft  # Typically the same as n_fft
    window_fn = torch.hann_window # Window function

    fig_save_path = './frequency_response_fig/'
    os.makedirs(fig_save_path, exist_ok=True)

    # load the data 
    data_set = WavDatasetForVerification(data_file_dir, list(range(n_user)), 50)
    print(len(data_set))
    
    verify_frequency_response(device, data_set=data_set, batch_size=batch_size, fig_store_path=fig_save_path)
