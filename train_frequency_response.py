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
from utils import *
import torchvision
from mobile_net_v3 import *
import torchaudio
from ECAPA_TDNN import *
from RealNVP import *
from GLOW import Glow
from math import log, sqrt, pi
import json
from model_fr import *

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def hook_fn(module, input, output):
    """ Store the output of the hook """
    global hooked_output
    hooked_output = output


def calc_loss_glow(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def compute_EER(sim_matrix):
    """
    Compute EER, FAR, FRR and the threshold at which EER occurs.

    Args:
    - sim_matrix (torch.Tensor): A similarity matrix of shape 
      (num of speakers, num of utterances, num of speakers).

    Returns:
    - EER (float): Equal error rate.
    - threshold (float): The threshold at which EER occurs.
    - FAR (float): False acceptance rate at EER.
    - FRR (float): False rejection rate at EER.
    """
    num_of_speakers, num_of_utters, _ = sim_matrix.shape
    
    # Initialize values
    diff = float('inf')
    EER = 0.0
    threshold = 0.5
    EER_FAR = 0.0
    EER_FRR = 0.0

    # Iterate over potential thresholds
    for thres in torch.linspace(0.5, 1.0, 501):
        sim_matrix_thresh = sim_matrix > thres

        # Compute FAR and FRR
        FAR = sum([(sim_matrix_thresh[i].sum() - sim_matrix_thresh[i, :, i].sum()).float()
                    for i in range(num_of_speakers)]) / (num_of_speakers - 1.0) / (num_of_utters) / num_of_speakers

        FRR = sum([(num_of_utters - sim_matrix_thresh[i, :, i].sum()).float()
                   for i in range(num_of_speakers)]) / (num_of_utters) / num_of_speakers

        # Update if this is the closest FAR and FRR we've seen so far
        if diff > abs(FAR - FRR):
            diff = abs(FAR - FRR)
            EER = ((FAR + FRR) / 2).item()
            threshold = thres.item()
            EER_FAR = FAR.item()
            EER_FRR = FRR.item()

    return EER, threshold, EER_FAR, EER_FRR


def apply_window(fft_magnitude, window_size):
    batch_size, num_of_utterance, frequency_bins = fft_magnitude.shape
    bins = frequency_bins // window_size
    fft_magnitude = fft_magnitude.contiguous()
    fft_magnitude = fft_magnitude.view(batch_size, num_of_utterance, bins, window_size)
    fft_magnitude = torch.sum(fft_magnitude, dim=-1)
    fft_magnitude = fft_magnitude.view(batch_size, num_of_utterance, -1)
    return fft_magnitude


def normalize_tensor(tensor1, tensor2, dim=-1):
    # Compute the mean and standard deviation along the specified dimension
    max = torch.max(tensor2, dim=dim, keepdim=True).values
    min = torch.min(tensor2, dim=dim, keepdim=True).values
    # Normalize the tensor
    normalized_tensor1 = (tensor1 - min) / (max - min + 1e-5)  # Adding a small value to avoid division by zero
    normalized_tensor2 = (tensor2 - min) / (max - min + 1e-5)  # Adding a small value to avoid division by zero
    
    return normalized_tensor1, normalized_tensor2


def train_and_test_model(device, model, ge2e_loss, loss_func, data_set, optimizer, scheduler, train_batch_size, test_batch_size,
                         n_fft=512, hop_length=256, win_length=512, window_fn = torch.hann_window, power=None,
                         num_epochs=2000, train_ratio=0.8, model_final_path=None):

    data_size = len(data_set)
    train_size = int(data_size * train_ratio)
    test_size = data_size - train_size
    train_tmp_set, test_tmp_set = torch.utils.data.random_split(data_set, [train_size, test_size])
    if model_final_path:
        with open(model_final_path + 'train_users.json', 'w') as file:
            json.dump(train_tmp_set.indices, file)
            file.close()
        with open(model_final_path + 'test_users.json', 'w') as file:
            json.dump(test_tmp_set.indices, file)
            file.close()
    train_loader = DataLoader(train_tmp_set, batch_size=train_batch_size, shuffle=True, drop_last=False)
    print(len(train_loader))
    test_loader = DataLoader(test_tmp_set, batch_size=test_batch_size, shuffle=True, drop_last=False)
    print(len(test_loader))

    # initialize torchaudio.
    sample_rate = 16000  # Sample rate is 16kHz
    target_high_freq = 1500   # Target frequency to clip at is 2500Hz
    target_low_freq = 500   # Target frequency to clip at is 2500Hz

    # Calculate the frequency resolution (freq per bin)
    freq_resolution = sample_rate / 8000
    # Calculate the number of bins needed to reach the target_freq
    num_bins_high = int(target_high_freq / freq_resolution)
    num_bins_low = int(target_low_freq / freq_resolution)

    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                ge2e_loss.train()
                dataloader = train_loader
            else:
                # set model to test
                ge2e_loss.eval()
                dataloader = test_loader

            # train each batch
            num_of_batches = 0
            loss_avg_batch_all = 0.0
            loss_avg_batch_all_piezo = 0.0
            loss_avg_batch_all_audio = 0.0
            acc_audio = 0.0
            acc_piezo = 0.0
            acc = 0.0

            EERs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_FARs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_FRRs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_threshes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)

            for batch_id, (piezo_clips, audio_clips, ids) in enumerate(dataloader):
                # get shape of input
                batch_size, n_uttr, _ = piezo_clips.shape

                with torch.set_grad_enabled(phase == 'train') and torch.autograd.set_detect_anomaly(True):
                    
                    if phase == 'train':
                        # training
                        # (batch_size, samples) (8000 raw audio signal, 16000 sample rate)
                        # process data 
                        # (batch_size, m, samples)
                        piezo_clips = piezo_clips.to(device)
                        audio_clips = audio_clips.to(device)
                        
                        _, n_uttr, _ = piezo_clips.shape
                        piezo_clips = piezo_clips.contiguous()
                        piezo_clips = piezo_clips.view(batch_size * n_uttr, -1)
                        audio_clips = audio_clips.contiguous()
                        audio_clips = audio_clips.view(batch_size * n_uttr, -1)

                        piezo_fft = torch.fft.fft(piezo_clips, dim=-1)
                        audio_fft = torch.fft.fft(audio_clips, dim=-1)
                        magnitude_piezo = torch.abs(piezo_fft)
                        magnitude_audio = torch.abs(audio_fft)
                        magnitude_piezo = magnitude_piezo[:, num_bins_low : num_bins_high]
                        magnitude_audio = magnitude_audio[:, num_bins_low : num_bins_high]
                        magnitude_piezo = magnitude_piezo.contiguous()
                        magnitude_audio = magnitude_audio.contiguous()
                        num_of_win = magnitude_piezo.size(-1) // 100
                        magnitude_piezo = magnitude_piezo.view(batch_size * n_uttr, num_of_win, 100)
                        magnitude_audio = magnitude_audio.view(batch_size * n_uttr, num_of_win, 100)
                        magnitude_piezo = apply_window(magnitude_piezo, 5)
                        magnitude_audio = apply_window(magnitude_audio, 5)

                        magnitude_piezo, magnitude_audio = normalize_tensor(magnitude_piezo, magnitude_audio)
                        magnitude_response = magnitude_piezo / (magnitude_audio + 1e-5)
                        magnitude_response = magnitude_response.contiguous()
                        magnitude_response = magnitude_response.view(batch_size * n_uttr, -1)

                        magnitude_response = model(magnitude_response)
                        magnitude_response = magnitude_response.contiguous()
                        magnitude_response = magnitude_response.view(batch_size, n_uttr, -1)
                        loss = ge2e_loss(magnitude_response)

                        loss_extractor = loss
                        # if epoch >= epoch_th:
                        #     loss_extractor += loss_conv 
                        loss_avg_batch_all += loss_extractor.item()
                        optimizer.zero_grad()
                        loss_extractor.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()

                    if phase == 'test':

                        piezo_clips = piezo_clips.to(device)
                        audio_clips = audio_clips.to(device)
                        
                        _, n_uttr, _ = piezo_clips.shape
                        piezo_clips = piezo_clips.contiguous()
                        piezo_clips = piezo_clips.view(batch_size * n_uttr, -1)
                        audio_clips = audio_clips.contiguous()
                        audio_clips = audio_clips.view(batch_size * n_uttr, -1)

                        piezo_fft = torch.fft.fft(piezo_clips, dim=-1)
                        audio_fft = torch.fft.fft(audio_clips, dim=-1)
                        magnitude_piezo = torch.abs(piezo_fft)
                        magnitude_audio = torch.abs(audio_fft)
                        magnitude_piezo = magnitude_piezo[:, num_bins_low : num_bins_high]
                        magnitude_audio = magnitude_audio[:, num_bins_low : num_bins_high]
                        magnitude_piezo = magnitude_piezo.contiguous()
                        magnitude_audio = magnitude_audio.contiguous()
                        num_of_win = magnitude_piezo.size(-1) // 100
                        magnitude_piezo = magnitude_piezo.view(batch_size * n_uttr, num_of_win, 100)
                        magnitude_audio = magnitude_audio.view(batch_size * n_uttr, num_of_win, 100)
                        magnitude_piezo = apply_window(magnitude_piezo, 10)
                        magnitude_audio = apply_window(magnitude_audio, 10)

                        magnitude_piezo, magnitude_audio = normalize_tensor(magnitude_piezo, magnitude_audio)
                        magnitude_response = magnitude_piezo / (magnitude_audio + 1e-5)
                        magnitude_response = magnitude_response.contiguous()
                        magnitude_response = magnitude_response.view(batch_size * n_uttr, -1)

                        magnitude_response = model(magnitude_response)
                        magnitude_response = magnitude_response.contiguous()
                        magnitude_response = magnitude_response.view(batch_size, n_uttr, -1)

                        embeddings_enroll, embeddings_verify = torch.split(magnitude_response, n_uttr // 2, dim=1)

                        embeddings_verify = embeddings_verify.contiguous()
                        centroids = get_centroids(embeddings_enroll)
                        centroids = centroids.contiguous()
                        sim_matrix = get_cossim(embeddings_verify, centroids)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[0] += EER
                        EER_FARs[0] += EER_FAR
                        EER_FRRs[0] += EER_FRR
                        EER_threshes[0] += EER_thresh
            if phase == 'train':
                epoch_loss_all = loss_avg_batch_all / len(dataloader)
                epoch_acc = acc / (len(dataloader) * train_batch_size)
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_all': epoch_loss_all})

            if phase == 'test':
                EERs /= len(dataloader)
                EER_FARs /= len(dataloader)
                EER_FRRs /= len(dataloader)
                EER_threshes /= len(dataloader)

                print("\nCentroids: AfP  Verification Input: AfP "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[0], EER_threshes[0], EER_FARs[0], EER_FRRs[0]))
                wandb.log({'epoch': epoch, 'EER/C_AfP_VI_AfP': EERs[0], 'FAR/C_AfP_VI_AfP': EER_FARs[0], 'FRR/C_AfP_VI_AfP': EER_FRRs[0]})
                wandb.log({'epoch': epoch, 'threshold/C_AfP_VI_AfP': EER_threshes[0]})
                

    return model


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips/piezobuds/' # folder where stores the data for training and test
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.01
    n_user = 69
    train_ratio = 0.9
    num_of_epoches = 800
    train_batch_size = 4
    test_batch_size = 3

    n_fft = 512  # Size of FFT, affects the frequency granularity
    hop_length = 256  # Typically n_fft // 4 (is None, then hop_length = n_fft // 2 by default)
    win_length = n_fft  # Typically the same as n_fft
    window_fn = torch.hann_window # Window function

    comment = 'frequency_response'

    ge2e_loss = GE2ELoss_ori(device).to(device)

    optimizer = torch.optim.Adam([
        {'params': ge2e_loss.parameters()},
    ], lr=lr, weight_decay = 2e-5)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.97)
    
    # create the folder to store the model
    model_struct = 'model_' + comment
    # initialize the wandb configuration
    time_stamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    wandb.init(
        # team name
        entity="piezobuds",
        # set the project name
        project="PiezoBuds",
        # params of the task
        name=model_struct+'_'+time_stamp
    )
    model_store_pth = pth_store_dir + model_struct + '/'
    os.makedirs(model_store_pth, exist_ok=True)
    model_final_path = model_store_pth + time_stamp + '/'
    os.makedirs(model_final_path, exist_ok=True)

    # load the data 
    data_set = WavDatasetForVerification(data_file_dir, list(range(n_user)), 50)
    print(len(data_set))

    loss_func = nn.MSELoss()
    model = FrequencyResponse(50, 256).to(device)

    model = train_and_test_model(device=device, model=model, ge2e_loss=ge2e_loss, loss_func=loss_func, data_set=data_set, optimizer=optimizer, scheduler=lr_scheduler,
                                                       train_batch_size=train_batch_size, test_batch_size=test_batch_size, n_fft=n_fft, 
                                                       hop_length=hop_length, win_length=win_length, window_fn=window_fn, power=None,
                                                       num_epochs=num_of_epoches, train_ratio=train_ratio, model_final_path=model_final_path)

    torch.save(model.state_dict(), model_final_path+'model.pth')

    
