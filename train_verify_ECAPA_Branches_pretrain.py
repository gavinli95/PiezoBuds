import os
import sys
sys.path.insert(0, "./ECAPA-TDNN-main/")
import json
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
# from ECAPA_TDNN_branches import *
from model import ECAPA_TDNN
from RealNVP import *
from GLOW import Glow
from math import log, sqrt, pi
from biGlow import *
from AAMSoftmax import *

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

def train_and_test_model(device, models, loss_func,
                         data_set, optimizer, scheduler,
                         train_batch_size, test_batch_size,
                         model_final_path,
                         num_epochs=2000, train_ratio=0.8, comment='default_model_description'):
    # load the dataset
    # data_size = len(data_set)
    # train_size = int(data_size * train_ratio)
    # test_size = data_size - train_size
    # train_tmp_set, test_tmp_set = torch.utils.data.random_split(data_set, [train_size, test_size])
    (train_tmp_set, test_tmp_set) = data_set
    if model_final_path:
        with open(model_final_path + 'train_users.json', 'w') as file:
            json.dump(train_tmp_set.n_user_list, file)
            file.close()
        with open(model_final_path + 'test_users.json', 'w') as file:
            json.dump(test_tmp_set.n_user_list, file)
            file.close()
    train_loader = DataLoader(train_tmp_set, batch_size=train_batch_size, shuffle=True, drop_last=False)
    print(len(train_loader))
    test_loader = DataLoader(test_tmp_set, batch_size=test_batch_size, shuffle=True, drop_last=False)
    print(len(test_loader))

    # load the models and ge2e loss
    extractor = models

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        print(comment)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                extractor.train()
                loss_func.train()
                dataloader = train_loader
            else:
                # set model to test
                extractor.eval()
                loss_func.eval()
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

            for batch_id, (audio_clips, ids) in enumerate(dataloader):
                # get shape of input
                batch_size, n_uttr, _ = audio_clips.shape
                ids = ids.contiguous().view(batch_size * n_uttr)
                ids = ids

                with torch.set_grad_enabled(phase == 'train') and torch.autograd.set_detect_anomaly(True):
                    
                    if phase == 'train':
                        # training
                        audio_clips = audio_clips.to(device)
                        
                        _, n_uttr, _ = audio_clips.shape
                        audio_clips = audio_clips.contiguous().view(batch_size * n_uttr, -1)

                        embeddings_audio = extractor(audio_clips)
                        # embeddings_audio = embeddings_audio.contiguous().view(batch_size, n_uttr, -1)

                        loss, ac = loss_func(embeddings_audio, ids)

                        loss_extractor = loss
                        loss_avg_batch_all += loss_extractor.item()
                        acc += ac
                        num_of_batches += 1
                        optimizer.zero_grad()
                        loss_extractor.backward()
                        torch.nn.utils.clip_grad_norm_(extractor.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()

                    if phase == 'test':
                        audio_clips = audio_clips.to(device)
                        
                        _, n_uttr, f_len = audio_clips.shape
                        audio_clips = audio_clips.contiguous()
                        n_uttr_enroll = n_uttr - n_uttr // 4
                        n_uttr_verify = n_uttr // 4
                        audio_clips_enroll, audio_clips_verify = torch.split(audio_clips, [n_uttr_enroll, n_uttr_verify], dim=1)

                        audio_clips = audio_clips.view(batch_size * n_uttr, -1)

                        embeddings_audio = extractor(audio_clips)
                        embeddings_audio = embeddings_audio.contiguous().view(batch_size * n_uttr, -1)
                        loss, ac = loss_func(embeddings_audio, ids)

                        loss_extractor = loss
                        loss_avg_batch_all += loss_extractor.item()
                        acc += ac
                        num_of_batches += 1

                        
                        with torch.set_grad_enabled(False) and torch.autograd.set_detect_anomaly(True):
                            audio_clips_enroll = audio_clips_enroll.contiguous().view(batch_size * n_uttr_enroll, -1)
                            embeddings_audio_enroll = extractor(audio_clips_enroll)
                            embeddings_audio_enroll = embeddings_audio_enroll.contiguous().view(batch_size, n_uttr_enroll, -1)
                            audio_clips_verify = audio_clips_verify.contiguous().view(batch_size * n_uttr_verify, -1)
                            embeddings_audio_verify = extractor(audio_clips_verify)
                            embeddings_audio_verify = embeddings_audio_verify.contiguous().view(batch_size, n_uttr_verify, -1)
                            
                        centroids = get_centroids(embeddings_audio_enroll)
                        sim_matrix = get_cossim(embeddings_audio_verify, centroids)
                        
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[0] += EER
                        EER_FARs[0] += EER_FAR
                        EER_FRRs[0] += EER_FRR
                        EER_threshes[0] += EER_thresh


            if phase == 'train':
                epoch_loss_all = loss_avg_batch_all / len(dataloader)
                # epoch_acc = acc / (len(dataloader) * train_batch_size)
                epoch_acc = acc / num_of_batches
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                print(f'{phase} Acc: {epoch_acc.item():.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_all': epoch_loss_all})
                wandb.log({'epoch': epoch, f'Acc/{phase}': epoch_acc})
            
            if phase == 'test':
                epoch_loss_all = loss_avg_batch_all / len(dataloader)
                # epoch_acc = acc / (len(dataloader) * train_batch_size)
                epoch_acc = acc / num_of_batches
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                print(f'{phase} Acc: {epoch_acc.item():.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_all': epoch_loss_all})
                wandb.log({'epoch': epoch, f'Acc/{phase}': epoch_acc})

                EERs /= len(dataloader)
                EER_FARs /= len(dataloader)
                EER_FRRs /= len(dataloader)
                EER_threshes /= len(dataloader)

                print("Centroids: A  Verification Input: A "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[0], EER_threshes[0], EER_FARs[0], EER_FRRs[0]))
                wandb.log({'epoch': epoch, 'EER/C_A_VI_A': EERs[0], 'FAR/C_A_VI_A': EER_FARs[0], 'FRR/C_A_VI_A': EER_FRRs[0]})
                wandb.log({'epoch': epoch, 'threshold/C_A_VI_A': EER_threshes[0]})
                

    return extractor


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = '/mnt/hdd/gen/processed_data/voxceleb1/wav_clips_500ms' # folder where stores the data for training and test
    train_data_file_dir = '/mnt/hdd/gen/processed_data/voxceleb1/wav_clips_1000ms/train/'
    test_data_file_dir = '/mnt/hdd/gen/processed_data/voxceleb1/wav_clips_1000ms/test/'
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.001
    n_user = [i for i in range(10001, 10252)]
    # n_user = [i for i in range(10001, 11252)]
    train_ratio = 0.8
    num_of_epoches = 400
    train_batch_size = 128
    test_batch_size = 128

    n_fft = 512  # Size of FFT, affects the frequency granularity
    hop_length = 256  # Typically n_fft // 4 (is None, then hop_length = n_fft // 2 by default)
    win_length = n_fft  # Typically the same as n_fft
    window_fn = torch.hann_window # Window function

    comment = 'ecapatdnn_ours'

    extractor = ECAPA_TDNN(1024)

    loaded_state = torch.load(pth_store_dir + 'pretrain_ecapa_tdnn.model')
    state_a = extractor_a.state_dict()
    state_p = extractor_p.state_dict()
    for name, param in loaded_state.items():
        origname = name
        name = remove_prefix(origname, 'speaker_encoder.')
        if name in state_a:
            if state_a[name].size() == loaded_state[origname].size():
                state_a[name].copy_(loaded_state[origname])
                state_p[name].copy_(loaded_state[origname])
    extractor_a.load_state_dict(state_a)
    extractor_p.load_state_dict(state_p)
    extractor.to(device)


    ge2e_loss = GE2ELoss_ori(device).to(device)
    aamsoftmax = AAMsoftmax(len(n_user), 0.2, 30, device).to(device)

    optimizer = torch.optim.Adam([
        {'params': extractor.parameters()},
        {'params': aamsoftmax.parameters()},
    ], lr=lr, weight_decay = 2e-5)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.97)
    
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
    # data_set = WavDatasetForVerification(data_file_dir, list(range(n_user)), 40)
    train_data_set = Voxceleb1Dataset(train_data_file_dir, n_user, 8, device=device)
    test_data_set = Voxceleb1Dataset(test_data_file_dir, n_user, 8, device=device)
    data_set = (train_data_set, test_data_set)

    loss_func = nn.HuberLoss()

    models = (extractor)
    extractor = train_and_test_model(device=device, models=models, loss_func=aamsoftmax, data_set=data_set, optimizer=optimizer, scheduler=lr_scheduler,
                                                       train_batch_size=train_batch_size, test_batch_size=test_batch_size, model_final_path=model_final_path,
                                                       num_epochs=num_of_epoches, train_ratio=train_ratio, comment=comment)

    torch.save(extractor.state_dict(), model_final_path+'extractor.pth')

    
