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


def hook_fn(module, input, output):
    """ Store the output of the hook """
    global hooked_output
    hooked_output = output


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
    threshold = 0.0
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

def train_and_test_model(device, models, loss_func, data_set, optimizer, train_batch_size, test_batch_size,
                         n_fft=512, hop_length=256, win_length=512, window_fn = torch.hann_window, power=None,
                         num_epochs=2000, train_ratio=0.8, contain_sicnet=False, contain_2_extractor=False):
    # writer = SummaryWriter()
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
    data_size = len(data_set)
    train_size = int(data_size * train_ratio)
    test_size = data_size - train_size
    train_tmp_set, test_tmp_set = torch.utils.data.random_split(data_set, [train_size, test_size])
    train_loader = DataLoader(train_tmp_set, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_tmp_set, batch_size=test_batch_size, shuffle=True, drop_last=True)
    
    if contain_2_extractor:
        sinc_a, sinc_p, extractor_a, extractor_p = models
    else:
        sinc_a, sinc_p, extractor = models
    
    # functions for STFT matrix calculation
    spectrogram = torchaudio.transforms.Spectrogram(
                                                    n_fft=n_fft,
                                                    win_length=win_length,
                                                    hop_length=hop_length,
                                                    window_fn=window_fn,
                                                    power=power,  # For power spectrogram, use 2. For complex spectrogram, use None.
                                                    # batch_first=True,
                                                    # sample_rate=16000
                                                ).to(device)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80).to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                sinc_a.train()
                sinc_p.train()
                if contain_2_extractor:
                    extractor_a.train()
                    extractor_p.train()
                else:
                    extractor.train()
                dataloader = train_loader
            else:
                # set model to test
                sinc_a.eval()
                sinc_p.eval()
                if contain_2_extractor:
                    extractor_a.eval()
                    extractor_p.eval()
                else:
                    extractor.eval()
                dataloader = test_loader

            # train each batch
            loss_avg_batch_all = 0.0
            loss_avg_batch_all_piezo = 0.0
            loss_avg_batch_all_audio = 0.0
            acc_audio = 0.0
            acc_piezo = 0.0

            for batch_id, (piezo_clips, audio_clips, ids) in enumerate(dataloader):
                # get shape of input
                batch_size, _ = piezo_clips.shape

                with torch.set_grad_enabled(phase == 'train') and torch.autograd.set_detect_anomaly(True):
                    
                    if phase == 'train':
                        # training
                        # (batch_size, samples) (8000 raw audio signal, 16000 sample rate)
                        # process data using sincnet
                        # (batch_size, samples) -> (batch_size, sinc_out_channel, samples)
                        piezo_clips = piezo_clips.unsqueeze(1).to(device)
                        audio_clips = audio_clips.unsqueeze(1).to(device)
                        if contain_sicnet:
                            piezo_clips = sinc_p(piezo_clips)
                            # audio_clips = sinc_a(audio_clips)
                        _, channel_clip, _ = piezo_clips.shape
                        piezo_clips.contiguous()
                        piezo_clips = piezo_clips.view(batch_size * channel_clip, -1)
                        audio_clips.contiguous()
                        audio_clips = audio_clips.view(batch_size * 1, -1) # changed to 1

                        piezos = amplitude_to_db(spectrogram(piezo_clips).abs())
                        _, f_len, t_len = piezos.shape
                        audios = amplitude_to_db(spectrogram(audio_clips).abs())
                        _, f_len_a, t_len_a = audios.shape

                        # audio-based normalization
                        # piezos, audios = normalize_tensors_based_audio(tensor_p=piezos, tensor_a=audios)
                        #reshape
                        piezos.contiguous()
                        piezos = piezos.view(batch_size, channel_clip, f_len, t_len)
                        audios.contiguous()
                        audios = audios.view(batch_size, 1, f_len_a, t_len_a) ## changed to 1

                        if contain_2_extractor:
                            pred_ids_piezo = extractor_p(piezos)
                            pred_ids_audio = extractor_a(audios)
                        else:
                            pred_ids_piezo = extractor(piezos)
                            pred_ids_audio = extractor(audios)

                        ids_gpu = ids.to(device)
                        loss_class_piezo = loss_func(pred_ids_piezo, ids_gpu)
                        loss_class_audio = loss_func(pred_ids_audio, ids_gpu)
                        acc_audio += np.sum(np.argmax(pred_ids_audio.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())
                        acc_piezo += np.sum(np.argmax(pred_ids_piezo.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())
                        loss_extractor = loss_class_piezo + loss_class_audio
                        loss_avg_batch_all += loss_extractor.item()
                        loss_avg_batch_all_audio += loss_class_audio.item()
                        loss_avg_batch_all_piezo += loss_class_piezo.item()

                        optimizer.zero_grad()
                        loss_extractor.backward()
                        torch.nn.utils.clip_grad_norm_(sinc_a.parameters(), 10.0)
                        torch.nn.utils.clip_grad_norm_(sinc_p.parameters(), 10.0)
                        if contain_2_extractor:
                            torch.nn.utils.clip_grad_norm_(extractor_p.parameters(), 10.0)
                            torch.nn.utils.clip_grad_norm_(extractor_a.parameters(), 10.0)
                        else:
                            torch.nn.utils.clip_grad_norm_(extractor.parameters(), 10.0)
                        optimizer.step()

                    if phase == 'test':
                        # tesing
                        # (batch_size, samples) (8000 raw audio signal, 16000 sample rate)
                        # process data using sincnet
                        # (batch_size, samples) -> (batch_size, sinc_out_channel, samples)
                        piezo_clips = piezo_clips.unsqueeze(1).to(device)
                        audio_clips = audio_clips.unsqueeze(1).to(device)
                        if contain_sicnet:
                            piezo_clips = sinc_p(piezo_clips)
                            # audio_clips = sinc_a(audio_clips)
                        _, channel_clip, _ = piezo_clips.shape
                        piezo_clips.contiguous()
                        piezo_clips = piezo_clips.view(batch_size * channel_clip, -1)
                        audio_clips.contiguous()
                        audio_clips = audio_clips.view(batch_size * 1, -1)

                        piezos = amplitude_to_db(spectrogram(piezo_clips).abs())
                        _, f_len, t_len = piezos.shape
                        audios = amplitude_to_db(spectrogram(audio_clips).abs())
                        _, f_len_a, t_len_a = audios.shape

                        # audio-based normalization
                        # piezos, audios = normalize_tensors_based_audio(tensor_p=piezos, tensor_a=audios)
                        #reshape
                        piezos.contiguous()
                        piezos = piezos.view(batch_size, channel_clip, f_len, t_len)
                        audios.contiguous()
                        audios = audios.view(batch_size, 1, f_len_a, t_len_a)
                        
                        if contain_2_extractor:
                            pred_ids_piezo = extractor_p(piezos)
                            pred_ids_audio = extractor_a(audios)
                        else:
                            pred_ids_piezo = extractor(piezos)
                            pred_ids_audio = extractor(audios)

                        ids_gpu = ids.to(device)
                        loss_class_piezo = loss_func(pred_ids_piezo, ids_gpu)
                        loss_class_audio = loss_func(pred_ids_audio, ids_gpu)
                        acc_audio += np.sum(np.argmax(pred_ids_audio.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())
                        acc_piezo += np.sum(np.argmax(pred_ids_piezo.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())
                        loss_extractor = loss_class_piezo + loss_class_audio
                        loss_avg_batch_all += loss_extractor.item()
                        loss_avg_batch_all_audio += loss_class_audio.item()
                        loss_avg_batch_all_piezo += loss_class_piezo.item()
                        
            epoch_loss_all = loss_avg_batch_all / len(dataloader)
            epoch_loss_audio = loss_avg_batch_all_audio / len(dataloader)
            epoch_loss_piezo = loss_avg_batch_all_piezo / len(dataloader)

            if phase == 'train':
                epoch_acc_audio = acc_audio / (len(dataloader) * train_batch_size)
                epoch_acc_piezo = acc_piezo / (len(dataloader) * train_batch_size)
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                print(f'{phase} Acc Audio: {epoch_acc_audio:.4f}')
                print(f'{phase} Acc Piezo: {epoch_acc_piezo:.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_all': epoch_loss_all})
                wandb.log({'epoch': epoch, f'Loss/{phase}_audio': epoch_loss_audio})
                wandb.log({'epoch': epoch, f'Loss/{phase}_piezo': epoch_loss_piezo})

                wandb.log({'epoch': epoch, f'Acc/{phase}_audio': epoch_acc_audio}) 
                wandb.log({'epoch': epoch, f'Acc/{phase}_piezo': epoch_acc_piezo}) 
            if phase == 'test':
                epoch_acc_audio = acc_audio / (len(dataloader) * test_batch_size)
                epoch_acc_piezo = acc_piezo / (len(dataloader) * test_batch_size)
                print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')
                print(f'{phase} Acc Audio: {epoch_acc_audio:.4f}')
                print(f'{phase} Acc Piezo: {epoch_acc_piezo:.4f}')
                wandb.log({'epoch': epoch, f'Loss/{phase}_extractor': epoch_loss_all})
                wandb.log({'epoch': epoch, f'Loss/{phase}_audio': epoch_loss_audio})
                wandb.log({'epoch': epoch, f'Loss/{phase}_piezo': epoch_loss_piezo})

                wandb.log({'epoch': epoch, f'Acc/{phase}_audio': epoch_acc_audio}) 
                wandb.log({'epoch': epoch, f'Acc/{phase}_piezo': epoch_acc_piezo}) 

    return (sinc_a, sinc_p, extractor)


if __name__ == "__main__":

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = './processed_data/wav_clips/piezobuds/' # folder where stores the data for training and test
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.001
    n_user = 51
    train_ratio = 0.8
    num_of_epoches = 300
    train_seperate = False
    sinc_out_channel = 4
    sinc_kernel_size = 512
    contain_sincnet = True
    contain_2_extractor = True
    train_batch_size = 512
    test_batch_size = 128

    n_fft = 512  # Size of FFT, affects the frequency granularity
    hop_length = 256  # Typically n_fft // 4 (is None, then hop_length = n_fft // 2 by default)
    win_length = n_fft  # Typically the same as n_fft
    window_fn = torch.hann_window # Window function

    if contain_sincnet:
        comment = 'sincnet_out_{}_ksize_{}_mobilenetv3small_nfft_{}_hop_{}_classification'.format(sinc_out_channel, sinc_kernel_size, n_fft, hop_length)
    else:
        comment = 'nfft_{}_hop_{}_classification'.format(sinc_out_channel, sinc_kernel_size, n_fft, hop_length)
    if contain_2_extractor:
        comment += '_2_extractor'
    # comment = 'mobilenetv3large1d_960_hop_256_t_16_class_pwr_spec_49u' # simple descriptions of specifications of this model, for example, 't_f' means we use the model which contains time and frequency nn layers

    if contain_sincnet == False:
        sinc_out_channel = 1
    # SincNet initialization
    sinc_p = SincConv_fast(out_channels=sinc_out_channel, kernel_size=sinc_kernel_size).to(device)
    sinc_a = SincConv_fast(out_channels=sinc_out_channel, kernel_size=sinc_kernel_size).to(device)

    # extractor initialization
    if contain_2_extractor == False:
        extractor = torchvision.models.mobilenet_v3_large(pretrained=True)
        extractor.features[0][0] = nn.Conv2d(sinc_out_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        extractor.classifier[3] = nn.Linear(extractor.classifier[3].in_features, n_user)
        # hook for feature layer output
        hook = extractor.avgpool.register_forward_hook(hook_fn)
        extractor.to(device)
    else:
        extractor_a = torchvision.models.mobilenet_v3_large(pretrained=True)
        extractor_a.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) ## changed to 1s
        extractor_a.classifier[3] = nn.Linear(extractor_a.classifier[3].in_features, n_user)
        # hook for feature layer output
        hook_a = extractor_a.avgpool.register_forward_hook(hook_fn)
        extractor_a.to(device)

        extractor_p = torchvision.models.mobilenet_v3_large(pretrained=True)
        extractor_p.features[0][0] = nn.Conv2d(sinc_out_channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        extractor_p.classifier[3] = nn.Linear(extractor_p.classifier[3].in_features, n_user)
        # hook for feature layer output
        hook = extractor_p.avgpool.register_forward_hook(hook_fn)
        extractor_p.to(device)

    if contain_sincnet:
        if contain_2_extractor:
            optimizer = torch.optim.Adam([
            {'params': extractor_a.parameters()},
            {'params': extractor_p.parameters()},
            {'params': sinc_a.parameters()},
            {'params': sinc_p.parameters()}
        ], lr=lr)
        else:
            optimizer = torch.optim.Adam([
            {'params': extractor.parameters()},
            {'params': sinc_a.parameters()},
            {'params': sinc_p.parameters()}
        ], lr=lr)
    else:
        if contain_2_extractor:
            optimizer = torch.optim.Adam([
            {'params': extractor_a.parameters()},
            {'params': extractor_p.parameters()},
        ], lr=lr)
        else:
            optimizer = torch.optim.Adam([
            {'params': extractor.parameters()},
        ], lr=lr)
    
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
    data_set = WavDataset('./processed_data/wav_clips/piezobuds/', n_user=n_user, is_multi_moda=True)
    print(len(data_set))

    loss_func = nn.CrossEntropyLoss()

    if contain_2_extractor:
        models = (sinc_a, sinc_p, extractor_a, extractor_p)
        (sinc_a, sinc_p, extractor_a, extractor_p) = train_and_test_model(device=device, models=models, loss_func=loss_func, data_set=data_set, optimizer=optimizer,
                                                        train_batch_size=train_batch_size, test_batch_size=test_batch_size, n_fft=n_fft, 
                                                        hop_length=hop_length, win_length=win_length, window_fn=window_fn, power=None,
                                                        num_epochs=num_of_epoches, train_ratio=train_ratio, contain_sicnet=True, contain_2_extractor=contain_2_extractor)

        torch.save(extractor_a.state_dict(), model_final_path+'extractor_a.pth')
        torch.save(extractor_a.state_dict(), model_final_path+'extractor_p.pth')
        torch.save(sinc_a.state_dict(), model_final_path+'sinc_a.pth')
        torch.save(sinc_p.state_dict(), model_final_path+'sinc_p.pth')
    else:
        models = (sinc_a, sinc_p, extractor)
        (sinc_a, sinc_p, extractor) = train_and_test_model(device=device, models=models, loss_func=loss_func, data_set=data_set, optimizer=optimizer,
                                                        train_batch_size=train_batch_size, test_batch_size=test_batch_size, n_fft=n_fft, 
                                                        hop_length=hop_length, win_length=win_length, window_fn=window_fn, power=None,
                                                        num_epochs=num_of_epoches, train_ratio=train_ratio, contain_sicnet=True, contain_2_extractor=contain_2_extractor)

        torch.save(extractor.state_dict(), model_final_path+'extractor.pth')
        torch.save(sinc_a.state_dict(), model_final_path+'sinc_a.pth')
        torch.save(sinc_p.state_dict(), model_final_path+'sinc_p.pth')
    
