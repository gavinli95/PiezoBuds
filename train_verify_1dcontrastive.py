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

def train_and_test_model(device, models, ge2e_loss, loss_func, data_set, optimizer, train_batch_size, test_batch_size,
                         n_fft=512, hop_length=256, win_length=512, window_fn = torch.hann_window, power=None,
                         num_epochs=2000, train_ratio=0.8):

    data_size = len(data_set)
    train_size = int(data_size * train_ratio)
    test_size = data_size - train_size
    train_tmp_set, test_tmp_set = torch.utils.data.random_split(data_set, [train_size, test_size])
    train_loader = DataLoader(train_tmp_set, batch_size=train_batch_size, shuffle=True, drop_last=False)
    print(len(train_loader))
    test_loader = DataLoader(test_tmp_set, batch_size=test_batch_size, shuffle=True, drop_last=False)
    print(len(test_loader))

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

    (extractor_a, extractor_p, converter) = models
    (ge2e_loss_a, ge2e_loss_p) = ge2e_loss

    # initialize torchaudio.

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                extractor_a.train()
                extractor_p.train()
                converter.train()
                ge2e_loss_a.train()
                ge2e_loss_p.train()
                dataloader = train_loader
            else:
                # set model to test
                extractor_a.eval()
                extractor_p.eval()
                converter.eval()
                ge2e_loss_a.eval()
                ge2e_loss_p.eval()
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
                        piezo_clips.contiguous()
                        piezo_clips = piezo_clips.view(batch_size * n_uttr, -1)
                        audio_clips.contiguous()
                        audio_clips = audio_clips.view(batch_size * n_uttr, -1)

                        embeddings_audio = extractor_a(audio_clips)
                        embeddings_piezo = extractor_p(piezo_clips)
                        embeddings_conv = torch.clone(embeddings_audio)
                        embeddings_conv.contiguous()
                        embeddings_conv = embeddings_conv.view(batch_size * n_uttr, 1, -1)
                        embeddings_conv = converter(embeddings_conv)
                        embeddings_conv.contiguous()
                        embeddings_conv.squeeze()

                        embeddings_audio.contiguous()
                        embeddings_audio = embeddings_audio.view(batch_size, n_uttr, -1)
                        embeddings_piezo.contiguous()
                        embeddings_piezo = embeddings_piezo.view(batch_size, n_uttr, -1)
                        embeddings_conv.contiguous()
                        embeddings_conv = embeddings_conv.view(batch_size, n_uttr, -1)

                        loss_a = ge2e_loss_a(embeddings_audio)
                        loss_p = ge2e_loss_p(embeddings_piezo)
                        centroids_p = get_centroids(embeddings_piezo)
                        cos_sim = get_modal_cossim(embeddings_conv, centroids_p)
                        loss_conv, _, _, _ = calc_loss(cos_sim)
                        
                        loss_extractor = loss_a + loss_p + loss_conv
                        loss_avg_batch_all += loss_extractor.item()

                        optimizer.zero_grad()
                        loss_extractor.backward()
                        torch.nn.utils.clip_grad_norm_(extractor_a.parameters(), 10.0)
                        torch.nn.utils.clip_grad_norm_(extractor_p.parameters(), 10.0)
                        torch.nn.utils.clip_grad_norm_(converter.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_a.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_p.parameters(), 1.0)
                        optimizer.step()

                    if phase == 'test':
                        # tesing
                        # (batch_size, samples) (8000 raw audio signal, 16000 sample rate)
                        # process data using sincnet
                        # (batch_size, m, samples)
                        piezo_clips = piezo_clips.to(device)
                        audio_clips = audio_clips.to(device)
                        
                        _, n_uttr, f_len = piezo_clips.shape
                        piezo_clips.contiguous()
                        piezo_clips = piezo_clips.view(batch_size * n_uttr, -1)
                        audio_clips.contiguous()
                        audio_clips = audio_clips.view(batch_size * n_uttr, -1)

                        embeddings_audio = extractor_a(audio_clips)
                        embeddings_piezo = extractor_p(piezo_clips)
                        embeddings_audio.contiguous()
                        embeddings_piezo.contiguous()
                        embeddings_audio = embeddings_audio.view(batch_size, n_uttr, -1)
                        embeddings_piezo = embeddings_piezo.view(batch_size, n_uttr, -1)

                        # split data to enroll and verify
                        embeddings_audio_enroll, embeddings_audio_verify = torch.split(embeddings_audio, n_uttr // 2, dim=1)
                        embeddings_piezo_enroll, embeddings_piezo_verify = torch.split(embeddings_piezo, n_uttr // 2, dim=1)
                        tmp_converter = UNet1D(in_channels=1, out_channels=1).to(device)
                        tmp_converter.load_state_dict(converter.state_dict())
                        tmp_converter.train()
                        embeddings_enroll = torch.cat((embeddings_audio_enroll, embeddings_piezo_enroll), dim=-1)





                        embeddings_conv = torch.clone(embeddings_audio)
                        embeddings_conv.contiguous()
                        embeddings_conv = embeddings_conv.view(batch_size * n_uttr, 1, -1)
                        embeddings_conv = converter(embeddings_conv)
                        embeddings_conv.contiguous()
                        embeddings_conv.squeeze()

                        embeddings_audio.contiguous()
                        embeddings_audio = embeddings_audio.view(batch_size, n_uttr, -1)
                        embeddings_piezo.contiguous()
                        embeddings_piezo = embeddings_piezo.view(batch_size, n_uttr, -1)
                        embeddings_conv.contiguous()
                        embeddings_conv = embeddings_conv.view(batch_size, n_uttr, -1)

                        loss_a = ge2e_loss_a(embeddings_audio)
                        loss_p = ge2e_loss_p(embeddings_piezo)
                        centroids_p = get_centroids(embeddings_piezo)
                        cos_sim = get_modal_cossim(embeddings_conv, centroids_p)
                        loss_conv, _, _, _ = calc_loss(cos_sim)
                        
                        loss_extractor = loss_a + loss_p + loss_conv
                        loss_avg_batch_all += loss_extractor.item()

                        sim_matrix = get_modal_cossim(embeddings_veri, centroids)
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

                wandb.log({'epoch': epoch, f'Acc/{phase}': epoch_acc})
            if phase == 'test':
                EERs /= len(dataloader)
                EER_FARs /= len(dataloader)
                EER_FRRs /= len(dataloader)
                EER_threshes /= len(dataloader)

                print("\nCentroids: AfP  Verification Input: AfP "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[0], EER_threshes[0], EER_FARs[0], EER_FRRs[0]))
                wandb.log({'epoch': epoch, 'EER/C_AfP_VI_AfP': EERs[0], 'FAR/C_AfP_VI_AfP': EER_FARs[0], 'FRR/C_AfP_VI_AfP': EER_FRRs[0]})
                wandb.log({'epoch': epoch, 'threshold/C_AfP_VI_AfP': EER_threshes[0]})
                

    return extractor


if __name__ == "__main__":

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips/piezobuds/' # folder where stores the data for training and test
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.001
    n_user = 58
    train_ratio = 0.8
    num_of_epoches = 800
    train_batch_size = 4
    test_batch_size = 2

    n_fft = 512  # Size of FFT, affects the frequency granularity
    hop_length = 256  # Typically n_fft // 4 (is None, then hop_length = n_fft // 2 by default)
    win_length = n_fft  # Typically the same as n_fft
    window_fn = torch.hann_window # Window function

    comment = 'nfft_{}_hop_{}_verification'.format(n_fft, hop_length)
    # comment = 'mobilenetv3large1d_960_hop_256_t_16_class_pwr_spec_49u' # simple descriptions of specifications of this model, for example, 't_f' means we use the model which contains time and frequency nn layers


    # extractor initialization
    # extractor = torchvision.models.mobilenet_v3_large(pretrained=True)
    # extractor.features[0][0] = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # extractor.classifier[3] = nn.Linear(extractor.classifier[3].in_features, n_user)
    # 

    # extractor_p = torchvision.models.resnet50(pretrained=True)
    # extractor_p.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # extractor_p.fc = nn.Linear(extractor_p.fc.in_features, 51)
    # # hook for feature layer output
    # hook = extractor_p.avgpool.register_forward_hook(hook_fn)
    # extractor_p.to(device)

    # extractor_a = torchvision.models.resnet50(pretrained=True)
    # extractor_a.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # extractor_a.fc = nn.Linear(extractor_a.fc.in_features, 51)
    # # hook for feature layer output
    # hook = extractor_a.avgpool.register_forward_hook(hook_fn)
    # extractor_a.to(device)

    extractor_a = ECAPA_TDNN(512, is_stft=True).to(device)
    extractor_p = ECAPA_TDNN(512, is_stft=True).to(device)
    ge2e_loss_a = GE2ELoss_ori(device).to(device)
    ge2e_loss_p = GE2ELoss_ori(device).to(device)
    converter = UNet1D(in_channels=1, out_channels=1).to(device)

    optimizer = torch.optim.Adam([
        {'params': extractor_a.parameters()},
        {'params': extractor_p.parameters()},
        {'params': ge2e_loss_a.parameters()},
        {'params': ge2e_loss_p.parameters()},
        {'params': converter.parameters()},
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
    data_set = WavDatasetForVerification(data_file_dir, list(range(n_user)), 50)
    print(len(data_set))

    loss_func = nn.CrossEntropyLoss()

    models = (extractor_a, extractor_p, converter)
    ge2e_loss = (ge2e_loss_a, ge2e_loss_p)
    extractor = train_and_test_model(device=device, models=models, ge2e_loss=ge2e_loss, loss_func=loss_func, data_set=data_set, optimizer=optimizer,
                                                       train_batch_size=train_batch_size, test_batch_size=test_batch_size, n_fft=n_fft, 
                                                       hop_length=hop_length, win_length=win_length, window_fn=window_fn, power=None,
                                                       num_epochs=num_of_epoches, train_ratio=train_ratio)

    torch.save(extractor.state_dict(), model_final_path+'extractor.pth')

    
