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

def train_and_test_model(device, extractor, ge2e_loss, loss_func, data_set, optimizer, train_batch_size, test_batch_size,
                         n_fft=512, hop_length=256, win_length=512, window_fn = torch.hann_window, power=None,
                         num_epochs=2000, train_ratio=0.8):
    # writer = SummaryWriter()
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
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

    # initialize torchaudio.

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                extractor.train()
                dataloader = train_loader
            else:
                # set model to test
                extractor.eval()
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

                        piezos = amplitude_to_db(spectrogram(piezo_clips).abs())
                        _, f_len, t_len = piezos.shape
                        piezos.contiguous()
                        piezos = piezos.view(batch_size, n_uttr, 1, f_len, t_len)
                        
                        audios = amplitude_to_db(spectrogram(audio_clips).abs())
                        _, f_len, t_len = audios.shape
                        audios.contiguous()
                        audios = audios.view(batch_size, n_uttr, 1, f_len, t_len)

                        combined = torch.cat((piezos, audios), dim=0)
                        combined.contiguous()
                        combined = combined.view(2 * batch_size * n_uttr, 1, f_len, t_len)

                        pred_ids_combined = extractor(combined)
                        embeddings_all = torch.clone(hooked_output)
                        embeddings_all.contiguous()
                        embeddings_all = embeddings_all.view(2 * batch_size, n_uttr, -1)
                        embeddings_piezo, embeddings_audio = torch.split(embeddings_all, batch_size, dim=0)
                        # loss_ge2e = ge2e_loss(embeddings_piezo) + ge2e_loss(embeddings_audio)
                        loss_ge2e = ge2e_loss(embeddings_all)

                        # ids_gpu = ids.to(device)
                        # ids_gpu = ids_gpu.reshape((batch_size * n_uttr))
                        # loss_class = loss_func(pred_ids_combined, ids_gpu)
                        # acc += np.sum(np.argmax(pred_ids_combined.cpu().data.numpy(), axis=1) == ids_gpu.cpu().data.numpy())
                        loss_extractor = loss_ge2e
                        loss_avg_batch_all += loss_extractor.item()

                        optimizer.zero_grad()
                        loss_extractor.backward()
                        torch.nn.utils.clip_grad_norm_(extractor.parameters(), 10.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
                        optimizer.step()

                    if phase == 'test':
                        # tesing
                        # (batch_size, samples) (8000 raw audio signal, 16000 sample rate)
                        # process data using sincnet
                        # (batch_size, m, samples)
                        piezo_clips = piezo_clips.to(device)
                        audio_clips = audio_clips.to(device)
                        
                        _, n_uttr, _ = piezo_clips.shape
                        piezo_clips.contiguous()
                        piezo_clips = piezo_clips.view(batch_size * n_uttr, -1)
                        audio_clips.contiguous()
                        audio_clips = audio_clips.view(batch_size * n_uttr, -1)

                        piezos = amplitude_to_db(spectrogram(piezo_clips).abs())
                        _, f_len, t_len = piezos.shape
                        piezos.contiguous()
                        piezos = piezos.view(batch_size, n_uttr, 1, f_len, t_len)
                        
                        audios = amplitude_to_db(spectrogram(audio_clips).abs())
                        _, f_len, t_len = audios.shape
                        audios.contiguous()
                        audios = audios.view(batch_size, n_uttr, 1, f_len, t_len)

                        combined = torch.cat((piezos, audios), dim=0)
                        combined.contiguous()
                        combined = combined.view(2 * batch_size * n_uttr, 1, f_len, t_len)

                        pred_ids_combined = extractor(combined)
                        embeddings_all = torch.clone(hooked_output)
                        embeddings_all.contiguous()
                        embeddings_all = embeddings_all.view(2 * batch_size, n_uttr, -1)
                        embeddings_piezo, embeddings_audio = torch.split(embeddings_all, batch_size, dim=0)
                        embeddings_enroll_piezo, embeddings_veri_piezo = torch.split(embeddings_piezo, n_uttr // 2, dim=1)
                        embeddings_enroll_audio, embeddings_veri_audio = torch.split(embeddings_audio, n_uttr // 2, dim=1)
                        embeddings_enroll = torch.cat((embeddings_enroll_piezo, embeddings_enroll_audio), dim=-1)
                        embeddings_veri = torch.cat((embeddings_veri_piezo, embeddings_veri_audio), dim=-1)
                        centroids = get_centroids(embeddings_enroll)

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
    
    data_file_dir = './processed_data/wav_clips/piezobuds/' # folder where stores the data for training and test
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

    extractor_p = torchvision.models.resnet50(pretrained=True)
    extractor_p.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    extractor_p.fc = nn.Linear(extractor_p.fc.in_features, 51)
    # hook for feature layer output
    hook = extractor_p.avgpool.register_forward_hook(hook_fn)
    extractor_p.to(device)

    extractor_a = torchvision.models.resnet50(pretrained=True)
    extractor_a.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    extractor_a.fc = nn.Linear(extractor_a.fc.in_features, 51)
    # hook for feature layer output
    hook = extractor_a.avgpool.register_forward_hook(hook_fn)
    extractor_a.to(device)

    ge2e_loss = GE2ELoss_ori(device).to(device)

    optimizer = torch.optim.Adam([
        {'params': extractor_a.parameters()},
        {'params': extractor_p.parameters()},
        {'params': ge2e_loss.parameters()}
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
    data_set = WavDatasetForVerification('./processed_data/wav_clips/piezobuds/', list(range(n_user)), 50)
    print(len(data_set))

    loss_func = nn.CrossEntropyLoss()

    extractor = train_and_test_model(device=device, extractor=extractor, ge2e_loss=ge2e_loss, loss_func=loss_func, data_set=data_set, optimizer=optimizer,
                                                       train_batch_size=train_batch_size, test_batch_size=test_batch_size, n_fft=n_fft, 
                                                       hop_length=hop_length, win_length=win_length, window_fn=window_fn, power=None,
                                                       num_epochs=num_of_epoches, train_ratio=train_ratio)

    torch.save(extractor.state_dict(), model_final_path+'extractor.pth')

    
