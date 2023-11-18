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
from math import log, sqrt, pi
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from biGlow import *


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

def draw_pca(tensor, save_path=None):
    '''
    The shape of tensor is (batch_size, num_of_utterance, features)
    '''
    n_user, n_uttr, _ = tensor.shape
    scaler = StandardScaler()
    tensor = tensor.view(n_user * n_uttr, -1)
    data = tensor.detach().cpu().numpy()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=2)  # Reduce to 2 components
    pca.fit(scaled_data)
    reduced_data = pca.transform(scaled_data)
    reduced_data = reduced_data.reshape(n_user, n_uttr, -1)

    cmap = plt.get_cmap('viridis')  # Get the colormap
    colors = cmap(np.linspace(0, 1, n_user))  # Sample 'n_user' colors from the colormap
    discrete_cmap = matplotlib.colors.ListedColormap(colors)  # Create a new colormap from these colors

    for i in range(n_user):
        plt.scatter(reduced_data[i, :, 0], reduced_data[i, :, 1], alpha=0.5, label=f'User {i}', c=[discrete_cmap(i)])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of tensor by user')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    plt.show()
    plt.close()


def test_model(device, models, data_set, test_batch_size,
               n_fft=512, hop_length=256, win_length=512, window_fn = torch.hann_window, power=None,
               num_epochs=20, fig_store_path=None):
    
    test_loader = DataLoader(data_set, batch_size=test_batch_size, shuffle=True, drop_last=False)
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

    # initialize torchaudio.
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        extractor_a.eval()
        extractor_p.eval()
        converter.eval()
        dataloader = test_loader

        EERs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
        EER_FARs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
        EER_FRRs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
        EER_threshes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)

        for batch_id, (piezo_clips, audio_clips, ids) in enumerate(dataloader):
            # get shape of input
            batch_size, n_uttr, _ = piezo_clips.shape

            # prepare testing data
            piezo_clips = piezo_clips.to(device)
            audio_clips = audio_clips.to(device)
            
            _, n_uttr, f_len = piezo_clips.shape
            piezo_clips = piezo_clips.contiguous()
            audio_clips = audio_clips.contiguous()
            piezo_clips_enroll, piezo_clips_verify = torch.split(piezo_clips, n_uttr // 2, dim=1)
            audio_clips_enroll, audio_clips_verify = torch.split(audio_clips, n_uttr // 2, dim=1)

            piezo_clips = piezo_clips.view(batch_size * n_uttr, -1)
            audio_clips = audio_clips.view(batch_size * n_uttr, -1)

            embeddings_audio = extractor_a(audio_clips)
            embeddings_piezo = extractor_p(piezo_clips)
            embeddings_audio = embeddings_audio.contiguous()
            embeddings_piezo = embeddings_piezo.contiguous()
            embeddings_audio = embeddings_audio.view(batch_size, n_uttr, -1)
            embeddings_piezo = embeddings_piezo.view(batch_size, n_uttr, -1)

            # split data to enroll and verify
            embeddings_audio_enroll, embeddings_audio_verify = torch.split(embeddings_audio, n_uttr // 2, dim=1)
            embeddings_piezo_enroll, embeddings_piezo_verify = torch.split(embeddings_piezo, n_uttr // 2, dim=1)
            tmp_embeddings_audio_verify = torch.clone(embeddings_audio_verify).to(device)
            tmp_embeddings_piezo_verify = torch.clone(embeddings_piezo_verify).to(device)
            tmp_embeddings_audio_enroll = torch.clone(embeddings_audio_enroll).to(device)
            tmp_embeddings_piezo_enroll = torch.clone(embeddings_piezo_enroll).to(device)

            # test verification using single modality
            # modality: Audio
            centroids_a = get_centroids(tmp_embeddings_audio_enroll)
            sim_matrix = get_cossim(tmp_embeddings_audio_verify, centroids_a)
            EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
            EERs[1] += EER
            EER_FARs[1] += EER_FAR
            EER_FRRs[1] += EER_FRR
            EER_threshes[1] += EER_thresh
            # modality: Piezo
            centroids_p = get_centroids(tmp_embeddings_piezo_enroll)
            sim_matrix = get_cossim(tmp_embeddings_piezo_verify, centroids_p)
            EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
            EERs[2] += EER
            EER_FARs[2] += EER_FAR
            EER_FRRs[2] += EER_FRR
            EER_threshes[2] += EER_thresh

            # test EEGLOW
            audio_clips_enroll = audio_clips_enroll.contiguous()
            piezo_clips_enroll = piezo_clips_enroll.contiguous()
            audio_clips_enroll = audio_clips_enroll.view(batch_size * n_uttr // 2, -1)
            piezo_clips_enroll = piezo_clips_enroll.view(batch_size * n_uttr // 2, -1)
            embeddings_audio_enroll = extractor_a(audio_clips_enroll)
            embeddings_piezo_enroll = extractor_p(piezo_clips_enroll)
            embeddings_piezo_enroll = embeddings_piezo_enroll.contiguous()
            embeddings_piezo_enroll = embeddings_piezo_enroll.view(batch_size * n_uttr // 2, 3, 8, 8)

            audio_clips_verify = audio_clips_verify.contiguous()
            piezo_clips_verify = piezo_clips_verify.contiguous()
            audio_clips_verify = audio_clips_verify.view(batch_size * n_uttr // 2, -1)
            piezo_clips_verify = piezo_clips_verify.view(batch_size * n_uttr // 2, -1)
            embeddings_audio_verify = extractor_a(audio_clips_verify)
            embeddings_piezo_verify = extractor_p(piezo_clips_verify)
            embeddings_piezo_verify = embeddings_piezo_verify.contiguous()
            embeddings_piezo_verify = embeddings_piezo_verify.view(batch_size * n_uttr // 2, 3, 8, 8)

            embeddings_piezo_verify = (embeddings_piezo_verify - torch.min(embeddings_piezo_verify, dim=1, keepdim=True).values) / (
                                        torch.max(embeddings_piezo_verify, dim=1, keepdim=True).values - torch.min(embeddings_piezo_verify, dim=1, keepdim=True).values)
            embeddings_audio_verify = (embeddings_audio_verify - torch.min(embeddings_audio_verify, dim=1, keepdim=True).values) / (
                                        torch.max(embeddings_audio_verify, dim=1, keepdim=True).values - torch.min(embeddings_audio_verify, dim=1, keepdim=True).values)

            embeddings_piezo_enroll = (embeddings_piezo_enroll - torch.min(embeddings_piezo_enroll, dim=1, keepdim=True).values) / (
                                        torch.max(embeddings_piezo_enroll, dim=1, keepdim=True).values - torch.min(embeddings_piezo_enroll, dim=1, keepdim=True).values)
            embeddings_audio_enroll = (embeddings_audio_enroll - torch.min(embeddings_audio_enroll, dim=1, keepdim=True).values) / (
                                        torch.max(embeddings_audio_enroll, dim=1, keepdim=True).values - torch.min(embeddings_audio_enroll, dim=1, keepdim=True).values)

            draw_pca(embeddings_audio_verify.view(batch_size, n_uttr // 2, -1), fig_store_path + '{}_{}_audio.png'.format(epoch, batch_id))
            draw_pca(embeddings_piezo_verify.view(batch_size, n_uttr // 2, -1), fig_store_path + '{}_{}_piezo.png'.format(epoch, batch_id))

            log_p_sum, logdet, z_outs, conditions = converter(embeddings_piezo_enroll, embeddings_audio_enroll)
            z_outs = converter.reverse(z_outs, conditions=conditions, reconstruct=False)
            z_outs = z_outs.contiguous()
            embeddings_conv_enroll = z_outs.view(batch_size, n_uttr // 2, -1)
        
            # getting verify embeddings
            log_p_sum, logdet, z_outs, conditions = converter(embeddings_piezo_verify, embeddings_audio_verify)
            for z in range(len(z_outs)):
                draw_pca(z_outs[z].view(batch_size, n_uttr // 2, -1), fig_store_path + '{}_{}_zout{}.png'.format(epoch, batch_id, z))
            z_outs = converter.reverse(z_outs, conditions=conditions, reconstruct=False)
            z_outs = z_outs.contiguous()
            embeddings_conv_verify = z_outs.view(batch_size, n_uttr // 2, -1)
            draw_pca(embeddings_conv_verify.view(batch_size, n_uttr // 2, -1), fig_store_path + '{}_{}_conv.png'.format(epoch, batch_id))

            centroids = get_centroids(embeddings_conv_enroll)         
            sim_matrix = get_modal_cossim(embeddings_conv_verify, centroids)
            
            EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
            EERs[0] += EER
            EER_FARs[0] += EER_FAR
            EER_FRRs[0] += EER_FRR
            EER_threshes[0] += EER_thresh

        EERs /= len(dataloader)
        EER_FARs /= len(dataloader)
        EER_FRRs /= len(dataloader)
        EER_threshes /= len(dataloader)

        print("\nCentroids: AfP  Verification Input: AfP "
                    "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[0], EER_threshes[0], EER_FARs[0], EER_FRRs[0]))
        wandb.log({'epoch': epoch, 'EER/C_AfP_VI_AfP': EERs[0], 'FAR/C_AfP_VI_AfP': EER_FARs[0], 'FRR/C_AfP_VI_AfP': EER_FRRs[0]})
        wandb.log({'epoch': epoch, 'threshold/C_AfP_VI_AfP': EER_threshes[0]})

        print("\nCentroids: A  Verification Input: A "
                    "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[1], EER_threshes[1], EER_FARs[1], EER_FRRs[1]))
        wandb.log({'epoch': epoch, 'EER/C_A_VI_A': EERs[1], 'FAR/C_A_VI_A': EER_FARs[1], 'FRR/C_A_VI_A': EER_FRRs[1]})
        wandb.log({'epoch': epoch, 'threshold/C_A_VI_A': EER_threshes[1]})

        print("\nCentroids: P  Verification Input: P "
                    "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[2], EER_threshes[2], EER_FARs[2], EER_FRRs[2]))
        wandb.log({'epoch': epoch, 'EER/C_P_VI_P': EERs[2], 'FAR/C_P_VI_P': EER_FARs[2], 'FRR/C_P_VI_P': EER_FRRs[2]})
        wandb.log({'epoch': epoch, 'threshold/C_P_VI_P': EER_threshes[2]})
                

    return (extractor_a, extractor_p, converter)


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips/piezobuds/' # folder where stores the data for training and test
    model_pth = 'model_ecapatdnn_w_conGlow_cap_wo_enroll_Huberloss_no_detach/2023_11_17_14_43/'
    pth_store_dir = '/mnt/ssd/huaili/PiezoBuds/pth_model/' + model_pth
    test_user_id_files = '/mnt/ssd/huaili/PiezoBuds/pth_model/' + model_pth + 'test_users.json'
    fig_store_path = './pca_figs/' + model_pth
    os.makedirs(fig_store_path, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    n_user = 69
    num_of_epoches = 800
    train_batch_size = 4
    test_batch_size = 3

    n_fft = 512  # Size of FFT, affects the frequency granularity
    hop_length = 256  # Typically n_fft // 4 (is None, then hop_length = n_fft // 2 by default)
    win_length = n_fft  # Typically the same as n_fft
    window_fn = torch.hann_window # Window function

    comment = 'ecapatdnn_w_converter_MSEloss_sync'.format(n_fft, hop_length)

    extractor_a = ECAPA_TDNN(1024, is_stft=False)
    extractor_p = ECAPA_TDNN(1024, is_stft=False)

    extractor_a.load_state_dict(torch.load(pth_store_dir + 'extractor_a.pth'))
    extractor_p.load_state_dict(torch.load(pth_store_dir + 'extractor_p.pth'))
    extractor_a.to(device)
    extractor_p.to(device)

    converter = conditionGlow(in_channel=3, n_flow=2, n_block=3)
    converter.load_state_dict(torch.load(pth_store_dir + 'converter.pth'))
    converter.to(device)

    # create the folder to store the model
    model_struct = 'model_' + comment
    # initialize the wandb configuration
    time_stamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    # wandb.init(
    #     # team name
    #     entity="piezobuds",
    #     # set the project name
    #     project="PiezoBuds",
    #     # params of the task
    #     name=model_struct+'_'+time_stamp
    # )

    # load the test user list
    with open(test_user_id_files, 'r') as file:
        test_user_ids = json.load(file)

    data_set = WavDatasetForVerification(data_file_dir, test_user_ids, 50)
    print(len(data_set))

    models = (extractor_a, extractor_p, converter)
    test_model(device=device, models=models, data_set=data_set, test_batch_size=4, fig_store_path=fig_store_path)


    
