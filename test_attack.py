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


def compute_ASR(sim_matrix, threshold):
    """
    Compute Attack Success Rate given sim matrix and threshold.

    Args:
    - sim_matrix (torch.Tensor): A similarity matrix of shape 
      (num of speakers, num of utterances, num of speakers).
    - threshold: Threshold from test 

    Returns:
    - ASR: attack success rate
    """
    num_of_speakers, num_of_utters, _ = sim_matrix.shape

    sim_matrix_thresh = sim_matrix > threshold

    # Compute ASR
    ASR = sum([(sim_matrix_thresh[i].sum() - sim_matrix_thresh[i, :, i].sum()).float()
                for i in range(num_of_speakers)]) / (num_of_speakers - 1.0) / (num_of_utters) / num_of_speakers

    # FRR = sum([(num_of_utters - sim_matrix_thresh[i, :, i].sum()).float()
    #             for i in range(num_of_speakers)]) / (num_of_utters) / num_of_speakers

    return ASR

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


def get_conv_centroids(piezo_clips_enroll, audio_clips_enroll, extractor_a, extractor_p, converter):
    '''
    return centroids of conv, audio and piezo for enrollment

    Args:
    piezo_clips_enroll: piezo clips for enrollment (Batch size, num of enrol utterance, _)
    audio_clips_enroll: audio clips for enrollment (Batch size, num of enrol utterance, _)
    extractor_a: ECAPA for Audio
    extractor_p: ECAPA for Piezo
    converter: ConGLOW

    returns:
    embeddings_conv_enroll_centriods, (Batch_size, 1, 192)
    embeddings_audio_enroll_centriods, 
    embeddings_piezo_enroll_centriods

    '''
    batch_size, n_uttr_enroll, _ = piezo_clips_enroll.shape
    audio_clips_enroll = audio_clips_enroll.contiguous().view(batch_size * n_uttr_enroll, -1)
    piezo_clips_enroll = piezo_clips_enroll.contiguous().view(batch_size * n_uttr_enroll, -1)

    embeddings_audio_enroll = extractor_a(audio_clips_enroll)
    embeddings_piezo_enroll = extractor_p(piezo_clips_enroll)
    
    embeddings_audio_enroll = embeddings_audio_enroll.contiguous().view(batch_size, n_uttr_enroll, -1)
    embeddings_piezo_enroll = embeddings_piezo_enroll.contiguous().view(batch_size, n_uttr_enroll, -1)

    embeddings_piezo_enroll_centriods = get_centroids(embeddings_piezo_enroll)
    embeddings_audio_enroll_centriods = get_centroids(embeddings_audio_enroll)

    embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods.unsqueeze(1).expand(batch_size, n_uttr_enroll, -1)
    embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods_expand.contiguous().view(batch_size * n_uttr_enroll, 3, 8, 8)
    embeddings_audio_enroll = embeddings_audio_enroll.contiguous().view(batch_size * n_uttr_enroll, 3, 8, 8)
    log_p_sum, logdet, z_outs = converter(embeddings_piezo_enroll_centriods_expand, embeddings_audio_enroll)
    z_out = converter.reverse(z_outs, reconstruct=True)
    embeddings_conv_enroll = z_out.contiguous().view(batch_size, n_uttr_enroll, -1)
    embeddings_conv_enroll_centriods = get_centroids(embeddings_conv_enroll)

    return embeddings_conv_enroll_centriods, embeddings_audio_enroll_centriods, embeddings_piezo_enroll_centriods


def get_embeddings_verify(centroids_piezo_enroll, piezo_clips_verify, audio_clips_verify, extractor_a, extractor_p, converter):
    '''
    return embeddings of conv, audio and piezo for verification

    Args:
    centroids_piezo_enroll: centroids of piezo embeddings when enrollment
    piezo_clips_verify: piezo clips for verification (Batch size, num of enrol utterance, _)
    audio_clips_verify: audio clips for verification (Batch size, num of enrol utterance, _)
    extractor_a: ECAPA for Audio
    extractor_p: ECAPA for Piezo
    converter: ConGLOW

    returns:
    embeddings_conv_verify, (Batch_size, n_uttr_verify, 192)
    embeddings_audio_verify, 
    embeddings_piezo_verify

    '''
    batch_size, n_uttr_verify, _ = piezo_clips_verify.shape

    audio_clips_verify = audio_clips_verify.contiguous().view(batch_size * n_uttr_verify, -1)
    piezo_clips_verify = piezo_clips_verify.contiguous().view(batch_size * n_uttr_verify, -1)

    embeddings_audio_verify = extractor_a(audio_clips_verify)
    embeddings_piezo_verify = extractor_p(piezo_clips_verify)

    embeddings_piezo_enroll_centriods_expand = centroids_piezo_enroll.unsqueeze(1).expand(batch_size, n_uttr_verify, -1)
    embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods_expand.contiguous().view(batch_size * n_uttr_verify, 3, 8, 8)
    embeddings_audio_verify = embeddings_audio_verify.contiguous().view(batch_size * n_uttr_verify, 3, 8, 8)
    log_p_sum, logdet, z_outs = converter(embeddings_piezo_enroll_centriods_expand, embeddings_audio_verify)
    z_out = converter.reverse(z_outs, reconstruct=True)
    embeddings_conv_verify = z_out.contiguous().view(batch_size, n_uttr_verify, -1)

    return embeddings_conv_verify, embeddings_audio_verify.view(batch_size, n_uttr_verify, -1), embeddings_piezo_verify.view(batch_size, n_uttr_verify, -1)



def test_model(device, models, data_set, test_batch_size,
               n_fft=512, hop_length=256, win_length=512, window_fn = torch.hann_window, power=None,
               num_epochs=10, fig_store_path=None, threshold=0.5):
    
    test_loader = DataLoader(data_set, batch_size=test_batch_size, shuffle=True, drop_last=False)
    print(len(test_loader))

    (extractor_a, extractor_p, converter) = models

    ASR_across_epoch = np.array([0, 0, 0]).astype(float)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        extractor_a.eval()
        extractor_p.eval()
        converter.eval()
        dataloader = test_loader

        ASR_within_epoch = np.array([0, 0, 0]).astype(float)
        for batch_id, (piezo_clips, audio_clips, ids) in enumerate(dataloader):
            # get shape of input
            batch_size, n_uttr, _ = piezo_clips.shape

            # prepare testing data
            piezo_clips = piezo_clips.to(device)
            audio_clips = audio_clips.to(device)

            n_uttr_enroll = n_uttr - n_uttr // 5
            n_uttr_verify = n_uttr // 5
            
            _, n_uttr, f_len = piezo_clips.shape
            piezo_clips = piezo_clips.contiguous()
            audio_clips = audio_clips.contiguous()
            piezo_clips_enroll, piezo_clips_verify = torch.split(piezo_clips, [n_uttr_enroll, n_uttr_verify], dim=1)
            audio_clips_enroll, audio_clips_verify = torch.split(audio_clips, [n_uttr_enroll, n_uttr_verify], dim=1)
            
            # getting centroids and embeddings for attack test
            centroids_conv_enroll, centroids_audio_enroll, centroids_piezo_enroll = get_conv_centroids(
                piezo_clips_enroll, audio_clips_enroll, extractor_a, extractor_p, converter
            )
            embeddings_conv_verify, embeddings_audio_verify, emebeddings_piezo_verify = get_embeddings_verify(
                centroids_conv_enroll, piezo_clips_verify, audio_clips_verify, extractor_a, extractor_p, converter
            )
            
            # test replay attack
            # C-A-P, verify: change piezo to white noise
            white_noise = torch.rand(piezo_clips_verify.shape) * 2 - 1
            white_noise = white_noise.to(device)
            white_noise = white_noise.contiguous()
            white_noise = white_noise.view(batch_size * n_uttr_verify, -1)
            embeddings_noise = extractor_p(white_noise)
            embeddings_noise = embeddings_noise.contiguous()
            embeddings_noise = embeddings_noise.view(batch_size, n_uttr_verify, -1)
            sim_matrix = get_cossim(
                torch.cat((embeddings_conv_verify, embeddings_audio_verify, embeddings_noise), dim=-1),
                torch.cat((centroids_conv_enroll, centroids_audio_enroll, centroids_piezo_enroll), dim=-1)
            )
            ASR = compute_ASR(sim_matrix, threshold=threshold)
            ASR_within_epoch[0] += ASR.item()

            # test mimic attack
            # C-A-P, verify: change piezo to audio
            sim_matrix = get_cossim(
                torch.cat((embeddings_conv_verify, embeddings_audio_verify, embeddings_audio_verify), dim=-1),
                torch.cat((centroids_conv_enroll, centroids_audio_enroll, centroids_piezo_enroll), dim=-1)
            )
            ASR = compute_ASR(sim_matrix, threshold=threshold)
            ASR_within_epoch[1] += ASR.item()
        
        ASR_within_epoch /= len(dataloader)
        print("ASR of Replay attack within epoch %d: %.4f" % (epoch, ASR_within_epoch[0]))
        print("ASR of Mimic attack within epoch %d: %.4f" % (epoch, ASR_within_epoch[1]))

        ASR_across_epoch += ASR_within_epoch

    ASR_across_epoch /= num_epochs
    print("ASR of Replay attack across %d epochs: %.8f" % (num_epochs, ASR_within_epoch[0]))
    print("ASR of Mimic attack across %d epochs: %.8f" % (num_epochs, ASR_within_epoch[1]))

    return None


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
    num_of_epoches = 20
    test_batch_size = 7
    threshold = 0.5036

    n_fft = 512  # Size of FFT, affects the frequency granularity
    hop_length = 256  # Typically n_fft // 4 (is None, then hop_length = n_fft // 2 by default)
    win_length = n_fft  # Typically the same as n_fft
    window_fn = torch.hann_window # Window function

    comment = 'conGlow_centroids_based_test'

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

    data_set = WavDatasetForVerification(data_file_dir, test_user_ids, 100)
    print(len(data_set))

    models = (extractor_a, extractor_p, converter)
    test_model(device=device, models=models, data_set=data_set, test_batch_size=4, 
               fig_store_path=fig_store_path, num_epochs=num_of_epoches,
               threshold=threshold)


    
