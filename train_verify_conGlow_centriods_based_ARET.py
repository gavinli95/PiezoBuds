import os
import sys
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
from ECAPA_TDNN import *
from RealNVP import *
from GLOW import Glow
from math import log, sqrt, pi
from biGlow import *
import torchvision
from models.AERT import *

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

def train_and_test_model(device, models, ge2e_loss, loss_func, 
                         data_set, optimizer, scheduler,
                         train_batch_size, test_batch_size,
                         model_final_path,
                         num_epochs=2000, train_ratio=0.8, comment='default_model_description'):
    # load the dataset
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
    extractor_a, extractor_p, converter, final_layer = models
    ge2e_loss_a, ge2e_loss_p, ge2e_loss_c = ge2e_loss

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        print(comment)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                extractor_a.train()
                extractor_p.train()
                converter.train()
                final_layer.train()
                ge2e_loss_a.train()
                ge2e_loss_p.train()
                ge2e_loss_c.train()
                dataloader = train_loader
            else:
                # set model to test
                extractor_a.eval()
                extractor_p.eval()
                converter.eval()
                final_layer.eval()
                ge2e_loss_a.eval()
                ge2e_loss_p.eval()
                ge2e_loss_c.eval()
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
                        piezo_clips = piezo_clips.to(device)
                        audio_clips = audio_clips.to(device)
                        
                        _, n_uttr, _ = piezo_clips.shape
                        piezo_clips = piezo_clips.contiguous().view(batch_size * n_uttr, -1)
                        audio_clips = audio_clips.contiguous().view(batch_size * n_uttr, -1)

                        mels_piezo = melextractor.get_spectrum(piezo_clips)
                        mels_audio = melextractor.get_spectrum(audio_clips)
                        mels_piezo = mels_piezo.unsqueeze(1)
                        mels_audio = mels_audio.unsqueeze(1)
                        embeddings_audio = extractor_a(mels_audio)
                        embeddings_piezo = extractor_p(mels_piezo)

                        embeddings_audio = embeddings_audio.contiguous().view(batch_size, n_uttr, -1)
                        embeddings_piezo = embeddings_piezo.contiguous().view(batch_size, n_uttr, -1)

                        loss_a = ge2e_loss_a(embeddings_audio)
                        loss_p = ge2e_loss_p(embeddings_piezo)

                        # embeddings_audio = embeddings_audio.detach()
                        # embeddings_piezo = embeddings_piezo.detach()

                        # cal converter loss
                        # normalize the data of different modalities
                        # embeddings_piezo = (embeddings_piezo - torch.min(embeddings_piezo, dim=1, keepdim=True).values) / (
                        #                     torch.max(embeddings_piezo, dim=1, keepdim=True).values - torch.min(embeddings_piezo, dim=1, keepdim=True).values)

                        # embeddings_audio = (embeddings_audio - torch.min(embeddings_audio, dim=1, keepdim=True).values) / (
                        #                     torch.max(embeddings_audio, dim=1, keepdim=True).values - torch.min(embeddings_audio, dim=1, keepdim=True).values)
                        
                        # reshape the embeddings to (B, U, 192)
                        embeddings_piezo_centriods = get_centroids(embeddings_piezo)
                        
                        # expand the centriods to the same size of condition
                        embeddings_piezo_centriods_expand = embeddings_piezo_centriods.unsqueeze(1).expand(batch_size, n_uttr, -1)
                        embeddings_piezo_centriods_expand = embeddings_piezo_centriods_expand.contiguous().view(batch_size * n_uttr, 1, 16, 16)
                        embeddings_audio = embeddings_audio.view(batch_size * n_uttr, 1, 16, 16)

                        log_p_sum, logdet, z_outs = converter(embeddings_piezo_centriods_expand, embeddings_audio)
                        z_out = converter.reverse(z_outs, reconstruct=True)
                        # only applicable to biGlow model
                        embeddings_conv = z_out.contiguous().view(batch_size, n_uttr, -1)
                        loss_conv = loss_func(embeddings_conv.view(batch_size * n_uttr, -1), embeddings_piezo.view(batch_size * n_uttr, -1))
                        loss_extractor = loss_a + loss_p + loss_conv
                        loss_avg_batch_all += loss_extractor.item()
                        optimizer.zero_grad()
                        loss_extractor.backward()
                        torch.nn.utils.clip_grad_norm_(extractor_a.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(extractor_p.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(converter.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_a.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_p.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_c.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()

                    if phase == 'test':
                        piezo_clips = piezo_clips.to(device)
                        audio_clips = audio_clips.to(device)
                        
                        _, n_uttr, f_len = piezo_clips.shape
                        piezo_clips = piezo_clips.contiguous()
                        audio_clips = audio_clips.contiguous()
                        n_uttr_enroll = n_uttr - n_uttr // 4
                        n_uttr_verify = n_uttr // 4
                        piezo_clips_enroll, piezo_clips_verify = torch.split(piezo_clips, [n_uttr_enroll, n_uttr_verify], dim=1)
                        audio_clips_enroll, audio_clips_verify = torch.split(audio_clips, [n_uttr_enroll, n_uttr_verify], dim=1)

                        piezo_clips = piezo_clips.view(batch_size * n_uttr, -1)
                        audio_clips = audio_clips.view(batch_size * n_uttr, -1)

                        mels_piezo = melextractor.get_spectrum(piezo_clips)
                        mels_audio = melextractor.get_spectrum(audio_clips)
                        mels_piezo = mels_piezo.unsqueeze(1)
                        mels_audio = mels_audio.unsqueeze(1)
                        embeddings_audio = extractor_a(mels_audio)
                        embeddings_piezo = extractor_p(mels_piezo)
                        embeddings_audio = embeddings_audio.contiguous().view(batch_size, n_uttr, -1)
                        embeddings_piezo = embeddings_piezo.contiguous().view(batch_size, n_uttr, -1)

                        # store the original audio/piezo verify & enroll data for further sim matrix calculation
                        tmp_embeddings_audio_enroll, tmp_embeddings_audio_verify = embeddings_audio.chunk(2, 1)
                        tmp_embeddings_piezo_enroll, tmp_embeddings_piezo_verify = embeddings_piezo.chunk(2, 1)
                        
                        # create a temporary converter along with its following trainable structures for the enrollment
                        tmp_converter = conditionGlow(in_channel=1, n_flow=2, n_block=4).to(device)
                        tmp_converter.load_state_dict(converter.state_dict())
                        tmp_final_layer = FClayer().to(device)
                        tmp_final_layer.load_state_dict(final_layer.state_dict())
                        tmp_ge2e_loss_c = GE2ELoss_ori(device).to(device)
                        tmp_ge2e_loss_c.load_state_dict(ge2e_loss_c.state_dict())
                        tmp_optimizer = torch.optim.Adam([
                            {'params': tmp_converter.parameters()},
                            {'params': tmp_final_layer.parameters()},
                            {'params': tmp_ge2e_loss_c.parameters()},
                        ], lr=lr)
                        tmp_converter.train()
                        tmp_final_layer.train()
                        tmp_ge2e_loss_c.train()
                        with torch.set_grad_enabled(True) and torch.autograd.set_detect_anomaly(True):
                            for _ in range(0):
                                audio_clips_enroll = audio_clips_enroll.contiguous().view(batch_size * n_uttr_enroll, -1)
                                piezo_clips_enroll = piezo_clips_enroll.contiguous().view(batch_size * n_uttr_enroll, -1)

                                mels_piezo_enroll = melextractor.get_spectrum(piezo_clips_enroll)
                                mels_audio_enroll = melextractor.get_spectrum(audio_clips_enroll)
                                mels_piezo_enroll = mels_piezo_enroll.unsqueeze(1)
                                mels_audio_enroll = mels_audio_enroll.unsqueeze(1)
                                embeddings_audio_enroll = extractor_a(mels_audio_enroll)
                                embeddings_piezo_enroll = extractor_p(mels_piezo_enroll)
                                embeddings_audio_enroll = embeddings_audio_enroll.contiguous().view(batch_size, n_uttr, -1)
                                embeddings_piezo_enroll = embeddings_piezo_enroll.contiguous().view(batch_size, n_uttr, -1)

                                # normalize the embeddings of piezo modality
                                # embeddings_piezo_enroll = (embeddings_piezo_enroll - torch.min(embeddings_piezo_enroll, dim=1, keepdim=True).values) / (
                                #     torch.max(embeddings_piezo_enroll, dim=1, keepdim=True).values - torch.min(embeddings_piezo_enroll, dim=1, keepdim=True).values)
                                # embeddings_audio_enroll = (embeddings_audio_enroll - torch.min(embeddings_audio_enroll, dim=1, keepdim=True).values) / (
                                #     torch.max(embeddings_audio_enroll, dim=1, keepdim=True).values - torch.min(embeddings_audio_enroll, dim=1, keepdim=True).values)
                                
                                embeddings_piezo_enroll_centriods = get_centroids(embeddings_piezo_enroll)
                                embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods.unsqueeze(1).expand(batch_size, n_uttr_enroll, -1)

                                embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods_expand.contiguous().view(batch_size * n_uttr_enroll, 1, 16, 16)
                                embeddings_audio_enroll = embeddings_audio_enroll.view(batch_size * n_uttr_enroll, 1, 16, 16)
                                log_p_sum, logdet, z_outs = tmp_converter(embeddings_piezo_enroll_centriods_expand, embeddings_audio_enroll)
                                z_out = tmp_converter.reverse(z_outs, reconstruct=True)
                                embeddings_conv = z_out.contiguous().view(batch_size, n_uttr_enroll -1)
                                loss_conv = loss_func(embeddings_conv.view(batch_size * n_uttr_enroll, -1), embeddings_piezo_enroll.view(batch_size * n_uttr_enroll, -1))
                                tmp_optimizer.zero_grad()
                                loss_conv.backward()
                                torch.nn.utils.clip_grad_norm_(tmp_converter.parameters(), 3.0)
                                tmp_optimizer.step()
                        
                        tmp_converter.eval()
                        tmp_final_layer.eval()
                        tmp_ge2e_loss_c.eval()
                        with torch.set_grad_enabled(False) and torch.autograd.set_detect_anomaly(True):
                            audio_clips_enroll = audio_clips_enroll.contiguous().view(batch_size * n_uttr_enroll, -1)
                            piezo_clips_enroll = piezo_clips_enroll.contiguous().view(batch_size * n_uttr_enroll, -1)

                            mels_piezo_enroll = melextractor.get_spectrum(piezo_clips_enroll)
                            mels_audio_enroll = melextractor.get_spectrum(audio_clips_enroll)
                            mels_piezo_enroll = mels_piezo_enroll.unsqueeze(1)
                            mels_audio_enroll = mels_audio_enroll.unsqueeze(1)
                            embeddings_audio_enroll = extractor_a(mels_audio_enroll)
                            embeddings_piezo_enroll = extractor_p(mels_piezo_enroll)
                            embeddings_audio_enroll = embeddings_audio_enroll.contiguous().view(batch_size, n_uttr_enroll, -1)
                            embeddings_piezo_enroll = embeddings_piezo_enroll.contiguous().view(batch_size, n_uttr_enroll, -1)

                            audio_clips_verify = audio_clips_verify.contiguous().view(batch_size * n_uttr_verify, -1)
                            piezo_clips_verify = piezo_clips_verify.contiguous().view(batch_size * n_uttr_verify, -1)

                            mels_piezo_verify = melextractor.get_spectrum(piezo_clips_verify)
                            mels_audio_verify = melextractor.get_spectrum(audio_clips_verify)
                            mels_piezo_verify = mels_piezo_verify.unsqueeze(1)
                            mels_audio_verify = mels_audio_verify.unsqueeze(1)
                            embeddings_audio_verify = extractor_a(mels_audio_verify)
                            embeddings_piezo_verify = extractor_p(mels_piezo_verify)
                            embeddings_audio_verify = embeddings_audio_verify.contiguous().view(batch_size, n_uttr_verify, -1)
                            embeddings_piezo_verify = embeddings_piezo_verify.contiguous().view(batch_size, n_uttr_verify, -1)
                            
                            # normalize
                            # embeddings_piezo_verify = (embeddings_piezo_verify - torch.min(embeddings_piezo_verify, dim=1, keepdim=True).values) / (
                            #                            torch.max(embeddings_piezo_verify, dim=1, keepdim=True).values - torch.min(embeddings_piezo_verify, dim=1, keepdim=True).values)
                            # embeddings_audio_verify = (embeddings_audio_verify - torch.min(embeddings_audio_verify, dim=1, keepdim=True).values) / (
                            #                            torch.max(embeddings_audio_verify, dim=1, keepdim=True).values - torch.min(embeddings_audio_verify, dim=1, keepdim=True).values)

                            # embeddings_piezo_enroll = (embeddings_piezo_enroll - torch.min(embeddings_piezo_enroll, dim=1, keepdim=True).values) / (
                            #                            torch.max(embeddings_piezo_enroll, dim=1, keepdim=True).values - torch.min(embeddings_piezo_enroll, dim=1, keepdim=True).values)
                            # embeddings_audio_enroll = (embeddings_audio_enroll - torch.min(embeddings_audio_enroll, dim=1, keepdim=True).values) / (
                            #                            torch.max(embeddings_audio_enroll, dim=1, keepdim=True).values - torch.min(embeddings_audio_enroll, dim=1, keepdim=True).values)

                            # getting enrollment embeddings
                            embeddings_piezo_enroll_centriods = get_centroids(embeddings_piezo_enroll)
                            embeddings_audio_enroll_centriods = get_centroids(embeddings_audio_enroll)

                            embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods.unsqueeze(1).expand(batch_size, n_uttr_enroll, -1)
                            embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods_expand.contiguous().view(batch_size * n_uttr_enroll, 1, 16, 16)
                            embeddings_audio_enroll = embeddings_audio_enroll.contiguous().view(batch_size * n_uttr_enroll, 1, 16, 16)
                            log_p_sum, logdet, z_outs = tmp_converter(embeddings_piezo_enroll_centriods_expand, embeddings_audio_enroll)
                            z_out = tmp_converter.reverse(z_outs, reconstruct=True)
                            embeddings_conv_enroll = z_out.contiguous().view(batch_size, n_uttr_enroll, -1)
                            embeddings_conv_enroll_centriods = get_centroids(embeddings_conv_enroll)

                            embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods.unsqueeze(1).expand(batch_size, n_uttr_verify, -1)
                            embeddings_piezo_enroll_centriods_expand = embeddings_piezo_enroll_centriods_expand.contiguous().view(batch_size * n_uttr_verify, 1, 16, 16)
                            embeddings_audio_verify = embeddings_audio_verify.contiguous().view(batch_size * n_uttr_verify, 1, 16, 16)
                            log_p_sum, logdet, z_outs = tmp_converter(embeddings_piezo_enroll_centriods_expand, embeddings_audio_verify)
                            z_out = tmp_converter.reverse(z_outs, reconstruct=True)
                            embeddings_conv_verify = z_out.contiguous().view(batch_size, n_uttr_verify, -1)   
                            embeddings_audio_verify = embeddings_audio_verify.contiguous().view(batch_size, n_uttr_verify, -1)

                            embeddings_conv_verify_w_piezo_audio = torch.cat((embeddings_conv_verify, embeddings_piezo_verify, embeddings_audio_verify), dim=-1)
                            embeddings_conv_enroll_w_piezo_audio_centriods = torch.cat((embeddings_conv_enroll_centriods, embeddings_piezo_enroll_centriods, embeddings_audio_enroll_centriods), dim=-1)
                            sim_matrix = get_modal_cossim(embeddings_conv_verify_w_piezo_audio, embeddings_conv_enroll_w_piezo_audio_centriods)

                        
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[0] += EER
                        EER_FARs[0] += EER_FAR
                        EER_FRRs[0] += EER_FRR
                        EER_threshes[0] += EER_thresh

                        centroids_a = get_centroids(tmp_embeddings_audio_enroll)
                        sim_matrix = get_cossim(torch.clone(tmp_embeddings_audio_verify).to(device), centroids_a)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[1] += EER
                        EER_FARs[1] += EER_FAR
                        EER_FRRs[1] += EER_FRR
                        EER_threshes[1] += EER_thresh

                        centroids_p = get_centroids(tmp_embeddings_piezo_enroll)
                        sim_matrix = get_cossim(torch.clone(tmp_embeddings_piezo_verify).to(device), centroids_p)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[2] += EER
                        EER_FARs[2] += EER_FAR
                        EER_FRRs[2] += EER_FRR
                        EER_threshes[2] += EER_thresh

                        # AP concatenation
                        centroids_ap = torch.cat((centroids_a, centroids_p), dim=-1)
                        embeddings_ap = torch.cat((tmp_embeddings_audio_verify, tmp_embeddings_piezo_verify), dim=-1)
                        sim_matrix = get_cossim(torch.clone(embeddings_ap).to(device), centroids_ap)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[3] += EER
                        EER_FARs[3] += EER_FAR
                        EER_FRRs[3] += EER_FRR
                        EER_threshes[3] += EER_thresh

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

                print("\nCentroids: A  Verification Input: A "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[1], EER_threshes[1], EER_FARs[1], EER_FRRs[1]))
                wandb.log({'epoch': epoch, 'EER/C_A_VI_A': EERs[1], 'FAR/C_A_VI_A': EER_FARs[1], 'FRR/C_A_VI_A': EER_FRRs[1]})
                wandb.log({'epoch': epoch, 'threshold/C_A_VI_A': EER_threshes[1]})

                print("\nCentroids: P  Verification Input: P "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[2], EER_threshes[2], EER_FARs[2], EER_FRRs[2]))
                wandb.log({'epoch': epoch, 'EER/C_P_VI_P': EERs[2], 'FAR/C_P_VI_P': EER_FARs[2], 'FRR/C_P_VI_P': EER_FRRs[2]})
                wandb.log({'epoch': epoch, 'threshold/C_P_VI_P': EER_threshes[2]})

                print("\nCentroids: AP  Verification Input: AP "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[3], EER_threshes[3], EER_FARs[3], EER_FRRs[3]))
                wandb.log({'epoch': epoch, 'EER/C_AP_VI_AP': EERs[3], 'FAR/C_AP_VI_AP': EER_FARs[3], 'FRR/C_AP_VI_AP': EER_FRRs[3]})
                wandb.log({'epoch': epoch, 'threshold/C_AP_VI_AP': EER_threshes[3]})
                

    return (extractor_a, extractor_p, converter)


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips/piezobuds/' # folder where stores the data for training and test
    train_data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips/piezobuds_new/train/'
    test_data_file_dir = '/mnt/hdd/gen/processed_data/wav_clips/piezobuds_new/test/'
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.001
    n_user = 69
    train_ratio = 0.9
    num_of_epoches = 800
    train_batch_size = 4
    test_batch_size = 3

    n_fft = 512  # Size of FFT, affects the frequency granularity
    hop_length = 256  # Typically n_fft // 4 (is None, then hop_length = n_fft // 2 by default)
    win_length = n_fft  # Typically the same as n_fft
    window_fn = torch.hann_window # Window function

    comment = 'ARET_w_conGlow_cap_wo_enroll_Huberloss'
    extractor_a = RET_v2(embedding_size=256, input_dim=80)
    extractor_p = RET_v2(embedding_size=256, input_dim=80)

    loaded_state = torch.load(pth_store_dir + 'pre_trained_model/aert_70.pth')
    state_a = extractor_a.state_dict()
    state_p = extractor_p.state_dict()
    for name, param in loaded_state.items():
        origname = name
        if name in state_a:
            if state_a[name].size() == loaded_state[origname].size():
                state_a[name].copy_(loaded_state[origname])
                state_p[name].copy_(loaded_state[origname])
    extractor_a.load_state_dict(state_a)
    extractor_p.load_state_dict(state_p)
    extractor_a.to(device)
    extractor_p.to(device)


    # only use the mel spectrum calculate part in ecapa-tdnn
    melextractor = ECAPA_TDNN(1024, is_stft=False)
    loaded_state = torch.load(pth_store_dir + 'pretrain_ecapa_tdnn.model')
    state_mel = melextractor.state_dict()
    for name, param in loaded_state.items():
        origname = name
        name = remove_prefix(origname, 'speaker_encoder.')
        if name in state_mel:
            if state_mel[name].size() == loaded_state[origname].size():
                state_mel[name].copy_(loaded_state[origname])
    melextractor.load_state_dict(state_mel)
    melextractor.to(device)

    # extractor_a = ECAPA_TDNN(1024, is_stft=False)
    # extractor_p = ECAPA_TDNN(1024, is_stft=False)

    # loaded_state = torch.load(pth_store_dir + 'pretrain_ecapa_tdnn.model')
    # state_a = extractor_a.state_dict()
    # state_p = extractor_p.state_dict()
    # for name, param in loaded_state.items():
    #     origname = name
    #     name = remove_prefix(origname, 'speaker_encoder.')
    #     if name in state_a:
    #         if state_a[name].size() == loaded_state[origname].size():
    #             state_a[name].copy_(loaded_state[origname])
    #             state_p[name].copy_(loaded_state[origname])
    # extractor_a.load_state_dict(state_a)
    # extractor_p.load_state_dict(state_p)
    extractor_a.to(device)
    extractor_p.to(device)


    ge2e_loss_a = GE2ELoss_ori(device).to(device)
    ge2e_loss_p = GE2ELoss_ori(device).to(device)
    ge2e_loss_c = GE2ELoss_ori(device).to(device)
    converter = conditionGlow(in_channel=1, n_flow=2, n_block=4).to(device)
    final_layer = FClayer().to(device)

    optimizer = torch.optim.Adam([
        {'params': extractor_a.parameters()},
        {'params': extractor_p.parameters()},
        {'params': ge2e_loss_a.parameters()},
        {'params': ge2e_loss_p.parameters()},
        {'params': ge2e_loss_c.parameters()},
        {'params': converter.parameters()},
        {'params': final_layer.parameters()},
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
    train_data_set = WavDatasetForVerification(train_data_file_dir, list(range(n_user)), 40)
    test_data_set = WavDatasetForVerification(test_data_file_dir, list(range(n_user)), 40)
    data_set = (train_data_set, test_data_set)

    loss_func = nn.HuberLoss()

    models = (extractor_a, extractor_p, converter, final_layer)
    ge2e_loss = (ge2e_loss_a, ge2e_loss_p, ge2e_loss_c)
    extractor_a, extractor_p, converter = train_and_test_model(device=device, models=models, ge2e_loss=ge2e_loss, loss_func=loss_func, data_set=data_set, optimizer=optimizer, scheduler=lr_scheduler,
                                                       train_batch_size=train_batch_size, test_batch_size=test_batch_size, model_final_path=model_final_path,
                                                       num_epochs=num_of_epoches, train_ratio=train_ratio, comment=comment)

    torch.save(extractor_a.state_dict(), model_final_path+'extractor_a.pth')
    torch.save(extractor_p.state_dict(), model_final_path+'extractor_p.pth')
    torch.save(converter.state_dict(), model_final_path+'converter.pth')

    
