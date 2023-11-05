import os
import sys
import time
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import STFTFeatureSet
from my_models import *
from torch.utils.tensorboard import SummaryWriter
from UniqueDraw import UniqueDraw
from utils import *


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


def train_and_test_model(device, model, ge2e_loss, train_set, test_set, optimizer, a_res, p_res, num_epochs=2000, train_seperate=False):
    # writer = SummaryWriter()
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
    batch_size = 5
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    if train_seperate:
        model_a, model_p = model

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # train and test model
        # for phase in ['train', 'test']:
        for phase in ['train', 'test']:
            if phase == 'train':
                # set model to training
                ge2e_loss.train()
                if train_seperate == False:
                    model.train()
                else:
                    model_a.train()
                    model_p.train()
                dataloader = train_loader
            else:
                # set model to test
                ge2e_loss.eval()
                if train_seperate == False:
                    model.eval()
                else:
                    model_a.eval()
                    model_p.eval()
                dataloader = test_loader
            
            # train each batch
            num_of_batches = 0
            loss_avg_batch_all = 0.0
            loss_avg_batch_aa = 0.0
            loss_avg_batch_pp = 0.0

            EERs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_FARs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_FRRs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
            EER_threshes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)

            for batch_id, (utterance, ids) in enumerate(dataloader):
                # get shape of input
                # number of speakers, number of utterances
                # features * 2 (256 * 2), time frame (1000)
                num_of_speakers, num_of_utters, w, l = utterance.shape
                num_of_batches = num_of_batches + 1

                with torch.set_grad_enabled(phase == 'train') and torch.autograd.set_detect_anomaly(True):
                    
                    if phase == 'train':
                        # training
                        # batch_size, time frame, length of feature
                        utterance = torch.reshape(utterance, (num_of_speakers * num_of_utters, w, l)).to(device)
                        piezos = utterance[: , : p_res, : ].to(device)
                        audios = utterance[: , w//2: w//2 + a_res, : ].to(device)
                        piezos = piezos.permute(0, 2, 1).to(device)
                        audios = audios.permute(0, 2, 1).to(device)

                        if train_seperate == False:
                            embeddings_piezo = model(piezos)
                            embeddings_audio = model(audios)
                        else:
                            embeddings_piezo = model_p(piezos)
                            embeddings_audio = model_a(audios)

                        # reshape the output to (number of speaker, number of utter, length of feature)
                        embeddings_piezo = embeddings_piezo.contiguous()
                        embeddings_piezo = embeddings_piezo.reshape([num_of_speakers, num_of_utters, -1])
                        embeddings_audio = embeddings_audio.contiguous()
                        embeddings_audio = embeddings_audio.reshape([num_of_speakers, num_of_utters, -1])
                        # embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters, -1))
                        # embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters, -1))

                        # loss calculation
                        embeddings_combined = torch.cat((embeddings_piezo, embeddings_audio), dim=1)
                        loss_combined, _ = ge2e_loss(embeddings_combined, embeddings_combined)
                        loss_aa, loss_pp = ge2e_loss(embeddings_audio, embeddings_piezo)
                        loss_extractor = loss_combined
                        # loss_extractor = loss_aa + loss_pp
                        # loss_extractor = loss_aa + loss_pp + loss_combined
                        loss_avg_batch_all += loss_extractor.item()
                        loss_avg_batch_aa += loss_aa.item()
                        loss_avg_batch_pp += loss_pp.item()
                        optimizer.zero_grad()
                        # loss_aa.backward()
                        # loss_pp.backward()
                        loss_extractor.backward()

                        if train_seperate == False:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                        else:
                            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 3.0)
                            torch.nn.utils.clip_grad_norm_(model_p.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
                        optimizer.step()

                    if phase == 'test':
                        num_of_speakers, num_of_utters, w, l = utterance.shape
                        enrollment_batch, verification_batch = torch.split(utterance, int(num_of_utters/2), dim=1)

                        enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, w, l)).to(device)
                        enrollment_batch = enrollment_batch.permute(0, 2, 1)
                        enrollment_piezo = enrollment_batch[: , : , : p_res].to(device)
                        enrollment_audio = enrollment_batch[: , : , w//2: w//2+a_res].to(device)
                        verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, w, l)).to(device)
                        verification_batch = verification_batch.permute(0, 2, 1)
                        verification_piezo = verification_batch[: , : , : p_res].to(device)
                        verification_audio = verification_batch[: , : , w//2: w//2+a_res].to(device)
                        
                        if train_seperate == False:
                            embeddings_piezo = model(enrollment_piezo)
                            embeddings_audio = model(enrollment_audio)
                        else:
                            embeddings_piezo = model_p(enrollment_piezo)
                            embeddings_audio = model_a(enrollment_audio)
                        
                        # embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters//2, output_dim))
                        # embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters//2, output_dim))
                        embeddings_piezo = embeddings_piezo.contiguous()
                        embeddings_piezo = embeddings_piezo.reshape([num_of_speakers, num_of_utters, -1])
                        embeddings_audio = embeddings_audio.contiguous()
                        embeddings_audio = embeddings_audio.reshape([num_of_speakers, num_of_utters, -1])

                        centroids_piezo = get_centroids(embeddings_piezo)
                        centroids_audio = get_centroids(embeddings_audio)

                        if train_seperate == False:
                            embeddings_piezo = model(verification_piezo)
                            embeddings_audio = model(verification_audio)
                        else:
                            embeddings_piezo = model_p(verification_piezo)
                            embeddings_audio = model_a(verification_audio)

                        # embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters//2, output_dim))
                        # embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters//2, output_dim))
                        embeddings_piezo = embeddings_piezo.contiguous()
                        embeddings_piezo = embeddings_piezo.reshape([num_of_speakers, num_of_utters, -1])
                        embeddings_audio = embeddings_audio.contiguous()
                        embeddings_audio = embeddings_audio.reshape([num_of_speakers, num_of_utters, -1])

                        loss_aa, loss_pp = ge2e_loss(embeddings_audio, embeddings_piezo)
                        loss_extractor = loss_aa + loss_pp
                        loss_avg_batch_all += loss_extractor.item()
                        loss_avg_batch_aa += loss_aa.item()
                        loss_avg_batch_pp += loss_pp.item()

                        # EER of Piezo
                        sim_matrix = get_modal_cossim(embeddings_piezo, centroids_piezo)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[0] += EER
                        EER_FARs[0] += EER_FAR
                        EER_FRRs[0] += EER_FRR
                        EER_threshes[0] += EER_thresh

                        # EER of Audio
                        sim_matrix = get_modal_cossim(embeddings_audio, centroids_audio)
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[1] += EER
                        EER_FARs[1] += EER_FAR
                        EER_FRRs[1] += EER_FRR
                        EER_threshes[1] += EER_thresh

                        # EER of Audio + Piezo
                        sim_matrix = get_modal_cossim(torch.cat((embeddings_audio, embeddings_piezo), dim=-1), 
                                        torch.cat((centroids_audio, centroids_piezo), dim=-1))
                        EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                        EERs[2] += EER
                        EER_FARs[2] += EER_FAR
                        EER_FRRs[2] += EER_FRR
                        EER_threshes[2] += EER_thresh

            epoch_loss_all = loss_avg_batch_all / num_of_batches
            epoch_loss_a = loss_avg_batch_aa / num_of_batches
            epoch_loss_p = loss_avg_batch_pp / num_of_batches

            print(f'{phase} Loss Extractor: {epoch_loss_all:.4f}')

            wandb.log({'epoch': epoch, f'Loss/{phase}_extractor': epoch_loss_all})
            wandb.log({'epoch': epoch, f'Loss/{phase}_loss_a': epoch_loss_a})
            wandb.log({'epoch': epoch, f'Loss/{phase}_loss_p': epoch_loss_p})

            if phase == 'test':
                EERs /= float(num_of_batches)
                EER_FARs /= float(num_of_batches)
                EER_FRRs /= float(num_of_batches)
                EER_threshes /= float(num_of_batches)

                print("\nCentroids: Piezo  Verification Input: Piezo "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[0], EER_threshes[0], EER_FARs[0], EER_FRRs[0]))
                wandb.log({'epoch': epoch, 'EER/C_P_VI_P': EERs[0], 'FAR/C_P_VI_P': EER_FARs[0], 'FRR/C_P_VI_P': EER_FRRs[0]})
                wandb.log({'epoch': epoch, 'threshold/C_P_VI_P': EER_threshes[0]})

                print("\nCentroids: Audio  Verification Input: Audio "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[1], EER_threshes[1], EER_FARs[1], EER_FRRs[1]))
                wandb.log({'epoch': epoch, 'EER/C_A_VI_A': EERs[1], 'FAR/C_A_VI_A': EER_FARs[1], 'FRR/C_A_VI_A': EER_FRRs[1]})
                wandb.log({'epoch': epoch, 'threshold/C_A_VI_A': EER_threshes[1]})

                print("\nCentroids: Audio + Piezo  Verification Input: Audio + Piezo "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[2], EER_threshes[2], EER_FARs[2], EER_FRRs[2]))
                wandb.log({'epoch': epoch, 'EER/C_AP_VI_AP': EERs[2], 'FAR/C_AP_VI_AP': EER_FARs[2], 'FRR/C_AP_VI_AP': EER_FRRs[2]})
                wandb.log({'epoch': epoch, 'threshold/C_AP_VI_AP': EER_threshes[2]})

    return model if not train_seperate else (model_a, model_p)


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # get the training and test data set
    ids = list(range(22))
    # valid_ids = [4, 7, 13, 21, 30, 36, 29, 41]
    # valid_ids = [1, 5, 15, 20, 31, 29, 35, 39]
    # valid_ids = [2, 8, 9, 14]
    valid_ids = [3, 6, 9, 14]
    # valid_ids = [3, 6, 8, 1]
    # valid_ids = [11, 12, 14, 16]
    train_ids = [item for item in ids if item not in valid_ids]
    print(train_ids)
    print(valid_ids)
    
    data_file_dir = './processed_data/stft/res_256_hop_128_t_160/' # folder where stores the data for training and test
    pth_store_dir = './pth_model/'
    os.makedirs(pth_store_dir, exist_ok=True)

    # set the params of each train
    # ----------------------------------------------------------------------------------------------------------------
    # Be sure to go through all the params before each run in case the models are saved in wrong folders!
    # ----------------------------------------------------------------------------------------------------------------
    
    lr = 0.001
    num_of_epoches = 1500
    train_seperate = False

    comment = 'old_model_out_128_b_2_u_20' # simple descriptions of specifications of this model, for example, 't_f' means we use the model which contains time and frequency nn layers
    
    ge2e_loss = GE2ELoss(device)
    
    if train_seperate == False:
        a_res = 256
        p_res = 256
        # extractor = Extractor_w_LSTM(device=device, layer_n=5, input_dim=a_res, output_dim=a_res).to(device)
        # extractor = Extractor_w_F_2D(device=device, f_len=a_res).to(device)
        # extractor = Extractor_w_F_1D(in_channel_n=128, device=device, f_len=a_res).to(device)
        extractor = extractor1DCNN(device=device, channel_number=160, in_feature=256, out_feature=128).to(device)
        optimizer = torch.optim.SGD([
            {'params': extractor.parameters()},
            {'params': ge2e_loss.parameters()}
        ], lr=lr)
        model_n = 1
    else:
        a_res = 512
        p_res = 256
        # extractor_a = Extractor_w_F_2D(device=device, f_len=a_res).to(device)
        # extractor_p = Extractor_w_F_2D(device=device, f_len=p_res).to(device)
        extractor_a = Extractor_w_LSTM(device=device, layer_n=5, input_dim=a_res, output_dim=a_res).to(device)
        extractor_p = Extractor_w_LSTM(device=device, layer_n=5, input_dim=p_res, output_dim=p_res).to(device)
        # extractor_a = Extractor_w_F_1D(in_channel_n=50, device=device, f_len=a_res).to(device)
        # extractor_p = Extractor_w_F_1D(in_channel_n=50, device=device, f_len=p_res).to(device)
        extractor = (extractor_a, extractor_p)
        optimizer = torch.optim.SGD([
            {'params': extractor_a.parameters()},
            {'params': extractor_p.parameters()},
            {'params': ge2e_loss.parameters()}
        ], lr=lr)
        model_n = 2 # number of models: wish to train one model jointly or 2 models seperately
    
    # create the folder to store the model
    model_struct = 'model_n_' + str(model_n) + '_res_a' + str(a_res) + '_p' + str(p_res) + '_' + comment
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
    train_set = STFTFeatureSet(False, train_ids, 20, data_file_dir)
    test_set = STFTFeatureSet(False, valid_ids, 20, data_file_dir)
    
    
    extractor = train_and_test_model(device=device, model=extractor, ge2e_loss=ge2e_loss, 
                                 train_set=train_set, test_set=test_set,
                                 optimizer=optimizer, a_res=a_res, p_res=p_res, 
                                 num_epochs=num_of_epoches, train_seperate=train_seperate)
    
    if train_seperate == False:
        torch.save(extractor.state_dict(), model_final_path+'m.pth')
    else:
        torch.save(extractor[0].state_dict(), model_final_path+'a.pth')
        torch.save(extractor[1].state_dict(), model_final_path+'p.pth')