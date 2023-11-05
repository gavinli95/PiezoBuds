import sys
import time
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import STFTFeatureSet
from my_models import CNNModel, LSTMWithAttention, GE2ELoss, ConverterNetwork, extractor1DCNN, Converter_w_triplet, Converter_ALAH2X
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from UniqueDraw import UniqueDraw
from utils import calc_loss, cal_intra_loss
from utils import get_centroids, get_cossim, get_modal_cossim, cosine_similarity, get_distance_matrix, draw_distance_matrix, get_centroids_kmeans
import random
import torch.nn.functional as F


feature_length = 128
hidden_feature_converter = 512
out_length_converter = 256


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
    for thres in torch.linspace(0.5, 1.0, 50):
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


def compute_ASR(sim_matrix, thres):
    """
    Compute Attack Success Rate and the threshold

    Args:
    - sim_matrix (torch.Tensor): A similarity matrix of shape 
      (num of speakers, num of utterances, num of speakers).

    Returns:
    - ASR
    - threshold
    """
    num_of_speakers, num_of_utters, _ = sim_matrix.shape
    
    # Initialize values
    ASR = float('inf')
    threshold = 0.0

    # Iterate over potential thresholds
    # for thres in torch.linspace(0.5, 1.0, 50):
    sim_matrix_thresh = sim_matrix > thres

    # Compute FAR and FRR
    t_ASR = torch.sum(sim_matrix_thresh) / num_of_speakers / num_of_utters

    # Update if this is the closest FAR and FRR we've seen so far
    if t_ASR < ASR:
        ASR = t_ASR
        threshold = thres

    return ASR, threshold

def train_and_test_model(device, extractor, converter, ge2e_loss, is_cnn, train_set, test_set, optimizer, optimizer_converter, num_epochs=2000):
    writer = SummaryWriter()
    # number of user for train and test in each epoch. Train 4 users, test 2 users.
    batch_size = 4
    # train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=True)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        extractor.eval()
        converter.eval()
        dataloader = test_loader
            
        # train each batch
        num_of_batches = 0
        loss_avg_batch_extractor = 0.0
        loss_avg_batch_converter = 0.0

        EERs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
        EER_FARs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
        EER_FRRs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)
        EER_threshes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)

        ASRs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(float)

        for batch_id, (utterance, ids) in enumerate(dataloader):
            # get shape of input
            # number of speakers, number of utterances
            # features * 2 (256 * 2), time frame (1000)
            num_of_speakers, num_of_utters, w, l = utterance.shape
            num_of_batches = num_of_batches + 1

            enrollment_batch, verification_batch = torch.split(utterance, int(num_of_utters/2), dim=1)
            if is_cnn:
                # batch_size, 1, length of feature, time frame
                enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, 1, w, l)).to(device)
                enrollment_piezo = enrollment_batch[:, :, :w//2, :].to(device)
                enrollment_audio = enrollment_batch[:, :, w//2:, :].to(device)
                verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, 1, w, l)).to(device)
                verification_piezo = verification_batch[:, :, :w//2, :].to(device)
                verification_audio = verification_batch[:, :, w//2:, :].to(device)
            else:
                # batch_size, time frame, length of feature
                enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, w, l)).to(device)
                enrollment_batch = enrollment_batch.permute(0, 2, 1)
                enrollment_piezo = enrollment_batch[:, :, :w//2].to(device)
                enrollment_audio = enrollment_batch[:, :, w//2:].to(device)
                verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, w, l)).to(device)
                verification_batch = verification_batch.permute(0, 2, 1)
                verification_piezo = verification_batch[:, :, :w//2].to(device)
                verification_audio = verification_batch[:, :, w//2:].to(device)

                # enrollment training
                EERs_tmp = np.array([0, 0, 0, 0, 0, 0, 0]).astype(float)
                EER_FARs_tmp = np.array([0, 0, 0, 0, 0, 0, 0]).astype(float)
                EER_FRRs_tmp = np.array([0, 0, 0, 0, 0, 0, 0]).astype(float)
                EER_threshes_tmp = np.array([0, 0, 0, 0, 0, 0, 0]).astype(float)
                centroids_x_all_users = []
                centroids_y_all_users = []
                converters = []

                for i in range(4):
                    enroller = i
                    others = list(range(4)).remove(enroller)
                    converter_tmp = ConverterNetwork(256, hidden_feature_converter).to(device)
                    converter_tmp.load_state_dict(converter.state_dict())
                    ge2e_tmp = GE2ELoss(device)
                    ge2e_tmp.load_state_dict(ge2e_loss.state_dict())
                    optimizer_temp = torch.optim.SGD([
                                        {'params': converter_tmp.parameters()},
                                        {'params': ge2e_tmp.parameters()}
                                    ], lr=lr, momentum=0.9, nesterov=True)
                    
                    # train the model for enroller
                    with torch.set_grad_enabled(True):
                        extractor.train()
                        converter_tmp.train()

                        # enrollment_piezo = enrollment_piezo[:, :, :128]
                        # enrollment_audio_high = enrollment_audio[:, :, 128:]
                        # enrollment_audio_low = enrollment_audio[:, :, :128]    

                        for e in range(5):
                            # embeddings_x = converter(embeddings_piezo, embeddings_audio)
                            embeddings_piezo, _ = extractor(enrollment_piezo)
                            embeddings_audio, _ = extractor(enrollment_audio)
                            # embeddings_audio_l, _ = extractor(enrollment_audio_low)
                            embeddings_x, embeddings_y = converter_tmp(embeddings_audio, embeddings_piezo)

                            _, output_dim = embeddings_piezo.shape

                            embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters//2, output_dim))
                            embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters//2, output_dim))
                            # embeddings_audio_h = torch.reshape(embeddings_audio_h, (num_of_speakers, num_of_utters//2, output_dim))

                            embeddings_x = torch.reshape(embeddings_x, (num_of_speakers, num_of_utters // 2, output_dim))
                            embeddings_y = torch.reshape(embeddings_y, (num_of_speakers, num_of_utters // 2, output_dim))

                            centroids_x = get_centroids(embeddings_x)
                            centroids_piezo = get_centroids(embeddings_piezo)
                            centroids_y = get_centroids(embeddings_y)

                            # embeddings_x2y = torch.cat((embeddings_x, embeddings_y), dim=-1) # converter out dim * 2
                            # embeddings_x2y_w_ah = torch.cat((embeddings_x2y, embeddings_audio_h), dim=-1) # converter out dim * 2 + extractor out dim
                            # embeddings_x2y_w_ah_current_enroller = embeddings_x2y_w_ah[enroller].unsqueeze(0)
                            combined = torch.cat((embeddings_y, embeddings_piezo), dim=-1)
                            loss_yp, _ = ge2e_loss(combined, combined)
                            combined = torch.cat((embeddings_x, embeddings_piezo), dim=-1)
                            loss_xp, _ = ge2e_loss(combined, combined)

                            sim_matrix_y2p = get_modal_cossim(embeddings_y, centroids_piezo)
                            sim_matrix_p2y = get_modal_cossim(embeddings_piezo, centroids_y)
                            loss_converter_y2p, loss_converter_y2p_user = cal_intra_loss(sim_matrix_y2p, device)
                            loss_converter_p2y, loss_converter_p2y_user = cal_intra_loss(sim_matrix_p2y, device)
                            sim_matrix_x2p = get_modal_cossim(embeddings_x, centroids_piezo)
                            sim_matrix_p2x = get_modal_cossim(embeddings_piezo, centroids_x)
                            loss_converter_x2p, loss_converter_x2p_user = cal_intra_loss(sim_matrix_x2p, device)
                            loss_converter_p2x, loss_converter_p2x_user = cal_intra_loss(sim_matrix_p2x, device)

                            loss_converter_dis = (torch.sum(loss_converter_y2p_user[enroller]) + torch.sum(loss_converter_p2y_user[enroller])
                                                  + torch.sum(loss_converter_x2p_user[enroller]) + torch.sum(loss_converter_p2x_user[enroller]))/4

                            loss_converter = (loss_yp + loss_xp)/2 + loss_converter_dis

                            optimizer_temp.zero_grad()
                            loss_converter.backward()
                            torch.nn.utils.clip_grad_norm_(converter_tmp.parameters(), 3.0)
                            torch.nn.utils.clip_grad_norm_(ge2e_tmp.parameters(), 1.0)
                            optimizer_temp.step()

                    with torch.set_grad_enabled(False):
                        extractor.eval()
                        converter_tmp.eval()
                        # get centroids from enrollment
                        # embeddings_x = converter(embeddings_piezo, embeddings_audio)
                        embeddings_piezo, _ = extractor(enrollment_piezo)
                        embeddings_audio, _ = extractor(enrollment_audio)
                        # embeddings_audio_l, _ = extractor(enrollment_audio_low)

                        embeddings_x, embeddings_y = converter_tmp(embeddings_audio, embeddings_piezo)
                        embeddings_x = torch.reshape(embeddings_x, (num_of_speakers, num_of_utters // 2, output_dim))
                        embeddings_y = torch.reshape(embeddings_y, (num_of_speakers, num_of_utters // 2, output_dim))

                        _, output_dim = embeddings_piezo.shape
                        
                        embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters//2, output_dim))
                        embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters//2, output_dim))
                        # embeddings_audio_h = torch.reshape(embeddings_audio_h, (num_of_speakers, num_of_utters//2, output_dim))

                        centroids_audio = get_centroids(embeddings_audio)
                        # centroids_audio_l = get_centroids(embeddings_audio_l)
                        centroids_piezo = get_centroids(embeddings_piezo)
                        centroids_x = get_centroids(embeddings_x)
                        centroids_y = get_centroids(embeddings_y)

                        centroids_x_all_users.append(centroids_x[enroller])
                        centroids_y_all_users.append(centroids_y[enroller])
                        converters.append(converter_tmp)

                #calculate EER: using similarity converter(audio) with piezo
                white_noise = np.random.rand(num_of_speakers, num_of_utters // 2, output_dim)
                white_noise = torch.from_numpy(white_noise)
                white_noise = white_noise.to(device)
                white_noise_sum = torch.sum(white_noise, dim=-1).unsqueeze(-1)
                white_noise = white_noise / white_noise_sum

                # verification_piezo = verification_piezo[:, :, :128]
                # verification_audio_high = verification_audio[:, :, 128:]
                # verification_audio_low = verification_piezo[:, :, :128]

                embeddings_piezo, _ = extractor(verification_piezo)
                embeddings_audio, _ = extractor(verification_audio)
                # embeddings_audio_l, _ = extractor(verification_audio_low)
                
                embeddings_x_each_user = []
                embeddings_y_each_user = []
                embeddings_y_each_user_with_white_noise_attack = []
                embeddings_y_each_user_with_mimic_piezo_attack = []
                for x in range(4):
                    embeddings_x, embeddings_y = converters[x](embeddings_audio, embeddings_piezo)
                    embeddings_x_tmp = torch.reshape(embeddings_x, (num_of_speakers, num_of_utters // 2, out_length_converter))
                    embeddings_y_tmp = torch.reshape(embeddings_y, (num_of_speakers, num_of_utters // 2, out_length_converter))
                    embeddings_x_each_user.append(embeddings_x_tmp[x])
                    embeddings_y_each_user.append(embeddings_y_tmp[x])

                # get verification embeddings
                embeddings_piezo, _ = extractor(verification_piezo)
                embeddings_audio, _ = extractor(verification_audio)
                # embeddings_audio_l, _ = extractor(verification_audio_low)
                _, output_dim = embeddings_piezo.shape

                embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters//2, output_dim))
                embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters//2, output_dim))
                #embeddings_audio = torch.reshape(embeddings_audio_h, (num_of_speakers, num_of_utters//2, output_dim))
                
                centroids_x = centroids_x_all_users[0]
                centroids_y = centroids_y_all_users[0]
                centroids_x = centroids_x.unsqueeze(0)
                centroids_y = centroids_y.unsqueeze(0)
                for j in range(1, 4):
                    centroids_x = torch.cat((centroids_x, centroids_x_all_users[j].unsqueeze(0)))
                    centroids_y = torch.cat((centroids_y, centroids_y_all_users[j].unsqueeze(0)))
                centroids_x = centroids_x.to(device)
                centroids_y = centroids_y.to(device)

                embeddings_x = embeddings_x_each_user[0]
                embeddings_y = embeddings_y_each_user[0]
                embeddings_x = embeddings_x.unsqueeze(0)
                embeddings_y = embeddings_y.unsqueeze(0)
                for j in range(1, 4):
                    embeddings_x = torch.cat((embeddings_x, embeddings_x_each_user[j].unsqueeze(0)))
                    embeddings_y = torch.cat((embeddings_y, embeddings_y_each_user[j].unsqueeze(0)))
                # get EER using different combination
                # C means centroids, VI means verification input

                # C XP, VI XP
                # sim_matrix = get_modal_cossim(embeddings_piezo, centroids_x)
                sim_matrix = get_modal_cossim(torch.cat((embeddings_x, embeddings_piezo), dim=-1), 
                                              torch.cat((centroids_x, centroids_piezo), dim=-1))
                EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                EERs[-1] += EER
                EER_FARs[-1] += EER_FAR
                EER_FRRs[-1] += EER_FRR
                EER_threshes[-1] += EER_thresh
                # emulate attack
                # replay attack, audio + white noise
                # sim_matrix = get_modal_cossim(torch.cat((embeddings_x, white_noise), dim=-1), 
                #                               torch.cat((centroids_x, centroids_piezo), dim=-1))
                # ASR_aw, _ = compute_ASR(sim_matrix, EER_thresh)
                # # audio + audio
                # sim_matrix = get_modal_cossim(torch.cat((embeddings_x, embeddings_audio), dim=-1), 
                #                               torch.cat((centroids_x, centroids_piezo), dim=-1))
                # ASR_aa, _ = compute_ASR(sim_matrix, EER_thresh)
                # ASR = (ASR_aw + ASR_aa) / 2
                # ASRs[-1] += ASR

                # C piezo, VI piezo
                sim_matrix = get_modal_cossim(embeddings_piezo, centroids_piezo)
                EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                EERs[0] += EER
                EER_FARs[0] += EER_FAR
                EER_FRRs[0] += EER_FRR
                EER_threshes[0] += EER_thresh

                # C audio, VI audio
                sim_matrix = get_modal_cossim(embeddings_audio, centroids_audio)
                EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                EERs[1] += EER
                EER_FARs[1] += EER_FAR
                EER_FRRs[1] += EER_FRR
                EER_threshes[1] += EER_thresh

                # C Y, VI Y
                sim_matrix = get_modal_cossim(embeddings_y, centroids_y)
                EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                EERs[2] += EER
                EER_FARs[2] += EER_FAR
                EER_FRRs[2] += EER_FRR
                EER_threshes[2] += EER_thresh


                # C YP, VI YP
                # sim_matrix = get_modal_cossim(embeddings_piezo, centroids_x)
                sim_matrix = get_modal_cossim(torch.cat((embeddings_y, embeddings_piezo), dim=-1), 
                                              torch.cat((centroids_y, centroids_piezo), dim=-1))
                EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                EERs[3] += EER
                EER_FARs[3] += EER_FAR
                EER_FRRs[3] += EER_FRR
                EER_threshes[3] += EER_thresh


                # C audio + piezo, VI audio + piezo
                sim_matrix = get_modal_cossim(torch.cat((embeddings_audio, embeddings_piezo), dim=-1), 
                                        torch.cat((centroids_audio, centroids_piezo), dim=-1))
                EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(sim_matrix)
                EERs[4] += EER
                EER_FARs[4] += EER_FAR
                EER_FRRs[4] += EER_FRR
                EER_threshes[4] += EER_thresh
                # emulate attack
                # replay attack, audio + white noise
                # sim_matrix = get_modal_cossim(torch.cat((embeddings_audio, white_noise), dim=-1), 
                #                               torch.cat((centroids_audio, centroids_piezo), dim=-1))
                # ASR_aw, _ = compute_ASR(sim_matrix, EER_thresh)
                # # audio + audio
                # sim_matrix = get_modal_cossim(torch.cat((embeddings_audio, embeddings_audio), dim=-1), 
                #                               torch.cat((centroids_audio, centroids_piezo), dim=-1))
                # ASR_aa, _ = compute_ASR(sim_matrix, EER_thresh)
                # ASR = (ASR_aw + ASR_aa) / 2
                # ASRs[4] += ASR
                        
            wandb.define_metric("epoch")
            wandb.define_metric("Loss/*", step_metric="epoch")
            wandb.define_metric("EER/*", step_metric="epoch")
            wandb.define_metric("FAR/*", step_metric="epoch")
            wandb.define_metric("FRR/*", step_metric="epoch")
            wandb.define_metric("ASR/*", step_metric="epoch")

            EERs /= np.float(num_of_batches)
            EER_FARs /= np.float(num_of_batches)
            EER_FRRs /= np.float(num_of_batches)
            EER_threshes /= np.float(num_of_batches)
            ASRs /= np.float(num_of_batches)

            print("\nCentroids: Piezo  Verification Input: Piezo "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[0], EER_threshes[0], EER_FARs[0], EER_FRRs[0]))
            wandb.log({'epoch': epoch, 'EER/C_P_VI_P': EERs[0], 'FAR/C_P_VI_P': EER_FARs[0], 'FRR/C_P_VI_P': EER_FRRs[0]})

            print("\nCentroids: Audio  Verification Input: Audio "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[1], EER_threshes[1], EER_FARs[1], EER_FRRs[1]))
            wandb.log({'epoch': epoch, 'EER/C_A_VI_A': EERs[1], 'FAR/C_A_VI_A': EER_FARs[1], 'FRR/C_A_VI_A': EER_FRRs[1]})

            print("\nCentroids: Y  Verification Input: Y "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[2], EER_threshes[2], EER_FARs[2], EER_FRRs[2]))
            wandb.log({'epoch': epoch, 'EER/C_Y_VI_Y': EERs[2], 'FAR/C_Y_VI_Y': EER_FARs[2], 'FRR/C_Y_VI_Y': EER_FRRs[2]})

            print("\nCentroids: Y + Piezo  Verification Input: Y + Piezo "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[3], EER_threshes[3], EER_FARs[3], EER_FRRs[3]))
            wandb.log({'epoch': epoch, 'EER/C_YP_VI_YP': EERs[3], 'FAR/C_YP_VI_YP': EER_FARs[3], 'FRR/C_YP_VI_YP': EER_FRRs[3]})
            # wandb.log({'epoch': epoch, 'ASR/C_YP_VI_YP': ASRs[3]})

            print("\nCentroids: Audio + Piezo  Verification Input: Audio + Piezo "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[4], EER_threshes[4], EER_FARs[4], EER_FRRs[4]))
            wandb.log({'epoch': epoch, 'EER/C_AP_VI_AP': EERs[4], 'FAR/C_AP_VI_AP': EER_FARs[4], 'FRR/C_AP_VI_AP': EER_FRRs[4]})
            # wandb.log({'epoch': epoch, 'ASR/C_AP_VI_AP': ASRs[4]})
          
            print("\nCentroids: XP  Verification Input: XP "
                            "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EERs[-1], EER_threshes[-1], EER_FARs[-1], EER_FRRs[-1]))
            wandb.log({'epoch': epoch, 'EER/C_XP_VI_XP': EERs[-1], 'FAR/C_XP_VI_XP': EER_FARs[-1], 'FRR/C_XP_VI_XP': EER_FRRs[-1]})
            # wandb.log({'epoch': epoch, 'ASR/C_XP_VI_XP': ASRs[-1]})


    return model, converter


if __name__ == "__main__":

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(device)
    # initialize the wandb configuration
    time_stamp = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
    wandb.init(
        # team name
        entity="piezobuds",

        # set the project name
        project="PiezoBuds",

        # params of the task
        # TODO: add configuration summaray to the time stamp
        name=time_stamp
    )
    drawer = UniqueDraw()

    ids = list(range(42))

    valid_ids = [4, 7, 13, 21, 30, 36, 29, 41]
    # valid_ids = [4, 7, 13, 21]
    # valid_ids = [2, 8, 9, 14]
    # valid_ids = [3, 6, 9, 14]
    # valid_ids = [3, 6, 8, 1]
    # valid_ids = [11, 12, 14, 16]
    # valid_ids = drawer.draw_numbers(22, 4)
    train_ids = [item for item in ids if item not in valid_ids]
    print(train_ids)
    print(valid_ids)

    if len(sys.argv) != 5:
        # model = LSTMWithAttention(256, 128, 2, 128).to(device)
        model = extractor1DCNN(channel_number=160,
                               in_feature=256,
                               out_feature=feature_length,
                               device=device).to(device)
        is_CNN = False
        num_of_epoches = 50
        store_path = 'extractor_{}_4_1.pth'.format(feature_length)
        lr = 0.001
        model.load_state_dict(torch.load(store_path))
    else:
        if sys.argv[1] == '0':
            # CNN
            model = CNNModel(128).to(device)
            is_CNN = True
        else:
            # LSTM
            model = LSTMWithAttention(256, 128, 3, 128).to(device)
            is_CNN = False
        num_of_epoches = int(sys.argv[2])
        store_path = sys.argv[3]
        lr = float(sys.argv[4])
    converter = ConverterNetwork(256, hidden_feature_converter).to(device)
    converter.load_state_dict(torch.load('converter_{}_{}_ExEy_4.pth'.format(256, hidden_feature_converter)))
    # converter.apply(ConverterNetwork1DCNN.weights_init_xavier)
    
    train_set = STFTFeatureSet(False, train_ids, 30)
    test_set = STFTFeatureSet(False, valid_ids, 30)
    ge2e_loss = GE2ELoss(device)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()}
    ], lr=lr)
    optimizer_converter = torch.optim.SGD([
        {'params': converter.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=lr, momentum=0.9, nesterov=True)
    model, converter = train_and_test_model(device, model, converter, ge2e_loss, is_CNN, train_set, test_set, optimizer, optimizer_converter, num_of_epoches)
    # torch.save(converter.state_dict(), 'converter.pth')