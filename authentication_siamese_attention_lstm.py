import sys
import wandb
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import LatentFeatureSet, FrequencyFeatureSet, STFTFeatureSet
import tensorboard
from conv1dClassify import Conv1DNet, Classifier, Conv1DAutoencoder, SpeechEmbedder
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from UniqueDraw import UniqueDraw
import torch.nn.functional as F
from conv1dClassify import GE2ELoss
import random
from cnn_lstm_attention import ParallelModel
from sklearn.cluster import KMeans
from attention_lstm import STFTAttentionModel

from utils import get_centroids, get_cossim, cosine_similarity, get_distance_matrix, draw_distance_matrix, get_centroids_kmeans, get_modal_cossim


def cal_EER(sim_matrix, num_of_speakers, num_of_utters):
    # calculating EER
    diff = float('inf')
    EER = 0
    EER_thresh = 0
    EER_FAR = 0
    EER_FRR = 0

    for thres in [0.01 * i + 0.5 for i in range(51)]:
        sim_matrix_thresh = sim_matrix > thres

        FAR = (sum([sim_matrix_thresh[i].float().sum() - sim_matrix_thresh[i, :, i].float().sum()
                    for i in range(int(num_of_speakers))])
               / (num_of_speakers - 1.0) / (float(num_of_utters / 2)) / num_of_speakers)

        FRR = (sum([num_of_utters / 2 - sim_matrix_thresh[i, :, i].float().sum() for i in
                    range(int(num_of_speakers))])
               / (float(num_of_utters / 2)) / num_of_speakers)

        # Save threshold when FAR = FRR (=EER)
        if diff > abs(FAR - FRR):
            diff = abs(FAR - FRR)
            EER = (FAR + FRR) / 2
            EER_thresh = thres
            EER_FAR = FAR
            EER_FRR = FRR
    return EER, EER_thresh, EER_FAR, EER_FRR


def calculate_l2_distance(tensor1, tensor2, dim=None):
    differences = tensor1 - tensor2
    squared_differences = torch.square(differences)
    summed_squared_differences = torch.sum(squared_differences, dim=dim)
    l2_distance = torch.sqrt(summed_squared_differences)
    return l2_distance


def train_test_model(device, models, train_set, test_set, criterion, optimizers, train_ratio=0.7, num_epochs=3000):
    writer = SummaryWriter()

    batch_size = 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                models['piezo'].train()  # Set model to training mode
                models['audio'].train()
                dataloader = train_loader
            else:
                models['piezo'].eval()   # Set model to evaluate mode
                models['audio'].eval()
                dataloader = test_loader

            avg_EER = 0
            running_loss = 0.0
            running_loss_ge2e = 0.0
            running_loss_out_audio = 0.0
            # Iterate over data.
            batch_avg_EER = 0
            for batch_id, (utterance, ids) in enumerate(dataloader):
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    num_of_speakers, num_of_utters, w, l = utterance.shape

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        utterance = torch.reshape(utterance, (num_of_speakers * num_of_utters, 1, w, l)).to(
                            device)
                        piezos = utterance[:, :, :w//2, :].to(device)
                        audios = utterance[:, :, w//2:, :].to(device)

                        embeddings = models['piezo'](piezos)
                        embeddings_audio = models['piezo'](audios)
                        _, num_of_fea = embeddings.shape
                        embeddings = torch.reshape(embeddings, (num_of_speakers, num_of_utters, num_of_fea))
                        embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters, num_of_fea))

                        embeddings_connect = torch.cat((embeddings, embeddings_audio), dim=1)
                        loss_ge2e_connected = ge2e_loss_piezo(embeddings_connect)
                        loss_ge2e_piezo = ge2e_loss_piezo(embeddings)
                        loss_ge2e_audio = ge2e_loss_piezo(embeddings_audio)
                        loss_audio = calculate_l2_distance(embeddings, embeddings_audio, dim=1).mean()
                        # loss = loss_ge2e + 100 * loss_audio
                        loss = loss_ge2e_connected
                        # loss = loss_audio

                        optimizers['piezo'].zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models['piezo'].parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_piezo.parameters(), 1.0)
                        optimizers['piezo'].step()

                        # optimizers['audio'].zero_grad()
                        # loss_ge2e_audio.backward()
                        # torch.nn.utils.clip_grad_norm_(models['audio'].parameters(), 1.0)
                        # torch.nn.utils.clip_grad_norm_(ge2e_loss_audio.parameters(), 1.0)
                        # optimizers['audio'].step()
                    if phase == 'test':
                        enrollment_batch, verification_batch = torch.split(utterance, int(num_of_utters/2), dim=1)
                        enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, 1, w, l))
                        verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, 1, w, l))

                        perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
                        unperm = list(perm)
                        for i, j in enumerate(perm):
                            unperm[j] = i

                        verification_batch = verification_batch[perm]

                        enrollment_embeddings = models['piezo'](enrollment_batch[:, :, :w//2].to(device))
                        enrollment_embeddings_audio = models['piezo'](enrollment_batch[:, :, w//2:].to(device))
                        verification_embeddings = models['piezo'](verification_batch[:, :, :w//2].to(device))
                        verification_embeddings_audio = models['piezo'](verification_batch[:, :, w//2:].to(device))
                        verification_embeddings = verification_embeddings[unperm]
                        verification_embeddings_audio = verification_embeddings_audio[unperm]
                        # enrollment_embeddings = torch.cat((enrollment_embeddings, enrollment_embeddings_audio), dim=1)
                        # verification_embeddings = torch.cat((verification_embeddings, verification_embeddings_audio), dim=1)

                        enrollment_embeddings = torch.reshape(enrollment_embeddings, (num_of_speakers, num_of_utters // 2, num_of_fea))
                        enrollment_embeddings_audio = torch.reshape(enrollment_embeddings_audio, (num_of_speakers, num_of_utters // 2, num_of_fea))
                        # norms = torch.norm(enrollment_embeddings, p=2, dim=2, keepdim=True)
                        # enrollment_embeddings = enrollment_embeddings / norms

                        verification_embeddings = torch.reshape(verification_embeddings, (
                        num_of_speakers, num_of_utters // 2, num_of_fea))
                        verification_embeddings_audio = torch.reshape(verification_embeddings_audio, (
                            num_of_speakers, num_of_utters // 2, num_of_fea))
                        # norms = torch.norm(verification_embeddings, p=2, dim=2, keepdim=True)
                        # verification_embeddings = verification_embeddings / norms

                        # enrollment_centroids = np.array(get_centroids_kmeans(enrollment_embeddings))
                        # enrollment_centroids = torch.from_numpy(enrollment_centroids).to(device).float()
                        # enrollment_centroids_audio = np.array(get_centroids_kmeans(enrollment_embeddings_audio))
                        # enrollment_centroids_audio = torch.from_numpy(enrollment_centroids_audio).to(device).float()
                        enrollment_centroids = get_centroids(enrollment_embeddings)
                        enrollment_centroids_audio = get_centroids(enrollment_embeddings_audio)

                        distance_matrix_centroids = get_distance_matrix(torch.cat((enrollment_centroids, enrollment_centroids_audio), dim=0))

                        EERs = []; EER_FARs = []; EER_FRRs = []
                        user1_audio_centroid = enrollment_centroids_audio[0, :]
                        user1_piezo_centroid = enrollment_centroids[0, :]
                        user1_audio_centroid = torch.reshape(user1_audio_centroid, (1, 256))
                        user1_piezo_centroid = torch.reshape(user1_piezo_centroid, (1, 256))
                        user1_audio_embeddings = verification_embeddings_audio[0]
                        user1_piezo_embeddings = verification_embeddings[0]
                        tmp = torch.cat((user1_piezo_centroid, user1_piezo_embeddings, user1_audio_centroid, user1_audio_embeddings), dim=0)
                        distance_matrix = get_distance_matrix(tmp)

                        # centroids: piezo  verification input: piezo
                        sim_matrix = get_modal_cossim(verification_embeddings, enrollment_centroids)
                        EER, EER_thresh, EER_FAR, EER_FRR = cal_EER(sim_matrix, num_of_speakers, num_of_utters)
                        print("\nCentroids: Piezo  Verification Input: Piezo "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
                        EERs.append(EER); EER_FARs.append(EER_FAR); EER_FRRs.append(EER_FRR)

                        # centroids: audio  verification input: audio
                        sim_matrix = get_modal_cossim(verification_embeddings_audio, enrollment_centroids_audio)
                        # draw_weighted_graph(adjacency_matrix, './distance_graph/test.png')
                        EER, EER_thresh, EER_FAR, EER_FRR = cal_EER(sim_matrix, num_of_speakers, num_of_utters)
                        print("\nCentroids: Audio  Verification Input: Audio "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
                        EERs.append(EER); EER_FARs.append(EER_FAR); EER_FRRs.append(EER_FRR)

                        # centroids: piezo  verification input: audio
                        sim_matrix = get_modal_cossim(verification_embeddings_audio, enrollment_centroids)
                        EER, EER_thresh, EER_FAR, EER_FRR = cal_EER(sim_matrix, num_of_speakers, num_of_utters)
                        print("\nCentroids: Piezo  Verification Input: Audio "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
                        EERs.append(EER); EER_FARs.append(EER_FAR); EER_FRRs.append(EER_FRR)

                        # centroids: audio  verification input: piezo
                        sim_matrix = get_modal_cossim(verification_embeddings, enrollment_centroids_audio)
                        EER, EER_thresh, EER_FAR, EER_FRR = cal_EER(sim_matrix, num_of_speakers, num_of_utters)
                        print("\nCentroids: Audio  Verification Input: Piezo "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
                        EERs.append(EER); EER_FARs.append(EER_FAR); EER_FRRs.append(EER_FRR)

                        # centroids: audio + piezo  verification input: audio + piezo
                        sim_matrix = get_modal_cossim(torch.cat((verification_embeddings_audio, verification_embeddings), dim=-1),
                                                torch.cat((enrollment_centroids_audio, enrollment_centroids), dim=-1))
                        EER, EER_thresh, EER_FAR, EER_FRR = cal_EER(sim_matrix, num_of_speakers, num_of_utters)
                        print("\nCentroids: Audio + Piezo  Verification Input: Audio + Piezo "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
                        EERs.append(EER); EER_FARs.append(EER_FAR); EER_FRRs.append(EER_FRR)

                        # centroids: audio + piezo  verification input: piezo + audio
                        sim_matrix = get_modal_cossim(
                            torch.cat((verification_embeddings_audio, verification_embeddings), dim=-1),
                            torch.cat((enrollment_centroids, enrollment_centroids_audio), dim=-1))
                        EER, EER_thresh, EER_FAR, EER_FRR = cal_EER(sim_matrix, num_of_speakers, num_of_utters)
                        print("\nCentroids: Audio + Piezo  Verification Input: Piezo + Audio "
                              "\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
                        EERs.append(EER); EER_FARs.append(EER_FAR); EER_FRRs.append(EER_FRR)


                # statistics
                running_loss += loss.item()
                running_loss_ge2e += loss_ge2e_audio.item()
                running_loss_out_audio += loss_audio.item()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_loss_ge2e = running_loss_ge2e / len(dataloader.dataset)
            epoch_loss_out_audio = running_loss_out_audio / len(dataloader.dataset)
            # epoch_acc = running_corrects.float() / len(dataloader.dataset)

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # configure the metric w/ step metric
            wandb.define_metric("epoch")
            wandb.define_metric("Loss/*", step_metric="epoch")
            wandb.define_metric("Distance/*", step_metric="epoch")
            wandb.define_metric("EER/*", step_metric="epoch")
            wandb.define_metric("FAR/*", step_metric="epoch")
            wandb.define_metric("FRR/*", step_metric="epoch")

            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Loss/train_ge2e', epoch_loss_ge2e, epoch)
                writer.add_scalar('Loss/train_out_audio', epoch_loss_out_audio, epoch)
                # add to the wandb
                wandb.log({'epoch': epoch, 'Loss/train': epoch_loss, 'Loss/train_ge2e': epoch_loss_ge2e, 'Loss/train_out_audio': epoch_loss_out_audio})
            else:
                # if epoch % 50 == 0:
                    # draw_distance_matrix(distance_matrix, './distance_graph/{}.png'.format(epoch))
                # distance
                writer.add_scalar('Distance/CP1_to_CP2', distance_matrix_centroids[0][1], epoch)
                writer.add_scalar('Distance/CP1_to_CA1', distance_matrix_centroids[0][2], epoch)
                writer.add_scalar('Distance/CP1_to_CA2', distance_matrix_centroids[0][3], epoch)
                writer.add_scalar('Distance/CP2_to_CA1', distance_matrix_centroids[1][2], epoch)
                writer.add_scalar('Distance/CP2_to_CA2', distance_matrix_centroids[1][3], epoch)
                writer.add_scalar('Distance/CA1_to_CA2', distance_matrix_centroids[2][3], epoch)

                # add to the wandb
                wandb.log({'epoch': epoch, 'Distance/CP1_to_CP2': distance_matrix_centroids[0][1], 'Distance/CP1_to_CA1': distance_matrix_centroids[0][2],
                           'Distance/CP1_to_CA2': distance_matrix_centroids[0][3], 'Distance/CP2_to_CA1': distance_matrix_centroids[1][2],
                           'Distance/CP2_to_CA2': distance_matrix_centroids[1][3], 'Distance/CA1_to_CA2': distance_matrix_centroids[2][3]})

                # Centroids: Piezo, Verification Input: Piezo
                writer.add_scalar('EER/C_P_VI_P', EERs[0], epoch)
                writer.add_scalar('FAR/C_P_VI_P', EER_FARs[0], epoch)
                writer.add_scalar('FRR/C_P_VI_P', EER_FRRs[0], epoch)

                # add to the wandb
                wandb.log({'epoch': epoch, 'EER/C_P_VI_P': EERs[0], 'FAR/C_P_VI_P': EER_FARs[0], 'FRR/C_P_VI_P': EER_FRRs[0]})

                # Centroids: Audio, Verification Input: Audio
                writer.add_scalar('EER/C_A_VI_A', EERs[1], epoch)
                writer.add_scalar('FAR/C_A_VI_A', EER_FARs[1], epoch)
                writer.add_scalar('FRR/C_A_VI_A', EER_FRRs[1], epoch)

                wandb.log({'epoch': epoch, 'EER/C_A_VI_A': EERs[1], 'FAR/C_A_VI_A': EER_FARs[1], 'FRR/C_A_VI_A': EER_FRRs[1]})

                # Centroids: Piezo, Verification Input: Audio
                writer.add_scalar('EER/C_P_VI_A', EERs[2], epoch)
                writer.add_scalar('FAR/C_P_VI_A', EER_FARs[2], epoch)
                writer.add_scalar('FRR/C_P_VI_A', EER_FRRs[2], epoch)

                wandb.log({'epoch': epoch, 'EER/C_P_VI_A': EERs[2], 'FAR/C_P_VI_A': EER_FARs[2], 'FRR/C_P_VI_A': EER_FRRs[2]})

                # Centroids: Audio, Verification Input: Piezo
                writer.add_scalar('EER/C_A_VI_P', EERs[3], epoch)
                writer.add_scalar('FAR/C_A_VI_P', EER_FARs[3], epoch)
                writer.add_scalar('FRR/C_A_VI_P', EER_FRRs[3], epoch)
                
                wandb.log({'epoch': epoch, 'EER/C_A_VI_P': EERs[3], 'FAR/C_A_VI_P': EER_FARs[3], 'FRR/C_A_VI_P': EER_FRRs[3]})

                # Centroids: Audio + Piezo, Verification Input: Audio + Piezo
                writer.add_scalar('EER/C_AP_VI_AP', EERs[4], epoch)
                writer.add_scalar('FAR/C_AP_VI_AP', EER_FARs[4], epoch)
                writer.add_scalar('FRR/C_AP_VI_AP', EER_FRRs[4], epoch)

                wandb.log({'epoch': epoch, 'EER/C_AP_VI_AP': EERs[4], 'FAR/C_AP_VI_AP': EER_FARs[4], 'FRR/C_AP_VI_AP': EER_FRRs[4]})

                # Centroids: Audio + Piezo, Verification Input: Piezo + Audio
                writer.add_scalar('EER/C_AP_VI_PA', EERs[5], epoch)
                writer.add_scalar('FAR/C_AP_VI_PA', EER_FARs[5], epoch)
                writer.add_scalar('FRR/C_AP_VI_PA', EER_FRRs[5], epoch)

                wandb.log({'epoch': epoch, 'EER/C_AP_VI_PA': EERs[5], 'FAR/C_AP_VI_PA': EER_FARs[5], 'FRR/C_AP_VI_PA': EER_FRRs[5]})

                writer.add_scalar('Loss/test', epoch_loss, epoch)

                wandb.log({'epoch': epoch, 'Loss/test': epoch_loss})

    # test model

    return models


if __name__ == "__main__":
    # mode = sys.argv[1]
    mode = '3'

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(device)
    # device = "cpu"

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
    
    ids = list(range(22))
    valid_ids = drawer.draw_numbers(22, 4)
    # valid_ids = [6, 9]
    train_ids = [item for item in ids if item not in valid_ids]
    print(train_ids)
    print(valid_ids)

    model_piezo = STFTAttentionModel(256, 128, 256).to(device)
    model_audio = STFTAttentionModel(256, 128, 256).to(device)
    # model.load_state_dict(torch.load('model_auth_siamese.pth'))
    # model.load_state_dict(torch.load('model_p_to_a.pth'))
    print(model_piezo)
    train_set = STFTFeatureSet(False, train_ids, 30)
    test_set = STFTFeatureSet(False, valid_ids, 30)
    ge2e_loss_piezo = GE2ELoss(device)
    ge2e_loss_audio = GE2ELoss(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer_piezo = torch.optim.SGD([
        {'params': model_piezo.parameters()},
        {'params': ge2e_loss_piezo.parameters()}
    ], lr=0.001)
    optimizer_audio = torch.optim.SGD([
        {'params': model_audio.parameters()},
        {'params': ge2e_loss_audio.parameters()}
    ], lr=0.001)
    models = {'piezo': model_piezo,
              'audio': model_audio}
    optimizers = {'piezo': optimizer_piezo,
                  'audio': optimizer_audio}
    train_test_model(device, models, train_set, test_set, criterion, optimizers)
    torch.save(models['piezo'].state_dict(), 'model_auth_siamese_attention_lstm_piezo.pth')
    torch.save(models['audio'].state_dict(), 'model_auth_siamese_attention_lstm_audio.pth')
