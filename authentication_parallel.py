import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import LatentFeatureSet, FrequencyFeatureSet
import tensorboard
from conv1dClassify import Conv1DNet, Classifier
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from UniqueDraw import UniqueDraw
import torch.nn.functional as F
from conv1dClassify import GE2ELoss
import random

from utils import get_centroids, get_cossim


def get_embeddings(utterance, models):
    num_of_speakers, num_of_utters, num_of_features = utterance.shape
    utterance = torch.reshape(utterance, (num_of_speakers * num_of_utters, num_of_features)).to(
        device)
    piezo = utterance[:, :num_of_features // 2]
    audio = utterance[:, num_of_features // 2:]
    embeddings_piezo = models['model_piezo'](piezo)
    # embeddings_audio = models['model_audio'](audio)
    embeddings_audio = models['model_piezo'](audio)
    _, num_of_fea = embeddings_piezo.shape
    embeddings = torch.cat((embeddings_piezo, embeddings_audio), dim=1)
    embeddings = models['model_combined'](embeddings)
    embeddings = torch.reshape(embeddings, (num_of_speakers, num_of_utters, num_of_fea))
    embeddings_piezo = torch.reshape(embeddings_piezo, (num_of_speakers, num_of_utters, num_of_fea))
    embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters, num_of_fea))
    return embeddings, embeddings_piezo, embeddings_audio


def train_test_model(device, models, train_set, test_set, optimizers, num_epochs=20000):
    writer = SummaryWriter()

    batch_size = 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                models['model_piezo'].train()
                models['model_audio'].train()
                models['model_combined'].train()
                dataloader = train_loader
            else:
                models['model_piezo'].eval()
                models['model_audio'].eval()
                models['model_combined'].eval()
                dataloader = test_loader

            avg_EER = 0
            running_loss = 0.0
            # Iterate over data.
            batch_avg_EER = 0
            for batch_id, (utterance, noises) in enumerate(dataloader):
                # forward
                utterance = torch.concatenate((utterance, noises), dim=0)
                num_of_speakers, num_of_utters, num_of_features = utterance.shape
                with torch.set_grad_enabled(phase == 'train'):
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        embeddings, embeddings_piezo, embeddings_audio = get_embeddings(utterance, models)

                        loss_piezo = ge2e_loss_piezo(embeddings_piezo)
                        loss_audio = ge2e_loss_audio(embeddings_audio)
                        loss_combined = ge2e_loss_combined(embeddings)

                        optimizers['model_piezo'].zero_grad()
                        optimizers['loss_piezo'].zero_grad()
                        # optimizers['model_audio'].zero_grad()
                        # optimizers['loss_audio'].zero_grad()
                        optimizers['model_combined'].zero_grad()
                        optimizers['loss_combined'].zero_grad()
                        loss_combined.backward()
                        torch.nn.utils.clip_grad_norm_(models['model_piezo'].parameters(), 1.0)
                        # torch.nn.utils.clip_grad_norm_(models['model_audio'].parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(models['model_combined'].parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_piezo.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_audio.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss_combined.parameters(), 1.0)
                        optimizers['model_piezo'].step()
                        optimizers['loss_piezo'].step()
                        # optimizers['model_audio'].step()
                        # optimizers['loss_audio'].step()
                        optimizers['model_combined'].step()
                        optimizers['loss_combined'].step()

                    if phase == 'test':

                        enrollment_batch, verification_batch = torch.split(utterance, int(num_of_utters/2), dim=1)

                        enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, num_of_features))
                        verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, num_of_features))

                        perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
                        unperm = list(perm)
                        for i, j in enumerate(perm):
                            unperm[j] = i

                        verification_batch = verification_batch[perm]
                        enrollment_batch = torch.reshape(enrollment_batch,
                                                         (num_of_speakers, num_of_utters // 2, num_of_features))
                        verification_batch = torch.reshape(verification_batch,
                                                           (num_of_speakers, num_of_utters // 2, num_of_features))
                        enroll_embeddings, enroll_embeddings_piezo, enroll_embeddings_audio = get_embeddings(enrollment_batch, models)

                        loss_combined = ge2e_loss_combined(enroll_embeddings)

                        verify_embeddings, verify_embeddings_piezo, verify_embeddings_audio = get_embeddings(verification_batch, models)
                        _, _, num_of_fea = enroll_embeddings.shape

                        enrollment_embeddings = torch.reshape(enroll_embeddings, (num_of_speakers * num_of_utters // 2, num_of_fea))
                        verification_embeddings = torch.reshape(verify_embeddings, (num_of_speakers * num_of_utters // 2, num_of_fea))

                        verification_embeddings = verification_embeddings[unperm]

                        enrollment_embeddings = torch.reshape(enrollment_embeddings, (num_of_speakers, num_of_utters // 2, num_of_fea))
                        verification_embeddings = torch.reshape(verification_embeddings, (
                        num_of_speakers, num_of_utters // 2, num_of_fea))

                        enrollment_centroids = get_centroids(enrollment_embeddings)

                        sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)

                        # calculating EER
                        diff = 1; EER = 0; EER_thresh = 0; EER_FAR = 0; EER_FRR = 0

                        for thres in [0.01 * i + 0.5 for i in range(50)]:
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
                        batch_avg_EER += EER
                        print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thresh, EER_FAR, EER_FRR))
                        # avg_EER += batch_avg_EER / (batch_id + 1)
                        # avg_EER = avg_EER / hp.test.epochs
                        # print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
                # statistics
                running_loss += loss_combined.item()

            epoch_loss = running_loss / len(dataloader.dataset)
            # epoch_acc = running_corrects.float() / len(dataloader.dataset)

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Loss: {epoch_loss:.4f}')
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
            else:
                writer.add_scalar('EER/test', EER, epoch)
                writer.add_scalar('FAR/test', EER_FAR, epoch)
                writer.add_scalar('FRR/test', EER_FRR, epoch)
                writer.add_scalar('Loss/test', epoch_loss, epoch)

    # test model

    return models


if __name__ == "__main__":
    # mode = sys.argv[1]
    mode = '3'

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(device)
    # device = "cpu"

    drawer = UniqueDraw()
    ids = list(range(10))
    valid_ids = drawer.draw_numbers(10, 2)
    train_ids = [item for item in ids if item not in valid_ids]

    model_piezo = Conv1DNet(83).to(device)
    model_audio = Conv1DNet(83).to(device)
    model_combined = Conv1DNet(256 * 2).to(device)

    models = {
        'model_piezo': model_piezo,
        'model_audio': model_audio,
        'model_combined': model_combined
    }

    # print(model)
    train_set = FrequencyFeatureSet(True, train_ids, 40)
    test_set = FrequencyFeatureSet(True, valid_ids, 10)
    ge2e_loss_piezo = GE2ELoss(device)
    ge2e_loss_audio = GE2ELoss(device)
    ge2e_loss_combined = GE2ELoss(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizers = {
        'model_piezo': torch.optim.SGD(model_piezo.parameters(), lr=0.0001),
        'model_audio': torch.optim.SGD(model_audio.parameters(), lr=0.0001),
        'model_combined': torch.optim.SGD(model_combined.parameters(), lr=0.0001),
        'loss_piezo': torch.optim.SGD(ge2e_loss_piezo.parameters(), lr=0.00001),
        'loss_audio': torch.optim.SGD(ge2e_loss_audio.parameters(), lr=0.00001),
        'loss_combined': torch.optim.SGD(ge2e_loss_combined.parameters(), lr=0.00001)
    }
    # optimizer = torch.optim.SGD([
    #     {'params': model1.parameters()},
    #     {'params': model2.parameters()},
    #     {'params': ge2e_loss.parameters()}
    # ], lr=0.0001)
    train_test_model(device, models, train_set, test_set, optimizers)
