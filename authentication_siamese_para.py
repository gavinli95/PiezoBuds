import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import LatentFeatureSet, FrequencyFeatureSet, STFTFeatureSet
import tensorboard
from conv1dClassify import Conv1DNet, Classifier, Conv1DAutoencoder, STFTFeatureExtractor2D
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from UniqueDraw import UniqueDraw
import torch.nn.functional as F
from conv1dClassify import GE2ELoss
import random

from utils import get_centroids, get_cossim


def calculate_l2_distance(tensor1, tensor2, dim=None):
    differences = tensor1 - tensor2
    squared_differences = torch.square(differences)
    summed_squared_differences = torch.sum(squared_differences, dim=dim)
    l2_distance = torch.sqrt(summed_squared_differences)
    return l2_distance


def train_test_model(device, model_piezo, model_audio, train_set, test_set, criterion, optimizer_piezeo, optimizer_audio, train_ratio=0.7, num_epochs=80000):
    writer = SummaryWriter()

    batch_size = 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model_piezo.train()  # Set model to training mode
                model_audio.train()
                dataloader = train_loader
            else:
                model_piezo.eval()   # Set model to evaluate mode
                dataloader = test_loader

            avg_EER = 0
            running_loss = 0.0
            running_loss_ge2e = 0.0
            running_loss_out_audio = 0.0
            # Iterate over data.
            batch_avg_EER = 0
            for batch_id, utterance in enumerate(dataloader):
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    num_of_speakers, num_of_utters, w, l = utterance.shape

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        utterance = torch.reshape(utterance, (num_of_speakers * num_of_utters, w, l)).to(
                            device)
                        piezos = utterance[:, :w//2].to(device)
                        audios = utterance[:, w//2:].to(device)

                        embeddings = model_piezo(piezos)
                        embeddings_audio = model_audio(audios)
                        _, num_of_fea = embeddings.shape
                        embeddings = torch.reshape(embeddings, (num_of_speakers, num_of_utters, num_of_fea))
                        embeddings_audio = torch.reshape(embeddings_audio, (num_of_speakers, num_of_utters, num_of_fea))

                        loss_ge2e = ge2e_loss(embeddings)
                        loss_audio = calculate_l2_distance(embeddings, embeddings_audio).mean()

                        loss = loss_ge2e + loss_audio
                        # loss = loss_ge2e
                        # loss = loss_audio

                        optimizer_piezeo.zero_grad()
                        optimizer_audio.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_piezo.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(model_audio.parameters(), 3.0)
                        torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
                        optimizer_piezeo.step()
                        optimizer_audio.step()

                    if phase == 'test':
                        enrollment_batch, verification_batch = torch.split(utterance, int(num_of_utters/2), dim=1)
                        enrollment_batch = torch.reshape(enrollment_batch, (num_of_speakers * num_of_utters // 2, w, l))
                        verification_batch = torch.reshape(verification_batch, (num_of_speakers * num_of_utters // 2, w, l))

                        perm = random.sample(range(0, verification_batch.size(0)), verification_batch.size(0))
                        unperm = list(perm)
                        for i, j in enumerate(perm):
                            unperm[j] = i

                        verification_batch = verification_batch[perm]

                        enrollment_embeddings = model_piezo(enrollment_batch[:, :w//2].to(device))
                        verification_embeddings = model_piezo(verification_batch[:, :w//2].to(device))
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
                running_loss += loss.item()
                running_loss_ge2e += loss_ge2e.item()
                running_loss_out_audio += loss_audio.item()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_loss_ge2e = running_loss_ge2e / len(dataloader.dataset)
            epoch_loss_out_audio = running_loss_out_audio / len(dataloader.dataset)
            # epoch_acc = running_corrects.float() / len(dataloader.dataset)

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Loss: {epoch_loss:.4f}')
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Loss/train_ge2e', epoch_loss_ge2e, epoch)
                writer.add_scalar('Loss/train_out_audio', epoch_loss_out_audio, epoch)
            else:
                writer.add_scalar('EER/test', EER, epoch)
                writer.add_scalar('FAR/test', EER_FAR, epoch)
                writer.add_scalar('FRR/test', EER_FRR, epoch)
                writer.add_scalar('Loss/test', epoch_loss, epoch)

    # test model

    return model_piezo


if __name__ == "__main__":
    # mode = sys.argv[1]
    mode = '3'

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"

    drawer = UniqueDraw()
    ids = list(range(10))
    valid_ids = drawer.draw_numbers(10, 2)
    train_ids = [item for item in ids if item not in valid_ids]

    model_piezo = STFTFeatureExtractor2D(input_shape=(84, 100)).to(device)
    model_audio = STFTFeatureExtractor2D(input_shape=(84, 100)).to(device)
    # model.load_state_dict(torch.load('model_p_to_a.pth'))
    print(model_piezo)
    train_set = STFTFeatureSet(True, train_ids, 40)
    test_set = STFTFeatureSet(True, valid_ids, 20)
    ge2e_loss = GE2ELoss(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer_piezo = torch.optim.SGD([
        {'params': model_piezo.parameters()},
        {'params': ge2e_loss.parameters()}
    ], lr=0.001)
    optimizer_audio = torch.optim.SGD([
        {'params': model_audio.parameters()},
    ], lr=0.001)
    train_test_model(device, model_piezo, model_audio, train_set, test_set, criterion, optimizer_piezo, optimizer_audio)
    torch.save(model_piezo.state_dict(), 'model_auth.pth')
