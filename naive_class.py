import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader_from_numpy import LatentFeatureSet
import tensorboard
from conv1dClassify import Conv1DNet, Classifier
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter


def train_test_model(device, model, dataset, criterion, optimizer, train_ratio=0.7, num_epochs=2000):
    writer = SummaryWriter('./runs/')

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/test', epoch_loss, epoch)
                writer.add_scalar('Accuracy/test', epoch_acc, epoch)

    return model


if __name__ == "__main__":
    # mode = sys.argv[1]
    mode = '2'

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    root_piezo = './latent_feature/{}_piezo.npy'
    root_audio = './latent_feature/{}_audio.npy'

    labels = []
    piezos = []
    audios = []
    piezos_audios = []
    for i in range(10):
        piezo_features = np.load(root_piezo.format(i))
        audio_features = np.load(root_audio.format(i))
        piezo_audio_features = np.concatenate((piezo_features, audio_features), axis=1)

        piezo_list = piezo_features.tolist()
        audio_list = audio_features.tolist()
        piezo_audio_list = piezo_audio_features.tolist()

        piezos = piezos + piezo_list
        audios = audios + audio_list
        piezos_audios = piezos_audios + piezo_audio_list

        for j in range(len(piezo_list)):
            labels.append(i)
    labels = np.array(labels)
    piezos = np.array(piezos)
    audios = np.array(audios)
    piezos_audios = np.array(piezos_audios)

    if mode == '0':
        model = Conv1DNet(128).to(device)
        dataset = LatentFeatureSet(piezos, labels)
    if mode == '1':
        model = Conv1DNet(128).to(device)
        dataset = LatentFeatureSet(audios, labels)
    if mode == '2':
        model = Conv1DNet(256).to(device)
        dataset = LatentFeatureSet(piezos_audios, labels)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    train_test_model(device, model, dataset, criterion, optimizer)
