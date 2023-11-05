from utils import get_centroids, get_cossim, calc_loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractionCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FeatureExtractionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(input_size // 2 * 64, 512)  # Adjust the input size based on the pooling
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # reshape to (batch_size, num_channels, length)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class Conv1DNet(nn.Module):
    def __init__(self, input_size, num_classes=512, dropout_prob=0.5):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, stride=4, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(input_size // 8 * 32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # reshape to (batch_size, num_channels, length)
        x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.normalize(x, p=2, dim=1)
        return x


class Conv1DDecoder(nn.Module):
    def __init__(self, input_size, num_classes=512, dropout_prob=0.5):
        super(Conv1DDecoder, self).__init__()
        self.fc1 = nn.Linear(num_classes, 1024)
        self.fc2 = nn.Linear(1024, input_size // 8 * 32)
        self.deconv1 = nn.ConvTranspose1d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=4, padding=0, output_padding=0)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 32, -1)  # reshape to (batch_size, num_channels, length)
        x = F.relu(self.deconv1(x))
        x = self.dropout(x)
        x = F.relu(self.deconv2(x))
        x = x.view(x.size(0), -1)  # flatten
        return x


class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_size, num_feature=256, dropout_prob=0.5):
        super(Conv1DAutoencoder, self).__init__()
        self.encoder = Conv1DNet(input_size, num_feature, dropout_prob)
        self.decoder = Conv1DDecoder(input_size, num_feature, dropout_prob)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Conv1DExtractor(nn.Module):
    def __init__(self, input_size, dim_features=49):
        super(Conv1DExtractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_size * 128, 256)  # input_size * output channels of last Conv1d layer
        self.fc2 = nn.Linear(256, dim_features)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # reshape to (batch_size, num_channels, length)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(Classifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim.to(self.device) + self.b
        loss, _, _, _ = calc_loss(sim_matrix)
        return loss


class STFTFeatureExtractor2D(nn.Module):
    def __init__(self, input_shape, num_classes=256, lstm_hidden_size=1024):
        super(STFTFeatureExtractor2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Add LSTM layer
        self.lstm = nn.LSTM(input_size=800,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)  # Set batch_first to True

        self.fc1 = nn.Linear(lstm_hidden_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Add a single channel dimension to the input (batch_size, 1, w, l)
        x = x.unsqueeze(1)

        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Transpose the input to (batch_size, l, w*32)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        # Apply LSTM layer along the time dimension
        x, _ = self.lstm(x)

        # Apply fully connected layers
        x = x[:, -1, :]  # Use only the last output of the LSTM as the feature representation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Add a single channel dimension to the input (batch_size, 1, w, l)
        x = x.unsqueeze(1)

        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Transpose the input to (batch_size, l, w*32)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        return x


class LSTMModule(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(input_size=800,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)  # Set batch_first to True

    def forward(self, x):
        # Apply LSTM layer along the time dimension
        x, _ = self.lstm(x)
        return x


class FCModule(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCModule, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply fully connected layers
        x = x[:, -1, :]  # Use only the last output of the LSTM as the feature representation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SpeechEmbedder(nn.Module):

    def __init__(self, n_stft=84, n_hidden=256, num_layers=3, n_features=128):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(n_stft, n_hidden, num_layers=num_layers, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(n_hidden, n_features)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x



