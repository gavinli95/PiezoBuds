import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils import get_centroids, get_cossim, calc_loss, get_modal_cossim, cal_contrast_loss, cal_intra_loss
import torchvision

class CNNModel(nn.Module):
    def __init__(self, output_length):
        super(CNNModel, self).__init__()
        
        # First convolutional layer
        # Assuming input is (batch_size, 1, 100, 256)
        # After this layer, output becomes (batch_size, 4, 50, 128)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(2,2), stride=(2,2), padding=(0,0))
        
        # Second convolutional layer
        # After this layer, output becomes (batch_size, 16, 16, 32)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=(3,4), stride=(3,4), padding=(0,0))
        
        # Flattening and Fully Connected layers
        # Flattened size: 16 * 16 * 32 = 8192
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, output_length)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Flatten the output for the FC layers
        x = x.view(x.size(0), -1)
        
        x = nn.ReLU()(self.fc1(x))  # Activation after the first FC layer
        x = self.fc2(x)
        return x, 0


class Attention(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super(Attention, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.attention_layer = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (seq_len, batch, lstm_hidden_dim)
        attention_weights = self.attention_layer(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=0) # Softmax across time dimension
        
        # context_vector shape after sum: (batch, lstm_hidden_dim)
        context_vector = torch.sum(attention_weights * lstm_output, dim=0)
        return context_vector, attention_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMWithAttention, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_outputs, (hn, cn) = self.lstm(x)
        
        # Transposing lstm_outputs for attention mechanism
        # New shape: (seq_len, batch, lstm_hidden_dim)
        lstm_outputs = lstm_outputs.transpose(0, 1)
        
        context_vector, attention_weights = self.attention(lstm_outputs)
        output = self.fc(context_vector)
        l2_norm = torch.norm(output, p=2, dim=1, keepdim=True)
        output_normalized = output / l2_norm

        return output_normalized, attention_weights



class ConverterNetwork1DCNN(nn.Module):
    def __init__(self, embedding_length, num_channels):
        super(ConverterNetwork1DCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.relu1 = nn.LeakyReLU(0.01)

        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.relu2 = nn.LeakyReLU(0.01)

        self.conv3 = nn.Conv1d(in_channels=num_channels, out_channels=1, kernel_size=3, padding=1)
        
    @staticmethod
    def weights_init_xavier(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def weights_init_he(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x should be of shape (batch_size, 1, embedding_length) for 1D convolution
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        
        return x.squeeze(1) # To bring it back to the shape (batch_size, embedding_length)


class ConverterCNNNetwork(nn.Module):
    def __init__(self, embedding_length, hidden_size):
        super(ConverterCNNNetwork, self).__init__()

        self.dropout = nn.Dropout(0.3)
        
        # add-on Conv layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, stride=4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=128)
        )


    @staticmethod
    def weights_init_xavier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    @staticmethod
    def weights_init_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        
        batch_size, f_len = x.shape
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = x.contiguous()
        x = x.reshape([batch_size, -1])

        x = self.fc(x)
        return x


class ConverterNetwork(nn.Module):
    def __init__(self, embedding_length, hidden_size):
        super(ConverterNetwork, self).__init__()

        self.dropout = nn.Dropout(0.5)

        # First fully connected layer
        self.fc1 = nn.Linear(embedding_length, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch Normalization after first layer
        self.relu1 = nn.LeakyReLU(0.05)  # Switched to LeakyReLU for potentially better gradient flow

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)  # Batch Normalization after second layer
        self.relu2 = nn.LeakyReLU(0.05)  # Switched to LeakyReLU for potentially better gradient flow

        # Third fully connected layer
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size )
        self.bn3 = nn.BatchNorm1d(hidden_size)  # Batch Normalization after second layer
        self.relu3 = nn.LeakyReLU(0.05)  # Switched to LeakyReLU for potentially better gradient flow

        self.fc4 = nn.Linear(hidden_size, embedding_length)

        # self.fusionLayer = nn.Sequential(
        #     nn.Linear(embedding_length * 3, embedding_length * 2),
        #     nn.LeakyReLU(0.05),
        #     nn.Linear(embedding_length * 2, embedding_length),
        #     nn.LeakyReLU(0.05),
        #     nn.Linear(embedding_length, embedding_length)
        # )
        self.fusionLayer = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.05),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.05),
            nn.Linear(embedding_length, embedding_length)
        )


    @staticmethod
    def weights_init_xavier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    @staticmethod
    def weights_init_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ea, ep):
        x = self.fc1(ea)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        x = self.fc4(x)
        y = torch.cat((ea.unsqueeze(1), x.unsqueeze(1), ep.unsqueeze(1)), dim=-2)
        y = self.fusionLayer(y)
        y = y.squeeze()
        return x, y
        # return x


class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.ws = [nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True),
                   nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True),
                   nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True),
                   nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)]
        
        self.bs = [nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True),
                   nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True),
                   nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True),
                   nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)]
        
        self.alphas = [nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True),
                       nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True),
                       nn.Parameter(torch.tensor(0.25).to(device), requires_grad=True)]
        
        self.device = device

    def forward(self, embeddings_audio, embeddings_piezo):
        torch.clamp(self.ws[0], 1e-6)
        torch.clamp(self.ws[1], 1e-6)
        torch.clamp(self.ws[2], 1e-6)
        torch.clamp(self.ws[3], 1e-6)
        torch.clamp(self.alphas[0], 1e-6)
        torch.clamp(self.alphas[1], 1e-6)
        torch.clamp(self.alphas[2], 1e-6)

        centroids_piezo = get_centroids(embeddings_piezo)
        centroids_audio = get_centroids(embeddings_audio)

        cossim_piezo_to_piezo = get_cossim(embeddings_piezo, centroids_piezo)
        cossim_audio_to_audio = get_cossim(embeddings_audio, centroids_audio)
        # cossim_piezo_to_auido = get_cossim(embeddings_piezo, centroids_audio)
        # cossim_audio_to_piezo = get_cossim(embeddings_audio, centroids_piezo)
        cossim_piezo_to_auido = get_modal_cossim(embeddings_piezo, centroids_audio)
        cossim_audio_to_piezo = get_modal_cossim(embeddings_audio, centroids_piezo)

        sim_matrix = self.ws[0] * cossim_piezo_to_piezo.to(self.device) + self.bs[0]
        loss_pp, _, loss_pp_include, loss_pp_exclude = calc_loss(sim_matrix)
        # contrast loss
        loss_pp_contrast, _ = cal_contrast_loss(sim_matrix, self.device)

        sim_matrix = self.ws[1] * cossim_audio_to_audio.to(self.device) + self.bs[1]
        loss_aa, _, loss_aa_include, loss_aa_exclude = calc_loss(sim_matrix)
        # contrast loss
        loss_aa_contrast, _ = cal_contrast_loss(sim_matrix, self.device)

        sim_matrix = self.ws[2] * cossim_piezo_to_auido.to(self.device) + self.bs[2]
        # _, _, loss_pa = calc_loss(sim_matrix)
        _, _, loss_pa_include, _ = calc_loss(sim_matrix)
        # intra loss
        loss_pa_intra, _ = cal_intra_loss(sim_matrix, self.device)
        
        sim_matrix = self.ws[3] * cossim_audio_to_piezo.to(self.device) + self.bs[3]
        # _, _, loss_ap = calc_loss(sim_matrix)
        _, _, loss_ap_include, _ = calc_loss(sim_matrix)
        # intra loss
        loss_ap_intra, _ = cal_intra_loss(sim_matrix, self.device)

        # loss =  torch.sqrt(torch.abs(loss_aa * loss_pp)) + 2 * (loss_pa_include + loss_ap_include)
        # loss = 2.0 * (loss_pp_exclude + loss_aa_exclude) + 1.0 * (loss_ap_include + loss_pa_include + loss_aa_include + loss_pp_include)
        # loss =  2 * (loss_aa + loss_pp) + 1 * (loss_pa_include + loss_ap_include)
        # loss = 2 * (loss_aa_contrast + loss_pp_contrast) + 1 * (loss_pa_include + loss_ap_include)
        loss = loss_aa + loss_pp + loss_pa_intra + loss_ap_intra

        return loss_aa, loss_pp
    



class extractor1DCNN(nn.Module):

    def __init__(self, channel_number, in_feature, out_feature, device):
        super(extractor1DCNN, self).__init__()

        self.channel_number = channel_number
        self.device = device
        self.f_in = in_feature
        self.f_out = out_feature

        self.CNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.channel_number, out_channels=64, padding=1, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, padding=1, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, padding=1, kernel_size=3),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(),
        )

        self.FC = nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=in_feature),
            nn.LeakyReLU(),
            nn.Linear(in_features=in_feature, out_features=out_feature),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # _, f_bin, t_bin = x.shape
        # x = x.contiguous()
        # x = x.view(-1, t_bin, f_bin)
        x = self.CNN1D(x)
        x = torch.flatten(x, start_dim=1)
        x = self.FC(x)
        x_s = x.squeeze()
        return x_s
    

class Converter_w_triplet(nn.Module):
    def __init__(self, embedding_length, hidden_len, output_len):
        super(Converter_w_triplet, self).__init__()

        self.fc_layers_piezo = nn.Sequential(
            nn.Linear(embedding_length, hidden_len),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_len, output_len),
            nn.LeakyReLU()
        )
        self.fc_layers_audio_low = nn.Sequential(
            nn.Linear(embedding_length, hidden_len),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_len, output_len),
            nn.LeakyReLU()
        )


    @staticmethod
    def weights_init_xavier(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    @staticmethod
    def weights_init_he(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, p, a_l):
        x = self.fc_layers_piezo(p)
        y = self.fc_layers_audio_low(a_l)
        return x, y



class Converter_ALAH2X(nn.Module):
    def __init__(self, embedding_length, hidden_len, output_len):
        super(Converter_ALAH2X, self).__init__()

        self.al_net = ConverterNetwork(embedding_length=embedding_length * 2, hidden_size=hidden_len)
        # self.ah_net = ConverterNetwork(embedding_length=embedding_length * 2, hidden_size=hidden_len)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_length * 2, output_len),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(output_len, output_len)
        )

    def forward(self, a_l, a_h):
        hl = torch.cat((a_l, a_h), dim=-1)
        l = self.al_net(hl)
        # h = self.ah_net(a_h)
        x = self.fusion(hl)
        return x

class Extractor_w_TF(nn.Module):
    def __init__(self, device, f_len, t_len):
        super(Extractor_w_TF, self).__init__()
        self.device = device
        self.f_len = f_len
        self.t_len = t_len


        self.T_filter = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 1), padding=(1, 0), stride=(2, 1)), # (1, t_len, f_len) -> (32, t_len / 2, f_len)
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.F_filter = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 4), padding=(0, 1), stride=(1, 2)), # (32, t_len / 2, f_len) -> (64, t_len / 2, f_len / 2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 4), padding=(0, 1), stride=(1, 2)), # (64, t_len / 2, f_len / 2) -> (64, t_len / 2, f_len / 4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.TF_filter = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2)), # (64, t_len / 2, f_len / 4) -> (32, t_len / 4, f_len / 8)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=(2, 2)), # (32, t_len / 4, f_len / 8) -> (16, t_len / 8, f_len / 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.FC_layers = nn.Sequential(
            nn.Linear(int (self.f_len * self.t_len / 8), self.f_len * 4),
            nn.ReLU(),
            nn.Linear(self.f_len * 4, self.f_len),
            nn.ReLU(),
            nn.Linear(self.f_len, 256)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.T_filter(x)
        x = self.F_filter(x)
        x = self.TF_filter(x)
        x = x.flatten(1)
        x = self.FC_layers(x)

        return x
    
class Extractor_w_F_1D(nn.Module):
    def __init__(self, in_channel_n, device, f_len):
        super(Extractor_w_F_1D, self).__init__()
        self.device = device
        self.in_channel_n = in_channel_n

        self.F_filter = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel_n, out_channels=64, kernel_size=1), # (in_channel_n, f_len) -> (64, f_len)
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2), # (64, f_len) -> (32, f_len / 2)
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4, padding=1, stride=2), # (32, f_len / 2) -> (16, f_len / 4)
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(f_len * 4, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            # nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.F_filter(x)
        x = x.flatten(1)
        x = self.fc_layer(x)

        return x
    

class Extractor_w_F_2D(nn.Module):
    def __init__(self, device, f_len):
        super(Extractor_w_F_2D, self).__init__()
        self.device = device
        self.f_len = f_len

        self.F_filter = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 4), padding=(0, 1), stride=(1, 2)), # (1, 50, f_len) -> (32, 50, f_len / 2)
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 4), padding=(0, 1), stride=(1, 2)), # (32, 50, f_len / 2) -> (64, 50, f_len / 4)
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 4), padding=(0, 1), stride=(1, 2)), # (64, 50, f_len / 4) -> (32, 50, f_len / 8)
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=(1, 4), padding=(0, 1), stride=(1, 2)), # (32, 50, f_len / 8) -> (4, 50, f_len / 16)
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
        )

        self.FC_layers = nn.Sequential(
            nn.Linear(int(12.5 * self.f_len), 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.F_filter(x)
        x = x.flatten(1)
        x = self.FC_layers(x)

        return x

class Extractor_w_LSTM(nn.Module):
    def __init__(self, device, layer_n, input_dim, output_dim):
        super(Extractor_w_LSTM, self).__init__()
        self.device = device
        self.layer_n = layer_n
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.LSTM_layer = nn.LSTM(input_dim, output_dim, layer_n, batch_first=True)

        self.fc_layer = nn.Sequential(
            nn.Linear(output_dim * 4, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )

    def init_hidden(self, batch_size):
        hidden_state = Variable(torch.randn(self.layer_n, batch_size, self.output_dim).to(device=self.device))
        cell_state = Variable(torch.randn(self.layer_n, batch_size, self.output_dim).to(device=self.device))
        return (hidden_state, cell_state)

    def forward(self, x):
        batch_size, t_len, _ = x.size()
        self.hidden = self.init_hidden(batch_size=batch_size)
        x_lens = []

        for _ in range(batch_size):
            x_lens.append(t_len)
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        x, self.hidden = self.LSTM_layer(x, self.hidden)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        x = x.contiguous()
        x = x[:, -4:, :]
        x = x.flatten(1)
        x = self.fc_layer(x)

        return x


class Extractor_pure_FC(nn.Module):
    def __init__(self, device):
        super(Extractor_pure_FC, self).__init__()
        self.device = device

        self.fc_layers = nn.Sequential(
            nn.Linear(50*256, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256)
        )

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc_layers(x)

        return x
    
class GE2ELoss_ori(nn.Module):

    def __init__(self, device):
        super(GE2ELoss_ori, self).__init__()
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
    

class class_model_FC(nn.Module):

    def __init__(self, device, input_len, class_n):
        super(class_model_FC, self).__init__()
        self.device = device
        self.input_len = input_len
        self.class_n = class_n

        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_len, 1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1024, class_n, bias=True)
        )
    def forward(self, x):
        x = self.fc_layers(x)
        return x
    
class class_model_weighted(nn.Module):

    def __init__(self, device):
        super(class_model_weighted, self).__init__()
        self.device = device
        self.a = nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True)
        self.p = nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, a, p):
        x = self.a * a + self.p * p
        x = self.softmax(x)

        return x
    
class SpeechEmbedder(nn.Module):

    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(256, 512, num_layers=3, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(512, 128)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x


class CustomMobileNetV3(nn.Module):
    def __init__(self):
        super(CustomMobileNetV3, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)
        # Replace the initial conv layer
        self.mobilenet.features[0][0] = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # Replace the final classifier layer
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, 128)
    
    def forward(self, x):
        # Features till second last block
        x = self.mobilenet.features[:-2](x)
        output1 = x.clone()
        
        # Second last block
        x = self.mobilenet.features[-2](x)
        output2 = x.clone()
        
        # Last block and classifier
        x = self.mobilenet.features[-1](x)
        x = x.mean([2, 3])
        output3 = self.mobilenet.classifier(x)
        
        return output1, output2, output3



