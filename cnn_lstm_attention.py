import torch
import torch.nn as nn


class ParallelModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
            nn.Dropout(p=0.3),
            # 2. conv block
            # nn.Conv2d(in_channels=16,
            #           out_channels=32,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1
            #           ),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            # nn.Dropout(p=0.3),
        )
        # GLU
        self.glu = nn.GLU(dim=2)
        # LSTM block
        hidden_size = 256
        self.lstm = nn.LSTM(input_size=512, num_layers=1, hidden_size=hidden_size, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.1)
        self.attention_linear = nn.Linear(100, 1)  # 2*hidden_size for the 2 outputs of bidir LSTM

        # Linear softmax layer
        self.out_linear = nn.Sequential(
            nn.Linear(512 + hidden_size, 512),
            nn.Linear(512, num_emotions)
        )

        self.dropout_linear = nn.Dropout(p=0.0)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # conv embedding
        conv_embedding = self.conv2Dblock(x)  # (b,channel,freq,time)
        # conv_embedding = torch.transpose(conv_embedding, 1, 3)
        # conv_embedding = torch.flatten(conv_embedding, start_dim=2)  # do not flatten batch dimension
        conv_embedding = conv_embedding.permute(0, 3, 1, 2)
        # GLU
        glu_embedding = self.glu(conv_embedding)
        glu_embedding = torch.flatten(glu_embedding, start_dim=2)

        # lstm embedding
        # x_reduced = torch.squeeze(conv_embedding, 1)
        # x_reduced = x_reduced.permute(0, 2, 1)  # (b,t,freq)
        lstm_embedding, (h, c) = self.lstm(glu_embedding)  # (b, time, hidden_size*2)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        batch_size, T, Fr = lstm_embedding.shape
        attention_weights = [None] * Fr
        lstm_fre_embedding = lstm_embedding.permute(0, 2, 1)
        for f in range(Fr):
            embedding = lstm_fre_embedding[:, f, :]
            attention_weights[f] = self.attention_linear(embedding)
        attention_weights_norm = nn.functional.softmax(torch.stack(attention_weights, -1), -1)
        attention = torch.bmm(attention_weights_norm, lstm_fre_embedding)  # (Bx1xT)*(B,T,hidden_size*2)=(B,1,2*hidden_size)
        # attention = torch.unsqueeze(attention, 1)

        attention = torch.bmm(attention, glu_embedding).squeeze(dim=1)

        # concatenate
        complete_embedding = torch.cat([lstm_embedding[:, -1, :], attention], dim=1)

        output_logits = self.out_linear(complete_embedding)
        output_logits = self.dropout_linear(output_logits)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax, attention
