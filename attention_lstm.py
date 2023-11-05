import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, lstm_output):
        # Calculate attention scores
        scores = torch.matmul(lstm_output, self.attention_weights)
        attention_probs = F.softmax(scores, dim=1).unsqueeze(2)

        # Calculate context vector
        context_vector = torch.sum(attention_probs * lstm_output, dim=1)
        return context_vector


class STFTAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STFTAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, frames, frequency_bins)
        x = x.squeeze()
        x = x.permute(0, 2, 1)

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Apply attention
        context_vector = self.attention(lstm_out)

        # Output layer
        out = self.fc(context_vector)
        return out
