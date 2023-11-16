import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyResponse(nn.Module):
    def __init__(self, input_size, output_size):
        super(FrequencyResponse, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(input_size, 1024)  # First fully connected layer
        self.bn1 = nn.BatchNorm1d(1024)         # Batch normalization
        self.fc2 = nn.Linear(1024, 512)          # Second fully connected layer
        self.bn2 = nn.BatchNorm1d(512)          # Batch normalization
        self.fc3 = nn.Linear(512, output_size)  # Output layer

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # Apply ReLU activation after batch normalization
        x = F.relu(self.bn2(self.fc2(x)))  # Apply ReLU activation after batch normalization
        x = self.fc3(x)  # No activation for the final layer
        return x


class EmbeddingFusionModel(nn.Module):
    def __init__(self, embedding_dim_1, embedding_dim_2, output_dim):
        super(EmbeddingFusionModel, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(embedding_dim_1 + embedding_dim_2, 128)  # Fusion layer
        self.bn1 = nn.BatchNorm1d(128)         # Batch normalization
        self.fc2 = nn.Linear(128, 64)          # Intermediate layer
        self.bn2 = nn.BatchNorm1d(64)          # Batch normalization
        self.fc3 = nn.Linear(64, output_dim)   # Output layer

    def forward(self, embedding_1, embedding_2):
        # Concatenate the two embeddings along the feature dimension
        fused_embedding = torch.cat((embedding_1, embedding_2), dim=1)
        
        x = F.relu(self.bn1(self.fc1(fused_embedding)))  # Apply ReLU activation after batch normalization
        x = F.relu(self.bn2(self.fc2(x)))               # Apply ReLU activation after batch normalization
        x = self.fc3(x)  # No activation for the final layer
        return x
