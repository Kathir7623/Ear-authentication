import torch
import torch.nn as nn
import torch.nn.functional as F

class EarNet(nn.Module):
    """
    Custom CNN Architecture for Ear Biometric Recognition.
    Designed to be lightweight yet effective for feature extraction from 2D ear images.
    """
    def __init__(self, embedding_size=128):
        super(EarNet, self).__init__()
        
        # Block 1: Feature Extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers for Embedding
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, embedding_size)
        
    def forward(self, x):
        # Convolutional Layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Embedding
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 Normalization (Crucial for metric learning/triplet loss)
        x = F.normalize(x, p=2, dim=1)
        
        return x

class SiameseEarNet(nn.Module):
    """
    Siamese Network Wrapper for Ear Verification.
    Takes two images and calculates the similarity between their embeddings.
    """
    def __init__(self, embedding_size=128):
        super(SiameseEarNet, self).__init__()
        self.base_network = EarNet(embedding_size)
        
    def forward(self, img1, img2):
        out1 = self.base_network(img1)
        out2 = self.base_network(img2)
        return out1, out2

if __name__ == "__main__":
    # Test the model
    model = EarNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model Architecture Initialized Successfully.")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Embedding Shape: {output.shape}")
