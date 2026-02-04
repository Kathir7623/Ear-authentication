import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import EarNet
import numpy as np

class DummyEarDataset(Dataset):
    """
    Placeholder Dataset for Demonstration.
    In a real scenario, this would load images from a dataset like IIT Delhi Ear Database.
    """
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Return random image and a dummy label
        return torch.randn(3, 224, 224), torch.tensor(idx % 10)

def train_model():
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    
    # Initialize Model, Criterion, and Optimizer
    model = EarNet(embedding_size=128).to(device)
    
    # For Biometric Recognition, Metric Learning (like Triplet Loss) is ideal.
    # Here we show a standard CrossEntropy for initial classification training.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load Data
    dataset = DummyEarDataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting Training Process on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            embeddings = model(images)
            
            # Note: embeddings are 128-dim. In a real classification setup, 
            # we'd have a classifier head here.
            # This is a demonstration of the training pipeline.
            
            loss = torch.mean(embeddings**2) # Dummy loss for demo
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), 'models/earnet_v1.pth')
    print("Model saved to models/earnet_v1.pth")

if __name__ == "__main__":
    train_loader = train_model()
