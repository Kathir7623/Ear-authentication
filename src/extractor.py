import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from model import EarNet

class FeatureExtractor:
    def __init__(self, model_name='earnet'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'earnet':
            # Our custom ear-specific model
            self.model = EarNet(embedding_size=128)
            # Load weights if they exist
            import os
            weights_path = os.path.join('models', 'earnet_v1.pth')
            if os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print("Loaded custom EarNet weights.")
                
        elif model_name == 'mobilenet_v2':
            # Fast and lightweight for biometrics
            try:
                self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            except:
                self.model = models.mobilenet_v2(pretrained=True)
            self.model.classifier = nn.Identity()
            
        elif model_name == 'resnet18':
            try:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except:
                self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Identity()
            
        self.model.to(self.device)
        self.model.float() # Force all parameters to float32
        self.model.eval()

    def extract(self, preprocessed_image):
        """
        Extracts feature vector (embedding) from preprocessed image.
        input: (3, 224, 224) numpy array
        output: feature vector as numpy array
        """
        img_tensor = torch.from_numpy(preprocessed_image).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tensor)
            
        return features.cpu().numpy().flatten()
