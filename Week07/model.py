import torch
import torch.nn as nn


class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(512, 90)  # Assuming 90 classes, adjust accordingly
        self.temperature = nn.Parameter(torch.ones(1))  # Temperature parameter for scaling

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        logits = self.classifier(features.float())
        return logits / self.temperature.exp()