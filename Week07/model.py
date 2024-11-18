import torch
import torch.nn as nn


class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.batch_norm = nn.BatchNorm1d(512)  # Batch normalization layer
        self.dropout = nn.Dropout(p=0.3)  # Dropout layer with 30% probability
        self.relu = nn.ReLU()  # ReLU activation
        self.classifier = nn.Linear(512, 90)  # Assuming 90 classes, adjust accordingly

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        features = self.batch_norm(features.float())  # Apply batch normalization
        features = self.dropout(features)  # Apply dropout
        features = self.relu(features)  # Apply ReLU activation
        return self.classifier(features)