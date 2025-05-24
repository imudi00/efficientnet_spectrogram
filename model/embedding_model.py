import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetEmbedding(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        for name, param in self.base_model.named_parameters():
            if "features.6" in name or "features.7" in name:
                param.requires_grad = True

        self.features = self.base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(1280, embedding_size)
        self.l2_norm = nn.functional.normalize

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = self.l2_norm(x, dim=1)
        return x