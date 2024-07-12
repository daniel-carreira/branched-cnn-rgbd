import torch
import torch.nn as nn
import torchvision.models as models

class BranchedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # RGB Network
        self._rgb_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._rgb_feature_extractor = nn.Sequential(*list(self._rgb_feature_extractor.children())[:-1])

        # Depth Network
        self._depth_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._depth_feature_extractor = nn.Sequential(*list(self._depth_feature_extractor.children())[:-1])
        self._depth_feature_extractor[0] = nn.Conv2d(1, 64, kernel_size=7)

        # Fusion
        self.gru = nn.GRU(input_size=1024, hidden_size=128, num_layers=2, dropout=0.4)

        # Classification
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x1, x2):
        x1 = self._rgb_feature_extractor(x1)
        x1 = torch.flatten(x1, 1)
        x2 = self._depth_feature_extractor(x2)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), dim=1)
        x = x.unsqueeze(0)
        _, hidden = self.gru(x)
        x = self.fc(hidden[-1])
        x = self.softmax(x)
        return x