import timm
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


# Gender ResNet50
class GenderClassifierRS(nn.Module):
  def __init__(self, num_classes=2):
    super().__init__()
    self.base_model = timm.create_model('resnet50', pretrained=True)
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])
    self.classifier = nn.Linear(self.base_model.get_classifier().in_features, num_classes)

  def forward(self,x):
    x = self.features(x)
    output = self.classifier(x)
    return output
  
class GenderClassifierEN(nn.Module):
  def __init__(self, num_classes=2):
    super().__init__()
    self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])
    self.classifier = nn.Linear(self.base_model.get_classifier().in_features, num_classes)

  def forward(self,x):
    x = self.features(x)
    output = self.classifier(x)
    return output    

class GenderClassifierRETFound(nn.Module):
  pass


print(timm.list_models())