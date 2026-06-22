import timm
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from config import RETFOUND_REPO


# Gender ResNet50
class GenderClassifierRS(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = timm.create_model('resnet18', pretrained=True,num_classes=0)
        self.classifier = nn.Linear(self.base_model.num_features, num_classes)

    def forward(self,x):
        x = self.base_model(x)
        output = self.classifier(x)
        return output
    

class GenderClassifierRETFound(nn.Module): 
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(RETFOUND_REPO)
        in_features = self.base_model.config.hidden_size
        self.classifier = nn.Linear(in_features,num_classes)

    def forward(self,x):
        outputs = self.base_model(x)
        cls_feature = outputs.last_hidden_state[:, 0, :]
        output = self.classifier(cls_feature)
        return output
    
class AgeClassifierRS(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.base_model = timm.create_model('resnet18', pretrained=True,num_classes=0)
        self.classifier = nn.Linear(self.base_model.num_features, num_classes)

    def forward(self,x):
        x = self.base_model(x)
        output = self.classifier(x)
        return output
    

class AgeClassifierRETFound(nn.Module): 
    def __init__(self, num_classes=5):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(RETFOUND_REPO)
        in_features = self.base_model.config.hidden_size
        self.classifier = nn.Linear(in_features,num_classes)

    def forward(self,x):
        outputs = self.base_model(x)
        cls_feature = outputs.last_hidden_state[:, 0, :]
        output = self.classifier(cls_feature)
        return output



class AgeRegressionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model('resnet50', pretrained=True,num_classes=0)
        self.classifier = nn.Linear(self.base_model.num_features, 1)

    def forward(self,x):
        x = self.base_model(x)
        output = self.classifier(x)
        return output
    

class AgeRegressionRETFound(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(RETFOUND_REPO)
        in_features = self.base_model.config.hidden_size
        self.classifier = nn.Linear(in_features,1)


    def forward(self,x):
        outputs = self.base_model(x)
        cls_feature = outputs.last_hidden_state[:, 0, :]
        output = self.classifier(cls_feature)
        return output

class MultiTaskResNet(nn.Module):
    def __init__(self,  output_dimension_0 = 2, output_dimension_1 = 1):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(RETFOUND_REPO)
        in_features = self.base_model.config.hidden_size
        
        self.classifier_0 = nn.Linear(in_features,output_dimension_0)
        self.classifier_1 = nn.Linear(in_features,output_dimension_1)
        
    def forward(self, x):
        x = self.base_model(x)
        output_0 = self.classifier_0(x)
        output_1 = self.classifier_1(x)
        
        return output_0, output_1 #sex, age
            
        

class MultiTaskRETFound(nn.Module):
    def __init__(self, output_dimension_0 = 2, output_dimension_1 = 1):
        super().__init__()
        self.base_model = timm.create_model('resnet50', pretrained=True,num_classes=0)
        in_features = self.base_model.config.hidden_size
               
        self.classifier_0 = nn.Linear(in_features,output_dimension_0)
        self.classifier_1 = nn.Linear(in_features,output_dimension_1)


    def forward(self,x):
        outputs = self.base_model(x)
        cls_feature = outputs.last_hidden_state[:, 0, :]
        output_0 = self.classifier_0(cls_feature)
        output_1 = self.classifier_1(cls_feature)
        return output_0, output_1


# class GenderClassifierEN(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.base_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
#         self.classifier = nn.Linear(self.base_model.num_features, num_classes)

#     def forward(self,x):
#         x = self.base_model(x)
#         output = self.classifier(x)
#         return output    