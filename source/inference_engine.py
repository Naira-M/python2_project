from source.data_utils import construct_loader

import torch
import torch.nn as nn
import torchvision.models as models

num_classes = 2


# Define the FaceClassifier inference engine
class FaceClassifier:
    def __init__(self, model_path, batch_size=32):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def classify(self, image_paths, image_labels):
        data_loader = construct_loader(image_paths, image_labels, batch_size=self.batch_size, train=False)
        
        predictions = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return predictions