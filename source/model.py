from datetime import datetime
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from source.utils import load_image


class ImageClassifier:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.model = resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, train_loader, test_loader,
                    checkpoint_path, metadata_path,
                    num_epochs=10, device='cuda'):
        # Move the model to the device
        self.model = self.model.to(device)
        best_valid_loss = float('inf')
        best_valid_acc = 0.0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                loss.backward()
                self.optimizer.step()

                # Update statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc_train = correct_train / total_train
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Train Accuracy: {epoch_acc_train:.4f}')

            # Evaluation phase
            validation_accuracy, validation_loss, _ = self.evaluate_model(
                test_loader, device)
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Validation Loss: {validation_loss:.4f}, '
                  f'Validation Accuracy: {validation_accuracy:.4f}')

            # Save model checkpoint if validation loss improves
            if validation_loss < best_valid_loss:
                best_valid_acc = validation_accuracy
                best_valid_loss = validation_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f'Saving model checkpoint at epoch {epoch+1}')

        # Save model's metadata
        today = datetime.today()
        metadata = {
            "problem": "classification",
            "n_classes": 2,
            "label_to_class": {0: "real", 1: "fake"},
            "model": "Resnet18",
            "dataset_used": "140k-real-and-fake-faces",
            "related_research_papers": ["https://arxiv.org/abs/1512.03385"],
            "validation_loss": best_valid_loss,
            "validation_accuracy": best_valid_acc,
            "training_date": str(today.date())
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        return best_valid_loss

    def evaluate_model(self, test_loader, device='cpu'):
        self.model.eval()  # Set the model to evaluation mode
        correct, total, running_loss = 0, 0, 0.0
        misclassified_images = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                misclassified_images_batch = [
                    (img_path, label.item()) for img_path, label, predicted in zip(
                        test_loader.dataset.image_paths, labels, predicted
                        ) if label != predicted]
                misclassified_images.extend(misclassified_images_batch)

        test_accuracy = correct / total
        test_loss = running_loss / len(test_loader.dataset)
        return test_accuracy, test_loss, misclassified_images

    def predict(self, image_file, device='cpu'):
        image = load_image(image_file)
        image = image.to(device)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        img_class = "Fake" if predicted.item() == 1 else "Real"
        return img_class

    def load_model(self, checkpoint_path, device='cpu'):
        self.model = resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.model.to(device)
