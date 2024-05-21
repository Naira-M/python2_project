import torch


# Function to load a saved model checkpoint and evaluate on test data
def evaluate_saved_model(model, testloader, criterion, checkpoint_path):
    # Load the saved model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()  

    correct, total, running_loss = 0, 0, 0.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Evaluate on test data
    with torch.no_grad():
        for images, labels in testloader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Update statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy and average loss
    test_accuracy = correct / total
    test_loss = running_loss / len(testloader.dataset)

    return test_loss, test_accuracy
