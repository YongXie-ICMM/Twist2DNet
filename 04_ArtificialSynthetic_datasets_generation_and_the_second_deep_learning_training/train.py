import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import datetime


class AngleDataset(Dataset):
    """
    Dataset class: used to load image data and their corresponding angle labels.
    """

    def __init__(self, root_dir, mode='train', transform=None, image_files=None):
        """
        Initialize the dataset.
        root_dir: The folder path where images are stored.
        mode: Dataset mode, can be 'train', 'valid', or 'test'.
        transform: Transformations to be applied to the images (e.g., resizing, cropping, etc.).
        image_files: Optionally specify a list of specific image files for this dataset.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        if image_files:
            self.image_files = image_files
        else:
            self.image_files = os.listdir(self.root_dir)

    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get the image and its corresponding angle at the specified index.
        idx: Index of the image.
        return: Tensor representation of the image and its corresponding angle.
        """
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        # Extract angle information from the filename.
        angle = float(self.image_files[idx].split('_')[1])

        if self.mode == 'train' and self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor without other transformations during validation and testing.
            image = transforms.ToTensor()(image)
        return image, angle


class CustomResNet(nn.Module):
    """
    Custom model based on the ResNet framework.
    """

    def __init__(self):
        super(CustomResNet, self).__init__()
        # Load the ResNet-18 model without using pretrained weights.
        self.resnet = models.resnet18(pretrained=False)
        # Replace the last fully connected layer.
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer.
        # Add custom fully connected layers for regression task.
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Add batch normalization layer.
        self.dropout1 = nn.Dropout(0.5)  # Add Dropout layer.
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)  # Batch normalization.
        x = self.dropout1(x)  # Dropout.
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def train_model(model, criterion, optimizer, train_loader, valid_loader, n_epochs, device):
    """
    Function to train the model.
     model: The model to be trained.
     criterion: The loss function to be used.
     optimizer: The optimization algorithm to be used.
     train_loader: The training data loader.
     valid_loader: The validation data loader.
     n_epochs: Number of training epochs.
     device: Training device ('cuda' or 'cpu').
    return: The trained model.
    """
    model.train()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a filename to save training information.
    log_filename = f"Train_{current_time}.txt"

    log_file = open(log_filename, "w")
    for epoch in range(1, n_epochs + 1):

        log_file.write(
            f'\n Epoch: {epoch}\t\n')
        train_loss = 0.0
        valid_loss = 0.0
        i = 0
        j = 0
        # Training phase
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            i = i + 1
            if i % 20 == 0:
                print("Train epoch:{}\n  {}/100".format(epoch, int((i / len(train_loader)) * 100)))
                print("train_loss:{}".format(loss))
                log_file.write(
                    f'Epoch: {epoch} \tTrain precess: {int((i / len(train_loader)) * 100)} \tTraining Loss: {loss:.6f}\n')
        # Validation phase
        model.eval()
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            valid_loss += loss.item() * data.size(0)
            j = j + 1
            if j % 20 == 0:
                print("VAL epoch:{}\n  {}/100".format(epoch, int((j / len(valid_loader)) * 100)))
                print("valid_loss:{}".format(loss))
                log_file.write(
                    f'Epoch: {epoch} \tVal precess: {int((j / len(valid_loader)) * 100)} \tTraining Loss: {loss:.6f}\n')
        train_loss /= len(train_loader.sampler)
        valid_loss /= len(valid_loader.sampler)

        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')
        log_file.write(
            f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}\t\n')
    # Save model weights
    save_path = "./save_weights/mos2_twist_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f'Model weights saved to {save_path}')
    log_file.close()
    return model


def main():
    # Image preprocessing steps: resize to the same size and convert to tensors.
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Define data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomResizedCrop((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    # For validation and testing
    valid_test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Create datasets
    full_dataset = AngleDataset(root_dir='./datasets_path/', transform=train_transform)

    # Split the dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    valid_size = int(0.2 * total_size)
    test_size = total_size - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, valid_size, test_size])

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device is {}".format(device))
    model = CustomResNet().to(device)
    criterion = nn.MSELoss()  # Use mean squared error loss for regression problem.
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Train the model
    n_epochs = 300
    train_model(model, criterion, optimizer, train_loader, valid_loader, n_epochs, device)


if __name__ == '__main__':
    main()
