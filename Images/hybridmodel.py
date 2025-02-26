import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset Paths
data_dir = "D:/Research/Snaked-Project/Images"
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

# Data Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load Pretrained Models
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
resnext50 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

# Modify Fully Connected Layers
num_features_50 = resnet50.fc.in_features
num_features_101 = resnet101.fc.in_features
num_features_x = resnext50.fc.in_features
num_features_incep = inception.fc.in_features

resnet50.fc = nn.Identity()
resnet101.fc = nn.Identity()
resnext50.fc = nn.Identity()
inception.fc = nn.Identity()

# Hybrid Model
class HybridResNetInception(nn.Module):
    def __init__(self, resnet50, resnet101, resnext50, inception, num_classes):
        super(HybridResNetInception, self).__init__()
        self.resnet50 = resnet50
        self.resnet101 = resnet101
        self.resnext50 = resnext50
        self.inception = inception
        self.fc = nn.Linear(num_features_50 + num_features_101 + num_features_x + num_features_incep, num_classes)
    
    def forward(self, x):
        x1 = self.resnet50(x)
        x2 = self.resnet101(x)
        x3 = self.resnext50(x)
        x4 = self.inception(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fc(x)
        return x

# Initialize Hybrid Model
model = HybridResNetInception(resnet50, resnet101, resnext50, inception, len(train_dataset.classes)).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Training Function
def train_model(model, train_loader, epochs=15):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")

# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy of Hybrid Model is: {accuracy:.2f}%")
    return accuracy

# Train and Evaluate
train_model(model, train_loader, epochs=15)
evaluate_model(model, test_loader)
