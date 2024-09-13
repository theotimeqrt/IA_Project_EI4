# pip install torch torchvision pillow

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

class SiameseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Définir les transformations pour les images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

# Exemple de données (à remplacer par tes propres images et labels)
image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg']
labels = [0, 1]  # 0 pour des paires positives, 1 pour des paires négatives

dataset = SiameseDataset(image_paths=image_paths, labels=labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialiser le modèle, la perte et l'optimiseur
model = SiameseNetwork()
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
for epoch in range(10):  # Ajuste le nombre d'époques selon tes besoins
    for img1, img2 in dataloader:
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        label = torch.ones(img1.size(0))  # Exemple : paires positives
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Fonction pour comparer deux images avec le modèle entraîné
def compare_images(img_path1, img_path2, model, transform):
    model.eval()
    image1 = transform(Image.open(img_path1)).unsqueeze(0)
    image2 = transform(Image.open(img_path2)).unsqueeze(0)
    with torch.no_grad():
        output1, output2 = model(image1, image2)
        distance = torch.norm(output1 - output2)
        return distance.item()

# Exemple de comparaison
img_path1 = 'path_to_image1.jpg'
img_path2 = 'path_to_image2.jpg'
distance = compare_images(img_path1, img_path2, model, transform)
print(f'Distance between images: {distance}')

