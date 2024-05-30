import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((126, 126)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DATASET_DIR = 'C:/Users/paulz/Desktop/projet_master/input_data'
seed = torch.Generator().manual_seed(1)
input_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
train_dataset, valid_dataset = random_split(input_dataset, [0.8, 0.2], generator=seed)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

# Modèle Personalisé
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 56 * 56, 128)
#         self.fc2 = nn.Linear(128, num_classes)
    
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 56 * 56)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

num_classes = len(input_dataset.classes)

# model = SimpleCNN(num_classes)

#Modèle Pré-entrainé
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

num_epochs = 10
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# print('Finished Training')

# Reprise TP9
# ---- La boucle sur les epochs

for epoch in range(num_epochs):

  # ---- Entrainement

  model.train()
  train_loss = 0
  train_correct = 0
  for inputs, labels in train_loader:

      inputs = inputs.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      logit_outputs = model(inputs)
      loss = criterion(logit_outputs, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      train_correct += (logit_outputs.argmax(1) == labels).sum().item()

  # ---- Validation

  model.eval()
  valid_loss = 0
  valid_correct = 0

  for inputs, labels in valid_loader:


      inputs = inputs.to(device)
      labels = labels.to(device)


      logit_outputs = model(inputs)
      loss = criterion(logit_outputs, labels)


      valid_loss += loss.item()
      valid_correct += (logit_outputs.argmax(1) == labels).sum().item()

  # ---- Engistrement et affichage

  # Calcul de la perte moyenne et exactitude moyenne
  train_mean_loss = train_loss / len(train_loader.dataset)
  valid_mean_loss = valid_loss / len(valid_loader.dataset)
  train_mean_accuracy = train_correct / len(train_loader.dataset)
  valid_mean_accuracy = valid_correct / len(valid_loader.dataset)
                                     
#   # Sauvegarde dans l'historique pour tensorboard
#   writer.add_scalars("Mean Loss", 
#                      {"Train": train_mean_loss,
#                       "Valid": valid_mean_loss}, 
#                      epoch + 1)
#   writer.add_scalars("Mean accuracy", 
#                      {"Train": train_mean_accuracy,
#                       "Valid": valid_mean_accuracy}, 
#                      epoch + 1)
  
  # Sauvegarde du checkpoint
#   torch.save({"epoch": epoch + 1,
#               "model_state_dict": model.state_dict(),
#               "optimizer_tat_dict": optimizer.state_dict(),
#               "train_loss": train_mean_loss,
#               "valid_loss": valid_mean_loss,
#               "train_accuracy": train_mean_accuracy,
#               "valid_accuracy": valid_mean_accuracy}, 
#              f"{checkpoints_dir}/model_{epoch + 1}.pt")

  # Print
  print(f"Epoch {epoch + 1}/{num_epochs} : "
            f"train loss = {train_mean_loss:.4f}, "
            f"train accuracy = {train_mean_accuracy:.3%}, "
            f"valid loss = {valid_mean_loss:.4f}, "
            f"valid accuracy = {valid_mean_accuracy:.3%}.")