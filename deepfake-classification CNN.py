import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import gc

class KaggleDataset(Dataset):
    def __init__(self, file, directory, transform):
        self.data = pd.read_csv(file)
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_id = self.data.iloc[idx, 0]
        image_path = os.path.join(self.directory, f"{image_id}.png")
        image = Image.open(image_path)
        image = self.transform(image)
        
        if len(self.data.columns) > 1:
            label = self.data.iloc[idx, 1]
            return image, label
        else:
            return image

class CNN_v1(nn.Module):
    def __init__(self):
        super(CNN_v1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
                 
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
   
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNN_v2(nn.Module):
    def __init__(self):
        super(CNN_v2, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
                 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
   
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=2), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_cnn(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 15
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = (correct_predictions / total_predictions) * 100
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1} Train Acc: {train_acc}, Val Acc: {val_acc}')
        
        if counter >= patience:
            print(f"Early stopping")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def soft_voting_predict(model1, model2, test_loader, device, weights=[0.45, 0.55]):
    print(f"Performing soft voting")
    
    model1.eval()
    model2.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs1 = model1(data)
            outputs2 = model2(data)
            probs1 = torch.softmax(outputs1, dim=1)
            probs2 = torch.softmax(outputs2, dim=1)
            ensemble_probs = weights[0] * probs1 + weights[1] * probs2
            ensemble_preds = torch.argmax(ensemble_probs, dim=1)
            all_predictions.extend(ensemble_preds.cpu().numpy())
    
    return np.array(all_predictions)

def soft_voting_predict_validation(model1, model2, val_loader, device, weights=[0.45, 0.55]):
    model1.eval()
    model2.eval()
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs1 = model1(data)
            outputs2 = model2(data)
            probs1 = torch.softmax(outputs1, dim=1)
            probs2 = torch.softmax(outputs2, dim=1)
            ensemble_probs = weights[0] * probs1 + weights[1] * probs2
            ensemble_preds = torch.argmax(ensemble_probs, dim=1)
            total_predictions += targets.size(0)
            correct_predictions += (ensemble_preds == targets).sum().item()
    
    ensemble_acc =(correct_predictions / total_predictions) * 100
    return ensemble_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(16)
    print(f"Using device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = KaggleDataset(
        file='deepfake-classification-unibuc/train.csv',
        directory='deepfake-classification-unibuc/train',
        transform=train_transform
    )
    
    val_dataset = KaggleDataset(
        file='deepfake-classification-unibuc/validation.csv',
        directory='deepfake-classification-unibuc/validation',
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Training Model 1")
    model1 = CNN_v1()
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'max', patience=5, factor=0.5)
    
    model1 = train_cnn(
        model1, train_loader, val_loader, 
        criterion, optimizer1, scheduler1, 
        num_epochs=100, device=device
    )

    torch.save(model1.state_dict(), 'model1.pth')
    print("\nModel 1 complete!")
    
    del optimizer1, scheduler1
    gc.collect()

    print("\nTraining Model 2")
    model2 = CNN_v2()
    optimizer2 = optim.Adam(model2.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'max', patience=5, factor=0.5)
    
    model2 = train_cnn(
        model2, train_loader, val_loader, 
        criterion, optimizer2, scheduler2, 
        num_epochs=100, device=device
    )

    torch.save(model2.state_dict(), 'model2.pth')
    print("\nModel 2 completed!")   
    del optimizer2, scheduler2
    gc.collect()

    ensemble_val_acc = soft_voting_predict_validation(model1, model2, val_loader, device)
    print(f"Soft voting acc: {ensemble_val_acc}")

    test_dataset = KaggleDataset(
        file='deepfake-classification-unibuc/test.csv',
        directory='deepfake-classification-unibuc/test',
        transform=val_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    ensemble_predictions = soft_voting_predict(model1, model2, test_loader, device, weights=[0.55, 0.45])
    
    test = pd.read_csv('deepfake-classification-unibuc/test.csv')
    submission = pd.DataFrame({
        'image_id': test['image_id'],
        'label': ensemble_predictions
    })
    submission.to_csv('soft_voting_submission.csv', index=False)
    print("\nSubmission saved!")

if __name__ == "__main__":
    main()