import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

def load_celeba_attr(attr_path):
    """Load CelebA attributes from txt file"""
    with open(attr_path, 'r') as f:
        lines = f.readlines()
    num_images = int(lines[0])
    attr_names = lines[1].strip().split()
    data = {'image_id': [], 'Wearing_Hat': []}
    for line in lines[2:]:
        values = line.strip().split()
        image_id = values[0]
        hat_idx = attr_names.index('Wearing_Hat')
        wearing_hat = 1 if int(values[hat_idx + 1]) == 1 else 0
        
        data['image_id'].append(image_id)
        data['Wearing_Hat'].append(wearing_hat)
    
    return pd.DataFrame(data)

class HatDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(HatDetector, self).__init__()
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class CelebADataset(Dataset):
    def __init__(self, root_dir, image_files, labels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = image_files
        self.labels = labels
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            image = Image.new('RGB', (224, 224), 'black')
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {len(self.image_files)} test images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            image = Image.new('RGB', (224, 224), 'black')
        
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cuda'):
    model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_acc:.2f}%')
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_acc:.2f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

def predict(model, test_loader, device='cuda'):
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            for img_id, pred in zip(image_ids, predicted):
                img_id_no_ext = os.path.splitext(img_id)[0]
                predictions[img_id_no_ext] = 'Hat' if pred == 1 else 'No Hat'
    
    return predictions

def create_dataset_split(root_dir, attributes_df):
    """Create train and validation datasets"""
    image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        raise RuntimeError(f"No image files found in {root_dir}")
    print(f"Found {len(image_files)} images in {root_dir}")
    attr_map = {row['image_id']: row['Wearing_Hat'] for _, row in attributes_df.iterrows()}
    valid_images = []
    labels = []
    for img_file in image_files:
        if img_file in attr_map:
            valid_images.append(img_file)
            labels.append(attr_map[img_file])
    
    print(f"Using {len(valid_images)} images that have corresponding attributes")
    if len(valid_images) == 0:
        print("Warning: No valid images found!")
        print("First few attribute map keys:", list(attr_map.keys())[:5])
        print("First few image files:", image_files[:5])
        raise RuntimeError("No valid images found")
    num_images = len(valid_images)
    train_size = int(0.8 * num_images)
    
    train_images = valid_images[:train_size]
    train_labels = labels[:train_size]
    val_images = valid_images[train_size:]
    val_labels = labels[train_size:]
    
    return train_images, train_labels, val_images, val_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print('Loading CelebA dataset...')
    attributes_df = load_celeba_attr('imgs/Anno-20250321T194650Z-001/Anno/list_attr_celeba.txt')
    img_dir = 'imgs/img/img_align_celeba/img_align_celeba'
    train_images, train_labels, val_images, val_labels = create_dataset_split(img_dir, attributes_df)
    print(f'Training set size: {len(train_images)}, Validation set size: {len(val_images)}')
    train_dataset = CelebADataset(img_dir, train_images, train_labels, transform=transform)
    val_dataset = CelebADataset(img_dir, val_images, val_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    model = HatDetector()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print('Starting training...')
    train_model(model, train_loader, val_loader, criterion, optimizer, device=device)
    model.load_state_dict(torch.load('best_model.pth'))
    print('Loading test dataset...')
    test_dataset = TestDataset('test_set/test_set', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print('Making predictions...')
    predictions = predict(model, test_loader, device=device)
    submission = pd.DataFrame({
        'id': list(predictions.keys()),
        'class': list(predictions.values())
    })
    submission.to_csv('submission.csv', index=False)
    print('Submission file created successfully!')

if __name__ == '__main__':
    main() 