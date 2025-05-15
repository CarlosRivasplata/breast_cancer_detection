import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pyarrow.parquet as pq
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
from tqdm import tqdm
import pydicom
import numpy as np
import datetime

# Constants
DATA_ABS_PATH = Path("./data/CBIS-DDSM").resolve()
IMAGES_ABS_PATH = Path("./data/CBIS-DDSM/CBIS-DDSM/").resolve()
OUTPUT_TRAIN = DATA_ABS_PATH / "meta/train.parquet"
OUTPUT_TEST = DATA_ABS_PATH / "meta/test.parquet"

LABEL_MAP = {
    "BENIGN": 0,
    "BENIGN_WITHOUT_CALLBACK": 1,
    "MALIGNANT": 2
}

def load_dicom_image(path):
    """Load a DICOM image and convert it to PIL Image."""
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    # Normalize to 0-255 range
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    return Image.fromarray(image)

# Dataset Class
class CBISDDSMDataset(Dataset):
    def __init__(self, parquet_path, transform=None, label_map=None):
        self.table = pq.read_table(parquet_path)
        self.num_rows = self.table.num_rows
        self.transform = transform
        self.label_map = label_map or LABEL_MAP

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1).to_pydict()

        try:
            img1 = load_dicom_image(IMAGES_ABS_PATH / row["image_path"][0].lstrip('/'))
            img2 = load_dicom_image(IMAGES_ABS_PATH / row["cropped_path"][0].lstrip('/'))
            img3 = load_dicom_image(IMAGES_ABS_PATH / row["roi_path"][0].lstrip('/'))
        except Exception as e:
            raise RuntimeError(f"Image load failed at index {idx}: {e}")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Stack as channels: shape will be (3, H, W)
        image = torch.cat([img1, img2, img3], dim=0)

        label = self.label_map[row["pathology"][0]]
        return image, label

# Training pipeline
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# Main training script
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Datasets and loaders
    train_dataset = CBISDDSMDataset(OUTPUT_TRAIN, transform=transform)
    test_dataset = CBISDDSMDataset(OUTPUT_TEST, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    model = models.resnet50(pretrained=True)
    # Use default conv1 (3 channels)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3-class output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Save final model
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_save_path = Path(__file__).parent / f"resnet_cbis_ddsm_{timestamp}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    main() 