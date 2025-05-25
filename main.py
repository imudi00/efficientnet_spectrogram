import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.embedding_model import EfficientNetEmbedding
from datasets.triplet_dataset import TripletDataset
from utils.train import train_triplet, validate_triplet
from configs import config

transform = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = TripletDataset(root_dir=config.data_root, song_ids=config.train_ids, transform=transform)
val_dataset = TripletDataset(root_dir=config.data_root, song_ids=config.val_ids, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetEmbedding(embedding_size=config.embedding_size).to(device)
loss_fn = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

os.makedirs(config.save_dir, exist_ok=True)
log_path = os.path.join(config.save_dir, "train_val_loss_log.txt")

with open(log_path, "w", encoding="utf-8") as log_file:
    for epoch in range(config.num_epochs):
        train_loss = train_triplet(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate_triplet(model, val_loader, loss_fn, device)
        log_line = f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        print(log_line)
        log_file.write(log_line + "\n")

        save_path = os.path.join(config.model_save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

print(f"\n학습 로그 '{log_path}' 저장")
