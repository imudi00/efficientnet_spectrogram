import torch

def train_triplet(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for anchor, positive, negative in data_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        loss = loss_fn(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

@torch.no_grad()
def validate_triplet(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    for anchor, positive, negative in data_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)
        loss = loss_fn(anchor_embed, positive_embed, negative_embed)
        total_loss += loss.item()
    return total_loss / len(data_loader)