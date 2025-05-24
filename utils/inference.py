import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

@torch.no_grad()
def extract_embeddings(model, inputs, device, batch_size=64):
    model.eval()
    embeddings = []
    if isinstance(inputs, torch.utils.data.DataLoader):
        for batch in inputs:
            batch = batch.to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)
    else:
        inputs = inputs.to(device)
        emb = model(inputs)
        return emb.cpu().numpy()