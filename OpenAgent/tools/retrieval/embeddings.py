import numpy as np
import torch

def get_embeddings(model, device, texts, batch_size=16):
    model.eval()
    model.to(device)
    # tbar = tqdm(dataloader)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.append(model.encode(batch, device=device))
    return np.concatenate(embeddings)