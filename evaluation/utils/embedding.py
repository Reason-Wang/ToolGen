import torch
import numpy as np
from openai import OpenAI
import tiktoken
from tqdm import tqdm


def truncate_text_tokens(text, max_tokens=4096):
    # Truncate texts to 4096 tokens
    encoding = tiktoken.get_encoding("cl100k_base")
    return encoding.encode(text)[:max_tokens]


def get_openai_embeddings(texts, batch_size, model, api_key):
    client =  OpenAI(api_key=api_key)
    texts = [text.replace("\n", " ") for text in texts]
    # Truncate texts to 4096 tokens
    truncated_text_tokens = [truncate_text_tokens(text) for text in texts]

    embeddings = []
    for i in tqdm(range(0, len(truncated_text_tokens), batch_size)):
        batch = truncated_text_tokens[i:i + batch_size]
        data = client.embeddings.create(input=batch, model=model).data
        embedding = [d.embedding for d in data]
        embeddings.extend(embedding)
    
    return np.array(embeddings)
    # return client.embeddings.create(input=texts, model=model).data[0].embedding


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

