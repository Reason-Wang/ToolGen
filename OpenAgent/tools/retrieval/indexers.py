import faiss
import numpy as np

class Indexer:
    def __init__(self, embeddings, vector_size, ids=None, similarity="cosine"):
        self.index = faiss.IndexFlatIP(vector_size)
        self.similarity = similarity
        if similarity == "cosine":
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, None]
        self.index.add(embeddings)
        if ids is None:
            self.ids = list(range(embeddings.shape[0]))
        else:
            self.ids = ids

    
    def add(self, embeddings, ids=None):
        if self.similarity == "cosine":
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, None]
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings)
        if ids is None:
            self.ids.extend(list(range(self.ids[-1] + 1, self.ids[-1] + 1 + embeddings.shape[0])))
        else:
            self.ids.extend(ids)

    def search(self, queries: np.array, top_n: int):
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)
        try:
            if self.similarity == "cosine":
                queries /= np.linalg.norm(queries, axis=1)[:, None]
            scores, indexes = self.index.search(queries, top_n)
        except AttributeError:
            print(queries)
        scores_ids = []
        for top_n_score, top_n_idx in zip(scores, indexes):
            top_n_score_id = []
            for s, i in zip(top_n_score, top_n_idx):
                top_n_score_id.append((s, self.ids[i]))
            scores_ids.append(top_n_score_id)

        return scores_ids