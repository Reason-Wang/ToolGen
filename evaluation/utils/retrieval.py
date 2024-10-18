import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import faiss

class BM25Indexer:
    def __init__(self, corpus, ids=None):
        self.corpus = corpus
        if ids is None:
            self.ids = list(range(embeddings.shape[0]))
        else:
            self.ids = ids
        self.tokenized_corpus = [word_tokenize(document.lower()) for document in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(self, queries, top_n=5):
        tokenized_queries = [word_tokenize(query.lower()) for query in queries]
        docs_scores = [self.bm25.get_scores(tokenized_query) for tokenized_query in tokenized_queries]
        docs_scores = [[(score, idx) for idx, score in enumerate(doc_scores)] for doc_scores in docs_scores]
        scores_ids = [sorted(doc_scores, reverse=True)[:top_n] for doc_scores in docs_scores]
        
        new_scores_ids = []
        for score_ids in scores_ids:
            new_score_ids = []
            for score, idx in score_ids:
                new_score_ids.append((score, self.ids[idx]))
            new_scores_ids.append(new_score_ids)
        
        return new_scores_ids

    


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


if __name__ == "__main__":
    texts = [
        "A man standing in front of a building",
        "Mooncake is a Chinese bakery product traditionally eaten during the Mid-Autumn Festival",
        "PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables ...",
        "The quick brown fox jumps over the lazy dog",
        "Barack Obama was the 44th president of the United States",
        "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China",
        "The domestic dog is a domesticated descendant of the wolf",
        "The original cat species to evolve into the domestic cat is the African wildcat",
        "Camels and llamas are common pack animals",
    ]
    query = ["Give me some facts about animals.", "What is the Great Wall of China?"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    query_embedding = model.encode(query)
    indexer = Indexer(embeddings, embeddings.shape[1])
    scores_docids = indexer.search(query_embedding, top_n=3)
    print(scores_docids)
    top_ids = [[score_id[1] for score_id in score_ids] for score_ids in scores_docids]
    print(top_ids)
    best_docs = [texts[ids[0]] for ids in top_ids]
    print(best_docs)
