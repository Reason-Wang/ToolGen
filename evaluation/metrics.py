
import numpy as np
from tqdm import trange
import sklearn
from utils.retrieval import Indexer
from utils.embedding import get_embeddings


def ndcg_score(model, tokenizer, queries, corpus, relevant_docs, batch_size=32, corpus_chunk_size=32):
    query_embeddings = get_embeddings(
        model,
        tokenizer,
        device="cuda",
        texts=queries,
    )
    doc_embeddings = get_embeddings(
        model,
        tokenizer,
        device="cuda",
        texts=list(corpus.values()),
    )

    indexer = Indexer(doc_embeddings, doc_embeddings.shape[1], ids=list(corpus.keys()))
    scores_docids = indexer.search(query_embeddings, 5)
    queries_ids = list(corpus.keys())
    ndcg_scores = compute_ndcg_scores(queries_ids, scores_docids, relevant_docs)

    return ndcg_scores

def compute_ndcg(relevant_docs_ids, score_docid, k):
    # Build the ground truth relevance scores and the model's predicted scores
    length = len(corpus_ids)
    true_relevance = np.zeros(length)
    predicted_scores = np.zeros(length)
    top_hits = score_docid
    for hit in top_hits:
        predicted_scores[corpus_ids.index(hit[1])] = hit[0]
        if hit[1] in relevant_docs_ids:
            true_relevance[corpus_ids.index(hit[1])] = 1

    return sklearn.metrics.ndcg_score([true_relevance], [predicted_scores], k=k)


def compute_ndcg_scores(queries_ids, scores_docids, relevant_docs):
    ndcg_scores = []
    for query_id, scores_docid in zip(queries_ids, scores_docids):
        relevant_docs_ids = relevant_docs[query_id]
        ndcg_score = compute_ndcg(relevant_docs_ids, scores_docid, k=5)
        ndcg_scores.append(ndcg_score)
    return np.mean(ndcg_scores)