from toolbench.retrieval.api_evaluator import APIEvaluator
from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
import pandas as pd
import os
import json
import numpy as np
import sklearn
from utils.retrieval import Indexer
from utils.embedding import get_embeddings, get_openai_embeddings
import click
import tiktoken


'''
python -m evaluation.eval_encoder \
    --model_name_or_path ../models/ToolLlama/retriever/bert_G1 \
    --data_path ../ToolBench/data/retrieval/G1 \
    --result_path "data/results/retrieval/bert_G1.json"
'''
@click.command()
@click.option("--model_name_or_path", type=str, default="")
@click.option("--api_key", type=str, default="")
@click.option("--stage", type=str, default="G1")
@click.option("--split", type=str, default="test")
@click.option("--result_path", type=str, default="")
@click.option("--corpus", type=str, default="")
def main(
    model_name_or_path: str,
    api_key: str,
    stage: str,
    split: str,
    result_path: str,
    corpus: str
):

    ir_test_queries = {}
    ir_relevant_docs = {}
    # train_samples = []
    def process_retrieval_ducoment(documents_df):
        ir_corpus = {}
        corpus2tool = {}
        for row in documents_df.itertuples():
            doc = json.loads(row.document_content)
            ir_corpus[row.docid] = (doc.get('category_name', '') or '') + ', ' + \
            (doc.get('tool_name', '') or '') + ', ' + \
            (doc.get('api_name', '') or '') + ', ' + \
            (doc.get('api_description', '') or '') + \
            ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
            ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
            ', return_schema: ' + json.dumps(doc.get('template_response', ''))
            corpus2tool[(doc.get('category_name', '') or '') + ', ' + \
            (doc.get('tool_name', '') or '') + ', ' + \
            (doc.get('api_name', '') or '') + ', ' + \
            (doc.get('api_description', '') or '') + \
            ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
            ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
            ', return_schema: ' + json.dumps(doc.get('template_response', ''))] = doc['category_name'] + '\t' + doc['tool_name'] + '\t' + doc['api_name']
        return ir_corpus, corpus2tool
        
    data_path = f"data/retrieval/{stage}"

    if corpus == "G123":
        documents_df = pd.read_csv('data/retrieval/corpus_G123.tsv', sep='\t')
        print("Using G123 corpus")
    else:
        documents_df = pd.read_csv(os.path.join(data_path, 'corpus.tsv'), sep='\t')
        print(f"Using {data_path} corpus")

    ir_corpus, _ = process_retrieval_ducoment(documents_df)

    # train_queries_df = pd.read_csv(os.path.join(data_path, 'train.query.txt'), sep='\t', names=['qid', 'query'])
    # for row in train_queries_df.itertuples():
    #     ir_train_queries[row.qid] = row.query
    query_path = "test.query.txt" if split == "test" else f"test_{stage}_{split}.query.txt"
    test_queries_df = pd.read_csv(os.path.join(data_path, query_path), sep='\t', names=['qid', 'query'])
    for row in test_queries_df.itertuples():
        ir_test_queries[row.qid] = row.query


    # labels_df = pd.read_csv(os.path.join(data_path, 'qrels.train.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    # for row in labels_df.itertuples():
    #     sample = InputExample(texts=[ir_train_queries[row.qid], ir_corpus[row.docid]], label=row.label)
    #     train_samples.append(sample)
    rel_path = "qrels.test.tsv" if split == "test" else f"qrels.test_{stage}_{split}.tsv"
    labels_df = pd.read_csv(os.path.join(data_path, rel_path), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    for row in labels_df.itertuples():
        ir_relevant_docs.setdefault(row.qid, set()).add(row.docid)

    # ir_evaluator = APIEvaluator(ir_test_queries, ir_corpus, ir_relevant_docs)
    # model_save_path = "toolbench/retrieval/retriever/"
    # ir_evaluator(model, output_path=model_save_path)
    if corpus == "G123":
        with open(f"data/retrieval/{stage}_toolid_to_full_tool_id.json", "r") as f:
            domain_toolid_to_full_toolid = json.load(f)
        new_ir_relevant_docs = {}
        for qid, docids in ir_relevant_docs.items():
            new_ir_relevant_docs[qid] = set()
            for docid in docids:
                try:
                    new_ir_relevant_docs[qid].add(domain_toolid_to_full_toolid[str(docid)])
                except KeyError:
                    print(f"Missing {docid}")
        ir_relevant_docs = new_ir_relevant_docs


    # Search
    queries = list(ir_test_queries.values())
    queries_ids = list(ir_test_queries.keys())

    corpus_ids = list(ir_corpus.keys())
    corpus_texts = [ir_corpus[cid] for cid in corpus_ids]



    def num_tokens_from_string(string, encoding_name: str):
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    num_tokens = 0
    for text in corpus_texts:
        num_tokens += num_tokens_from_string(text, "cl100k_base")
    print(f"Number of instances: {len(corpus_texts)}")
    print(f"Number of tokens: {num_tokens}")

    query_embeddings = get_openai_embeddings(
        queries,
        batch_size=512,
        model="text-embedding-3-large",
        api_key=api_key
    )
    # Check if file exists
    if corpus == "G123":
        doc_embedding_path = "data/retrieval/corpus_embeddings.npy"
    else:
        doc_embedding_path = f"{data_path}/corpus_embeddings.npy"

    if os.path.exists(doc_embedding_path):
        doc_embeddings = np.load(doc_embedding_path)
        print(f"Loaded embeddings from {doc_embedding_path}")
    else:
        doc_embeddings = get_openai_embeddings(
            corpus_texts,
            batch_size=512,
            model="text-embedding-3-large",
            api_key=api_key
        )
        np.save(doc_embedding_path, doc_embeddings)

    indexer = Indexer(doc_embeddings, doc_embeddings.shape[1], ids=corpus_ids)

    def compute_ndcg(relevant_docs_ids, score_docid, k):
        # Build the ground truth relevance scores and the model's predicted scores
        length = len(corpus_ids)
        # length = 100000
        true_relevance = np.zeros(length)
        predicted_scores = np.zeros(length)
        top_hits = score_docid
        for hit in top_hits:
            predicted_scores[corpus_ids.index(hit[1])] = hit[0]
            if hit[1] in relevant_docs_ids:
                true_relevance[corpus_ids.index(hit[1])] = 1
            # for relevant_doc_id in relevant_docs_ids:
            #     true_relevance[relevant_doc_id] = 1

        return sklearn.metrics.ndcg_score([true_relevance], [predicted_scores], k=k)


    relevant_docs = ir_relevant_docs
    # print(type(relevant_docs))

    # Here Toolbench simply sorted all documents by cosine similarity
    # We take 100 as a sufficient large number to reproduce the results
    # However, may be set top_n to be k is more resonable.
    scores_docids = indexer.search(query_embeddings, top_n=100)
    result_scores = {}
    for k in [1, 3, 5]:
        ndcg_scores = []
        for query_id, scores_docid in zip(queries_ids, scores_docids):
            # print(query_id)
            relevant_docs_ids = relevant_docs[query_id]
            ndcg_score = compute_ndcg(relevant_docs_ids, scores_docid, k=k)
            ndcg_scores.append(ndcg_score)
        result_scores[f"ndcg@{k}"] = np.mean(ndcg_scores)
        print(np.mean(ndcg_scores))

    # We save the evaluation results here
    results = {
        "model": model_name_or_path,
        "ndcg": result_scores,
        "logs": []
    }
    for query_id, query, scores_docid in zip(queries_ids, queries, scores_docids):
        # print(query_id)
        # print(type(relevant_docs))
        relevant_docs_ids = relevant_docs[query_id]
        relevant_docs_logs = []
        for rid in relevant_docs_ids:
            relevant_docs_logs.append({
                "docid": rid,
                "text": ir_corpus[rid]
            })
        pred_ids = [docid for score, docid in scores_docid][:5]
        pred_docs_logs = []
        for pid in pred_ids:
            pred_docs_logs.append({
                "docid": pid,
                "text": ir_corpus[pid]
            })
        results['logs'].append({
            "query_id": query_id,
            "query": query,
            "relevant_docs": relevant_docs_logs,
            "pred_docs": pred_docs_logs
        })

    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()