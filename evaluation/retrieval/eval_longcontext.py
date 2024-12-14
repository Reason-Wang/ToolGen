import time
from evaluation.toolbench.retrieval.api_evaluator import APIEvaluator
from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
import pandas as pd
import os
import json
import numpy as np
import sklearn
from evaluation.utils.retrieval import Indexer
from evaluation.utils.embedding import get_embeddings
import click
from evaluation.utils.utils import OpenAIChatModel
import ast
import re
from evaluation.utils.utils import get_toolbench_name

few_shot_prompts = {
    "G1": '''Query: I'm an aviation enthusiast and I'm curious about the current air traffic in my area. Can you fetch the information on all aircraft above 10,000 feet within 1000 kilometers from my location at latitude 40.7128 and longitude -74.0060? I'm particularly interested in their flight numbers, altitudes, and speeds. Id: ['aircraft_scatter_data_for_aircraftscatter']\nQuery: I want to explore the latest trends on Twitter. Can you provide me with a list of recent trending hashtags worldwide? Additionally, suggest popular hashtags in Argentina to stay updated with local discussions. Id: ['get_country_s_hashtags_for_twitter_hashtags', 'get_worldwide_hashtags_for_twitter_hashtags']\nQuery: I'm a crypto blogger and I want to write an article about the top 50 cryptocurrencies. Can you provide me with the current conversion rates for these cryptos in USD? Additionally, I would like to know the algorithm, proof type, and block time for each crypto. Id: ['toptier_summary_for_sciphercrypto', 'conversion_for_sciphercrypto']''',
    "G2": '''Query: I'm a food enthusiast and I'm looking for unique recipes using ingredients like avocado and quinoa. Can you provide me with some interesting recipes and also give me nutritional information about these ingredients? Additionally, can you find any historical facts about the origin of these ingredients? Id: ['get_main_campaigns_for_vouchery_io', 'search_on_ebay_for_ebay_search_result', 'ip2proxy_api_for_ip2proxy', 'get_asin_for_amazon_live_data', 'checkdisposableemail_for_check_disposable_email', 'newlyregistereddomains_for_check_disposable_email', 'emailvalidation_for_check_disposable_email']\nQuery: I'm organizing a company event and I want to create digital badges for the attendees. Can you assist me in generating QR codes that contain the attendees' information? Additionally, I need to reword a PDF document that includes the event schedule. Can you help me extract the text from the PDF and provide it in a downloadable format? Id: ['getpdf_for_reword_pdf', 'getall_for_reword_pdf', 'download_for_reword_pdf', 'generate_qr_code_for_qr_code_generator_v14']\nQuery: I want to analyze the URL 'cleverbikes.myshopify.com' to gather information about the website's platform, language, and ecommerce capabilities. Can you provide me with a detailed analysis? Additionally, decrypt an encrypted text using the secret key 'my-secret-key' and the AES decryption algorithm. Id: ['fish_api_fishes_for_fish_species', 'fish_api_group_for_fish_species', 'fish_api_fish_name_for_fish_species', 'etcompanybywebsite_for_returns_company_info_based_on_the_website', 'decryptstring_for_encryption_api', 'url_analysis_for_url_analysis']''',
    "G3": '''Query: I'm writing a comedy script and I need some hilarious jokes. Fetch jokes from the Chuck Norris Jokes API and provide me with a variety of jokes from the Jokes by API-Ninjas API to inspire my writing. Id: ['jokes_categories_for_chuck_norris', 'jokes_search_for_chuck_norris', 'jokes_random_for_chuck_norris', 'v1_jokes_for_jokes_by_api_ninjas']\nQuery: I'm a stock market enthusiast and want to explore the latest trends. Can you provide me with ownership data, company profiles, and annual distributions for a selection of stocks? Please include the performance IDs 0P0000OQN8 and F00000O2CG. Additionally, provide me with the 24-hour trading data for a variety of tickers. Id: ['stock_v2_get_ownership_for_morning_star', 'stock_v3_get_profile_for_morning_star', 'type_performance_get_annual_distributions_for_morning_star', 'get_24_hours_tickers_for_quantaex_market_data']\nQuery: I want to create a hilarious social media post. Can you provide a random manatee joke to include in the caption? Additionally, fetch some jokes from the Jokes by API-Ninjas endpoint to engage my followers. Finally, suggest some eye-catching visuals or memes that I can pair with the jokes. Id: ['v1_jokes_for_jokes_by_api_ninjas', 'get_by_id_for_manatee_jokes', 'find_all_for_manatee_jokes', 'random_for_manatee_jokes']'''

}
def form_prompt(stage, query):
    with open(f"data/retrieval/{stage}/tools_2000_prompt.txt", 'r') as f:
        tools_prompt = f.read()
    prompt_format = '''Select 5 tools that best match the query. You can use the following tools: {tools_prompt}

Examples:
{few_shot_examples}

Only generate the list of **five** ids for the tools you select, do not generate anything else. Your selection must be from the list of tools provided above.

Query: {query} Id:'''

    full_prompt = prompt_format.format(
        tools_prompt=tools_prompt,
        few_shot_examples=few_shot_prompts[stage],
        query=query
    )
    return full_prompt

'''
python -m evaluation.eval_encoder \
    --model_name_or_path ../models/ToolLlama/retriever/bert_G1 \
    --data_path ../ToolBench/data/retrieval/G1 \
    --result_path "data/results/retrieval/bert_G1.json"
'''
@click.command()
@click.option("--model_name_or_path", type=str, default="")
@click.option("--stage", type=str, default="G1")
@click.option("--split", type=str, default="test")
@click.option("--result_path", type=str, default="")
@click.option("--corpus", type=str, default="")
def main(
    model_name_or_path: str,
    stage: str,
    split: str,
    result_path: str,
    corpus: str
):
    # Load model
    model = OpenAIChatModel(
        model=model_name_or_path,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    

    # Load datasets
    ir_test_queries = {}
    ir_relevant_docs = {}
    def process_retrieval_ducoment(documents_df):
        ir_corpus = {}
        ir_corpus_toolbench_name_2_id = {}
        for row in documents_df.itertuples():
            doc = json.loads(row.document_content)
            ir_corpus[row.docid] = (doc.get('category_name', '') or '') + ', ' + \
            (doc.get('tool_name', '') or '') + ', ' + \
            (doc.get('api_name', '') or '') + ', ' + \
            (doc.get('api_description', '') or '') + \
            ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
            ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
            ', return_schema: ' + json.dumps(doc.get('template_response', ''))
            tool_name = doc.get('tool_name', '')
            api_name = doc.get('api_name', '')
            toolbench_name = get_toolbench_name(tool_name, api_name)
            ir_corpus_toolbench_name_2_id[toolbench_name] = row.docid
        return ir_corpus, ir_corpus_toolbench_name_2_id
        
    data_path = f"data/retrieval/{stage}"

    if corpus == "G123":
        documents_df = pd.read_csv('data/retrieval/corpus_G123.tsv', sep='\t')
        print("Using G123 corpus")
    else:
        documents_df = pd.read_csv(os.path.join(data_path, 'corpus.tsv'), sep='\t')
    ir_corpus, ir_corpus_toolbench_name_2_id = process_retrieval_ducoment(documents_df)
    nonexist_tool_id = len(ir_corpus)
    ir_corpus[nonexist_tool_id] = "nonexist_tool"
    ir_corpus_toolbench_name_2_id["nonexist_tool"] = nonexist_tool_id
    
    query_path = "test.query.txt" if split == "test" else f"test_{stage}_{split}.query.txt"
    test_queries_df = pd.read_csv(os.path.join(data_path, query_path), sep='\t', names=['qid', 'query'])
    for row in test_queries_df.itertuples():
        ir_test_queries[row.qid] = row.query


    rel_path = "qrels.test.tsv" if split == "test" else f"qrels.test_{stage}_{split}.tsv"
    labels_df = pd.read_csv(os.path.join(data_path, rel_path), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    for row in labels_df.itertuples():
        ir_relevant_docs.setdefault(row.qid, set()).add(row.docid)

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
    queries = list(ir_test_queries.values())[:30]
    queries_ids = list(ir_test_queries.keys())[:30]


    corpus_ids = list(ir_corpus.keys())
    corpus = [ir_corpus[cid] for cid in corpus_ids]

    # Get generation from model
    scores_docids = []
    for query in queries:
        full_prompt = form_prompt(stage, query)
        messages = [
            {"role": "user", "content": full_prompt}
        ]
        response = model.generate(messages)
        print(f"Response: {response}")
        # raise RuntimeError("Stop here")
        try:
            pattern = r'\[.*?\]'
            extracted_response = re.findall(pattern, response)[0]
            toolbench_names = ast.literal_eval(extracted_response)
        except Exception as e:
            print(f"Error: {response} gets {e}")
            score_docid = [(1, nonexist_tool_id)] * 100
            scores_docids.append(score_docid)
            continue

        score_docid = []
        for toolbench_name in toolbench_names:
            toolbench_name = toolbench_name.strip()
            try:
                docid = ir_corpus_toolbench_name_2_id[toolbench_name]
            except KeyError as e:
                print(f"Key error: {toolbench_name}")
                continue
            # Don't care about the score
            score_docid.append((1, docid))
        # pad with nonexist_tool
        while len(score_docid) < 100:
            score_docid.append((1, nonexist_tool_id))
        
        scores_docids.append(score_docid)

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
    result_scores = {}
    for k in [1, 3, 5]:
        ndcg_scores = []
        for query_id, scores_docid in zip(queries_ids, scores_docids):
            # print(query_id)
            try:
                relevant_docs_ids = relevant_docs[query_id]
            except KeyError:
                print("No relevant docs for query", query_id)
                continue
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
        try:
            relevant_docs_ids = relevant_docs[query_id]
        except KeyError:
            print("No relevant docs for query", query_id)
            continue
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