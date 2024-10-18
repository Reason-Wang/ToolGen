import json
import click
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from fastchat.conversation import get_conv_template
import torch
import numpy as np
from sklearn import metrics
from unidecode import unidecode
from evaluation.utils.utils import get_toolbench_name, AllowKeyWordsProcessor, DisjunctiveTrie
import pandas as pd
import os


def compute_ndcg(action_vocab, label_actions, pred_actions, pred_scores, k):
    action_space = len(action_vocab)
    true_relevance = np.zeros(action_space)
    predicted_scores = np.zeros(action_space)
    for pred_action, pred_score in zip(pred_actions, pred_scores):
        # We set all invalid actions to be zero score
        if pred_action in action_vocab:
            predicted_scores[action_vocab[pred_action]] = pred_score
        else:
            pass
        if pred_action in label_actions and pred_action in action_vocab:
            true_relevance[action_vocab[pred_action]] = 1
        # for label_action in label_actions:
        #     if label_action in action_vocab:
        #         true_relevance[action_vocab[label_action]] = 1
        #     else:
        #         print(label_action)

    return metrics.ndcg_score([true_relevance], [predicted_scores], k=k)


def constrained_beam_search(query, model, tokenizer, device, trie, constrain=True):
    conv = get_conv_template("llama-3")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(prompt)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    logits_processor = LogitsProcessorList([AllowKeyWordsProcessor(tokenizer, trie, inputs['input_ids'])])

    if constrain:
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=64,
            logits_processor=logits_processor,
            num_beams=5,
            num_return_sequences=5,
            eos_token_id=128009
        )
    else:
        print("Unconstrained beam search.")
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=64,
            num_beams=5,
            num_return_sequences=5,
            eos_token_id=128009
        )
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    return outputs

'''
python -m evaluation.eval_vagent \
    --model_name_or_path ../models/VAgent/llama-3-8b-tool-retrieval-G1 \
    --test_file data/retrieval/retrieval_test_G1_category_chat.json \
    --result_file data/results/retrieval/vagent_retrieval_G1_category.json
'''
@click.command()
@click.option('--model_name_or_path', type=str, help="The model's name or path.")
@click.option('--indexing', type=str, help="The indexing method.")
@click.option('--stage', type=str, help="The stage of the data.")
@click.option('--split', type=str, help="The split of the data.")
@click.option('--result_path', type=str, help="The result file.")
@click.option('--constrain', type=bool, help="Whether to constrain the beam search.")
def main(
    model_name_or_path: str,
    indexing: str,
    stage: str,
    split: str,
    result_path: str,
    constrain: bool
):
    ir_test_queries = {}
    ir_relevant_docs = {}
    def process_retrieval_ducoment(documents_df):
        ir_corpus = {}
        for row in documents_df.itertuples():
            doc = json.loads(row.document_content)
            tool_name = doc.get('tool_name', '')
            api_name = doc.get('api_name', '')
            toolbench_name = get_toolbench_name(tool_name, api_name)
            ir_corpus[row.docid] = toolbench_name
        
        return ir_corpus
        
    data_path = f"data/retrieval/{stage}"

    documents_df = pd.read_csv(os.path.join(data_path, 'corpus.tsv'), sep='\t')
    ir_corpus = process_retrieval_ducoment(documents_df)


    query_path = "test.query.txt" if split == "test" else f"test_{stage}_{split}.query.txt"
    test_queries_df = pd.read_csv(os.path.join(data_path, query_path), sep='\t', names=['qid', 'query'])
    for row in test_queries_df.itertuples():
        ir_test_queries[row.qid] = row.query

    rel_path = "qrels.test.tsv" if split == "test" else f"qrels.test_{stage}_{split}.tsv"
    labels_df = pd.read_csv(os.path.join(data_path, rel_path), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    for row in labels_df.itertuples():
        ir_relevant_docs.setdefault(row.qid, set()).add(row.docid)


    with open(f"data/toolenv/Tool2{indexing}Id.json", 'r') as f:
        Tool2Id = json.load(f)
    
    # Build token api document
    with open("data/toolenv/tools"+"/"+"tools.json", 'r', encoding="utf-8") as f:
        all_tools = json.load(f)

    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    if indexing == "Atomic":
        with open('data/virtual_tokens.txt', 'r') as f:
            virtual_tokens = f.readlines()
            virtual_tokens = [unidecode(vt.strip()) for vt in virtual_tokens]
        tokenizer.add_tokens(new_tokens=virtual_tokens, special_tokens=False)
    
    tokenizer.eos_token_id = 128009
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )



    candidates = list(Tool2Id.values())
    action_vocab = {action: i for i, action in enumerate(candidates)}
    token_ids = tokenizer(candidates, add_special_tokens=False).input_ids
    token_ids = [ids + [tokenizer.eos_token_id] for ids in token_ids]
    trie = DisjunctiveTrie(token_ids)

    model.eval()
    label_actions_list = []
    pred_actions_list = []
    # pred_scores = []
    
    queries = list(ir_test_queries.values())
    query_ids = list(ir_test_queries.keys())
    for query_id, query in tqdm(zip(query_ids, queries), total=len(queries)):

        outputs = constrained_beam_search(query, model, tokenizer, device, trie, constrain=constrain)
        pred_actions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_actions.extend(["" for _ in range(95)])
        pred_actions_list.append(pred_actions)

        relevant_doc_ids = ir_relevant_docs[query_id]
        label_actions = []
        for doc_id in relevant_doc_ids:
            try:
                label_actions.append(Tool2Id[ir_corpus[doc_id]])
            except KeyError:
                print(f"Unexisted tool name: {ir_corpus[doc_id]}")
                continue
        label_actions_list.append(label_actions)

    result_scores = {}
    ndcg_scores = []
    logs = []
    for k in [1, 3, 5]:
        for label_actions, pred_actions in zip(label_actions_list, pred_actions_list):
            # Each action is treated equally, so the actual value of actions does not matter
            pred_score = list(range(len(pred_actions), 0, -1))
            ndcg_score = compute_ndcg(
                action_vocab,
                label_actions = label_actions,
                pred_actions = pred_actions,
                pred_scores = pred_score,
                k=k)
            ndcg_scores.append(ndcg_score)
            logs.append({
                "label": label_actions,
                "pred": pred_actions,
            })
        result_scores[f"ndcg@{k}"] = np.mean(ndcg_scores)
        print(np.mean(ndcg_scores))
        ndcg_scores.clear()
    
    results = {
        "model": model_name_or_path,
        "ndcg": result_scores,
        "logs": logs
    }

    model_name = model_name_or_path.split("/")[-1]
    if constrain:
        model_name = model_name + "_constrained"
    result_file = result_path + f"{model_name}_{indexing}_{stage}_{split}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    

if __name__ == "__main__":
    main()