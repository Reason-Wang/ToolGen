import json
import click
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastchat.conversation import get_conv_template
import torch
import numpy as np
from sklearn import metrics
from unidecode import unidecode

# Build token api document
with open("data/toolenv/tools"+"/"+"tools.json", 'r', encoding="utf-8") as f:
    all_tools = json.load(f)
# print(len(all_tools))
# print(all_tools[0])

toolbench_virtual_token_to_document = {}
for tool in all_tools:
    if 'name' in tool:
        tool_name = tool['name']
    elif 'tool_name' in tool:
        tool_name = tool['tool_name']
    else:
        raise RuntimeError
    for api in tool['api_list']:
        token = f"<<{tool_name}&&{api['name']}>>"
        api_description = api['description']
        if api_description is None:
            api_description = "None."
        else:
            api_description = api_description.strip()
            if api_description == "":
                api_description = "None."
        toolbench_virtual_token_to_document[unidecode(token)] = {
            "name": unidecode(token),
            "description": api_description,
            "required": api['required_parameters'],
            "optional": api['optional_parameters']
        }

# document for finish action
toolbench_virtual_token_to_document["<<Finish>>"] = {
    "required": [],
    "optional": [
        {
            "name": "give_answer",
            "description": "Output the answer",
            "type": "string"
        },
        {
            "name": "give_up_and_restart",
            "description": "Unable to handle the task from this step",
            "type": "string"
        }
    ]
}
print(len(toolbench_virtual_token_to_document))


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
        if pred_action in label_actions:
            true_relevance[action_vocab[pred_action]] = 1
        # for label_action in label_actions:
        #     if label_action in action_vocab:
        #         true_relevance[action_vocab[label_action]] = 1
        #     else:
        #         print(label_action)

    return metrics.ndcg_score([true_relevance], [predicted_scores], k=k)

'''
python -m evaluation.eval_vagent \
    --model_name_or_path ../models/VAgent/llama-3-8b-tool-retrieval-G1 \
    --test_file data/retrieval/retrieval_test_G1_category_chat.json \
    --result_file data/results/retrieval/vagent_retrieval_G1_category.json
'''
@click.command()
@click.option('--model_name_or_path', type=str, help="The model's name or path.")
@click.option('--test_file', type=str, help="The test file.")
@click.option('--result_file', type=str, help="The result file.")
def main(
    model_name_or_path: str,
    test_file: str,
    result_file: str
):
    device = "cuda:0"
    with open(test_file, "r") as f:
        test_data = json.load(f)
        print(len(test_data))
    with open('data/virtual_tokens.txt', 'r', encoding='utf-8') as f:
        virtual_tokens = f.readlines()
        virtual_tokens = [unidecode(vt.strip()) for vt in virtual_tokens]
    action_vocab = {action: i for i, action in enumerate(virtual_tokens)}
    action_space = len(virtual_tokens)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.add_tokens(new_tokens=virtual_tokens, special_tokens=False)
    # print(f"<<👋 Onboarding Project&&Get Categories>> in vocab: {"<<👋 Onboarding Project&&Get Categories>>" in tokenizer.vocab}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    label_actions = []
    pred_actions = []
    pred_scores = []
    for example in test_data:
        query = example["conversations"][0]['content']
        virtual_tokens = example['conversations'][1]['content']

        conv = get_conv_template("llama-3")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        logits = model(**inputs).logits[0][-1]
        # print(logits.shape)
        topk = torch.topk(logits, 100)
        token_ids = topk.indices.tolist()
        token_values = topk.values.tolist()
        # print(token_ids)
        # print(token_values)
        actions = []
        for token_id in token_ids:
            action = tokenizer.decode(token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            actions.append(action)
        # virtual_token_ids = []
        # for vt in virtual_tokens:
        #     virtual_token_id = tokenizer.encode(vt, add_special_tokens=False)
        #     virtual_token_ids.append(virtual_token_id[0])
        
        pred_scores.append(token_values)
        pred_actions.append(actions)
        label_actions.append(virtual_tokens)

    result_scores = {}
    ndcg_scores = []
    for k in [1, 3, 5]:
        for label_action, pred_action, pred_score in zip(label_actions, pred_actions, pred_scores):
            ndcg_score = compute_ndcg(
                action_vocab, 
                label_actions = label_action,
                pred_actions = pred_action,
                pred_scores = pred_score,
                k=k)
            ndcg_scores.append(ndcg_score)
        result_scores[f"ndcg@{k}"] = np.mean(ndcg_scores)
        print(np.mean(ndcg_scores))
        ndcg_scores.clear()
    
    results = {
        "model": model_name_or_path,
        "ndcg": result_scores,
        "logs": []
    }
    for i in range(len(label_actions)):
        label_logs = []
        for label_action in label_actions[i]:
            if label_action in toolbench_virtual_token_to_document:
                label_logs.append({
                    "label": label_action,
                    "doc": toolbench_virtual_token_to_document[label_action]}
                )
            else:
                print("Un documented action: ", label_action)
                label_logs.append({
                    "label": label_action,
                    "doc": None
                })
        pred_logs = []
        for pred_action in pred_actions[i]:
            if pred_action in toolbench_virtual_token_to_document:
                pred_logs.append({
                    "pred": pred_action,
                    "doc": toolbench_virtual_token_to_document[pred_action]}
                )
            else:
                print("Invalid action: ", pred_action)
                pred_logs.append({
                    "pred": pred_action,
                    "doc": None
                })
        results["logs"].append({
            "query_id": test_data[i]['query_id'],
            "query": test_data[i]['conversations'][0]['content'],
            "relevant_actions": label_logs,
            "pred_actions": pred_logs
        })
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    

if __name__ == "__main__":
    main()