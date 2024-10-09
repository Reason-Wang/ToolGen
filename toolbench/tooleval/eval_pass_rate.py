from toolbench.tooleval.evaluators import load_registered_automatic_evaluator
import os
import json
import csv
from toolbench.tooleval.evaluators.registered_cls.rtl import AnswerStatus, TaskStatus, AnswerPass
import random
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from tqdm import tqdm
import numpy as np
from toolbench.tooleval.utils import test_sets, get_steps
import backoff

abs_dir = os.path.split(__file__)[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--converted_answer_path', type=str, default="", required=True, help='converted answer path')
    parser.add_argument('--save_path', type=str, default="", required=False, help='result save path')
    parser.add_argument('--reference_model', type=str, default="", required=False, help='model predictions path')
    parser.add_argument('--test_ids', type=str, default="", required=True, help='model predictions path')
    parser.add_argument('--evaluator', type=str, default="tooleval_gpt-3.5-turbo_default", required=False, help='which evaluator to use.')
    parser.add_argument('--max_eval_threads', type=int, default=30, required=False, help='max threads nums')
    parser.add_argument('--evaluate_times', type=int, default=4, required=False, help='how many times to predict with the evaluator for each solution path.')
    parser.add_argument('--test_set', nargs='+', default=['G1_instruction'], help='test set name')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite the existing result file')
    return parser.parse_args()

def write_results(filename: str, reference_model: str, label_cnt: dict) -> None:
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["query", "available_tools", "model_intermediate_steps", "model_final_step", "model", "query_id", "is_solved", ])
        for query_id in label_cnt:
            query = label_cnt[query_id]["query"]
            tool_names = label_cnt[query_id]["tool_names"]
            answer_steps = label_cnt[query_id]["answer_steps"]
            final_step = label_cnt[query_id]["final_step"]
            is_solved = label_cnt[query_id]["is_solved"]
            writer.writerow([query, tool_names, answer_steps, final_step, reference_model, query_id, is_solved,])
            

if __name__ == "__main__":
    args = parse_args()
    evaluators = [load_registered_automatic_evaluator(evaluator_name=args.evaluator, evaluators_cfg_path=os.path.join(abs_dir,'evaluators')) for _ in range(args.max_eval_threads)]
    print(evaluators)
    @backoff.on_exception(backoff.expo, Exception, max_time=15)
    def compute_pass_rate(query_id, example, evaluate_time):
        global evaluators
        evaluator = random.choice(evaluators)
        # print("Example:", example)
        # raise Exception("Test")
        answer_steps, final_step = get_steps(example)
        
        if "'name': 'Finish'" not in final_step:
            # print("Final step is not Finish.")
            return query_id, AnswerStatus.Unsolved, evaluate_time
        
        is_solved, is_solved_reason = evaluator.check_is_solved(
            {
                'query':example['query'],
                'available_tools':example['available_tools'],
            },
            example['answer'],
            return_reason=True
        )
        # print(is_solved, is_solved_reason)

        # return query_id, task_solvable, is_solved, label, reason, not_hallucinate, evaluate_time
        return query_id, is_solved, evaluate_time
        
    reference_model = args.reference_model
    output_list = []
    for test_set in args.test_set:
        reference_path = f"{args.converted_answer_path}/{reference_model}/{test_set}.json"
        test_ids = list(json.load(open(os.path.join(args.test_ids, test_set+".json"), "r")).keys())
        reference_examples = json.load(open(reference_path, "r"))
        if os.path.exists(f"{args.save_path}/{test_set}_{reference_model}.json") and not args.overwrite:
            existed_ids = list(json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r")).keys())
            label_cnt = json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r"))
        else:
            existed_ids = []
            label_cnt = {}
        
        with ThreadPoolExecutor(args.max_eval_threads) as pool:
            future = []
            for query_id in reference_examples:
                if str(query_id) not in test_ids:
                    continue
                if query_id in existed_ids:
                    continue
                for i in range(args.evaluate_times):
                    example = reference_examples[query_id]
                    # print(example)
                    future.append(pool.submit(
                        compute_pass_rate,
                        query_id,
                        example,
                        evaluate_time=i
                    ))

            for thd in tqdm(as_completed(future),total=len(future),ncols=100):
                query_id, is_solved, evaluate_time = thd.result()
                example = reference_examples[query_id]
                query = example["query"]
                tool_names = []
                if isinstance(example["available_tools"], dict):
                    # print(example["available_tools"])
                    for _, tool_dict in example["available_tools"].items():
                        if 'function' in tool_dict:
                            tool_name = tool_dict['function']["name"]
                            tool_names.append(tool_name)
                        elif 'name' in tool_dict:
                            tool_name = tool_dict['name']
                            tool_names.append(tool_name)
                            # print(tool_name)
                        else:
                            raise ValueError("Available_tools should have a key 'name' or 'function'.")
                elif isinstance(example["available_tools"], list):
                    for tool_dict in example["available_tools"]:
                        tool_name = tool_dict["name"]
                        tool_names.append(tool_name)
                else:
                    raise ValueError("Available_tools should be a dict or a list.")
                # print("Example:", example)
                answer_steps, final_step = get_steps(example)
                if query_id not in label_cnt:
                    label_cnt[query_id] = {}
                label_cnt[query_id]["query"] = query
                label_cnt[query_id]["tool_names"] = tool_names
                label_cnt[query_id]["answer_steps"] = answer_steps
                label_cnt[query_id]["final_step"] = final_step
                if 'is_solved' not in label_cnt[query_id]:
                    label_cnt[query_id]["is_solved"] = {}
                label_cnt[query_id]["is_solved"][evaluate_time] = str(is_solved)
                json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)
        json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)
        
        filename = f"{args.save_path}/{test_set}_{reference_model}.csv"
        write_results(filename, reference_model, label_cnt)
        scores = []
        for runtime in range(args.evaluate_times):
            score = 0 
            for query_id in label_cnt:
                solved_dict = {**label_cnt[query_id]['is_solved']}
                solved_dict = {int(k):v for k,v in solved_dict.items()}
                if runtime not in solved_dict:
                    score += 0
                    continue
                if solved_dict[runtime] == "AnswerStatus.Solved":
                    score += 1
                elif solved_dict[runtime] == "AnswerStatus.Unsure":
                    score += 0.5

            
            scores.append(score / len(label_cnt))
    

        
        print(f"Test set: {test_set}. Model: {reference_model}.")
        solve_rate = sum(scores) / len(scores) * 100
        std_dev = np.std(scores).item() * 100
        print(f"Solve rate: {solve_rate:.2f}% Std: {std_dev:.2f}%")
        results = {
            "solve_rate": solve_rate,
            "std_dev": std_dev,
        }

        filename = f"{args.save_path}/{test_set}_{reference_model}_results.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        
