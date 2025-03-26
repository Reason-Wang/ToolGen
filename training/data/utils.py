import json
import random

import pandas as pd


def read_jsonl_to_list(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list


def load_instruction_data(datasets, nums):
    instructions = []
    responses = []
    for (d, n) in zip(datasets, nums):
        data_path = f'{d}'
        with open(data_path, 'r') as f:
            data_list = json.load(f)[:n]

        for sample in data_list:
            instruction = sample['instruction']
            if 'input' in sample:
                instruction = instruction + ' ' + sample['input']
                instruction = instruction.strip()
            response = sample['output']

            instructions.append(instruction)
            responses.append(response)

    return instructions, responses


def load_chat_data(datasets, nums):
    assert len(datasets) == len(nums)
    messages_list = []
    for (d, n) in zip(datasets, nums):
        data_path = f'{d}'
        with open(data_path, 'r') as f:
            data_list = json.load(f)
        if n <= len(data_list):
            # randomly sample n conversations
            data_list = random.sample(data_list, n)
        messages_list.extend([data['conversations'] for data in data_list])

    return messages_list
