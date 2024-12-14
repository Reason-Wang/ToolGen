import time
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm
from evaluation.toolbench.utils import change_name, standardize
from transformers import LogitsProcessor
from typing import List
import torch

def get_toolbench_name(tool_name, api_name):
    tool_name = standardize(tool_name)
    api_name = change_name(standardize(api_name))
    toolbench_name = api_name+f"_for_{tool_name}"
    toolbench_name = toolbench_name[-64:]
    return toolbench_name

class DisjunctiveTrie:
    def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
        r"""
        A helper class that builds a trie with the words represented in `nested_token_ids`.
        """
        self.max_height = max([len(one) for one in nested_token_ids])

        root = {}
        for token_ids in nested_token_ids:
            level = root
            for tidx, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}

                level = level[token_id]

        if no_subsets and self.has_subsets(root, nested_token_ids):
            raise ValueError(
                "Each list in `nested_token_ids` can't be a complete subset of another list, but is"
                f" {nested_token_ids}."
            )

        self.trie = root

    def next_tokens(self, current_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = self.trie

        for current_token in current_seq:
            start = start[current_token]

        next_tokens = list(start.keys())

        return next_tokens

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])

    def has_subsets(self, trie, nested_token_ids):
        """
        Returns whether # of leaves == # of words. Otherwise some word is a subset of another.
        """
        leaf_count = self.count_leaves(trie)
        return len(nested_token_ids) != leaf_count


class AllowTokenIdsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, self.allowed_token_ids] = False
        scores = scores.masked_fill(mask, -1e10)

        return scores


class AllowKeyWordsProcessor(LogitsProcessor):
    ''' renxi.wang@mbzuai.ac.ae
    A logits processor that limit output text to be in a set of predefined keywords.
    tokenizer: tokenizer used to encode the keywords
    trie: DisjunctiveTrie of predefined keywords
    input_ids: input_ids of the prompt that the model is generating from
    return:
        scores: scores of the logits, where impossible tokens are masked
        For beam search, scores are log-softmax of logits, others are logits
    '''
    def __init__(self, tokenizer, trie, input_ids):
        self.tokenizer = tokenizer
        self.trie = trie
        self.input_ids = input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        input_length = self.input_ids.shape[1]
        generated_ids = input_ids[:, input_length:].tolist()
        new_token_ids = []
        for ids in generated_ids:
            try:
                next_token_ids = self.trie.next_tokens(ids)
            except KeyError as e:
                next_token_ids = [self.tokenizer.eos_token_id]
            if not next_token_ids:
                next_token_ids = [self.tokenizer.eos_token_id]
            new_token_ids.append(next_token_ids)
            
        for row, token_ids in enumerate(new_token_ids):
            mask = torch.ones_like(scores[row], dtype=torch.bool)
            mask[torch.tensor(token_ids)] = False
            scores[row, mask] = -1e10
        
        return scores


def openai_client_request(client, model, messages, num_retries: int = 5, return_dict: bool = True, **kwargs):
    print(f"Arguments: {kwargs}")
    response = {}
    # retry request (handles connection errors, timeouts, and overloaded API)
    for i in range(num_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            # response['success'] = True
            break
        except Exception as e:
            # response['success'] = False
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            time.sleep(10)
    if return_dict:
        return response
    else:
        return response.choices[0].message.content


class OpenAIChatModel:
    def __init__(self, model: str, api_key, api_base=None, api_version=None, azure_endpoint=None, temperature: float=None, stop: List[str]=None):
        self.model = model
        if api_base:
            self.client = OpenAI(api_key=api_key, api_base=api_base)
        else:
            self.client = OpenAI(api_key=api_key)
        self.temperature = temperature
        self.stop = stop

    def generate(self, messages: List, temperature: float = None, stop: List[str] = None, print_prompt=False):
        if print_prompt:
            print(messages)

        kwargs = {}
        if self.temperature:
            kwargs['temperature'] = self.temperature
        elif temperature:
            kwargs['temperature'] = temperature
        if self.stop:
            kwargs['stop'] = self.stop
        

        temperature=self.temperature if self.temperature else temperature,
        response = openai_client_request(
            client=self.client,
            model=self.model,
            messages=messages,
            return_dict=False,
            **kwargs
        )
        
        return response


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True