
from transformers import Constraint, StoppingCriteria, LogitsProcessor
import torch
from typing import List

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words:list, tokenizer, device):
        self.keywords = [torch.LongTensor(tokenizer.encode(w, add_special_tokens=False)[-5:]).to(device) for w in stop_words]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for k in self.keywords:
            if len(input_ids[0]) > len(k) and torch.equal(input_ids[0][-len(k):], k):
                return True
        return False


class IdsStoppingCriteria(StoppingCriteria):
    '''
    Stop when the model generates a specific sequence of token ids.
    '''
    def __init__(self, stop_ids: List[int], tokenizer, device):
        self.stop_ids = torch.LongTensor(stop_ids).to(device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(input_ids[0]) > len(self.stop_ids) and torch.equal(input_ids[0][-len(self.stop_ids):], self.stop_ids):
            return True
        return False


class TextStoppingCriteria(StoppingCriteria):
    '''
    Stop when the model generates a specific text. The most direct and also most expensive way to stop the generation.
    For k2, we need to check if there is "</s>" in generated text.
    '''
    def __init__(self, stop_text: List[str], tokenizer, device):
        self.stop_text = stop_text
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for text in self.stop_text:
            if text in generated_text:
                return True
        return False


    
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


class DisjunctiveConstraint(Constraint):
    r"""
    A special [`Constraint`] that is fulfilled by fulfilling just one of several constraints.

    Args:
        nested_token_ids (`List[List[int]]`):
            A list of words, where each word is a list of ids. This constraint is fulfilled by generating just one from
            the list of words.
    """

    def __init__(self, nested_token_ids: List[List[int]]):
        super(Constraint, self).__init__()

        if not isinstance(nested_token_ids, list) or len(nested_token_ids) == 0:
            raise ValueError(f"`nested_token_ids` has to be a non-empty list, but is {nested_token_ids}.")
        if any(not isinstance(token_ids, list) for token_ids in nested_token_ids):
            raise ValueError(f"`nested_token_ids` has to be a list of lists, but is {nested_token_ids}.")
        if any(
            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
            for token_ids in nested_token_ids
        ):
            raise ValueError(
                f"Each list in `nested_token_ids` has to be a list of positive integers, but is {nested_token_ids}."
            )

        self.trie = DisjunctiveTrie(nested_token_ids)
        self.token_ids = nested_token_ids

        self.seqlen = self.trie.max_height
        self.current_seq = []
        self.completed = False

    def advance(self):
        token_list = self.trie.next_tokens(self.current_seq)

        if len(token_list) == 0:
            return None
        else:
            return token_list

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        next_tokens = self.trie.next_tokens(self.current_seq)

        return token_id in next_tokens

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.current_seq.append(token_id)
            stepped = True
        else:
            reset = True
            self.reset()

        completed = self.trie.reached_leaf(self.current_seq)
        self.completed = completed

        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.current_seq = []

    def remaining(self):
        if self.completed:
            # since this can be completed without reaching max height
            return 0
        else:
            return self.seqlen - len(self.current_seq)

    def copy(self, stateful=False):
        new_constraint = DisjunctiveConstraint(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.current_seq = self.current_seq
            new_constraint.completed = self.completed

        return new_constraint


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