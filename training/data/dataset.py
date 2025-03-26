import torch
import transformers
from torch.utils.data import Dataset
from prompts.utils import tokenize_conversation, format_conversation
from utils.distributed import get_rank, is_main_process


class Seq2SeqDataset(Dataset):
    def __init__(self, tokenizer, sources, targets, max_length, template):
        super(Seq2SeqDataset, self).__init__()
        self.tokenizer = tokenizer
        self.sources = sources
        self.targets = targets
        self.max_length = max_length
        self.template = template
        self.has_print = False

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        instruction = self.template.format(instruction=self.sources[item], response="")
        if instruction.endswith(" "):
            instruction = instruction[:-1]
        response = self.targets[item]
        if not self.has_print:
            rank = get_rank()
            if rank == 0 or -1:
                print(f"Instruction: {instruction}")
                print(f"Response: {response}")
            self.has_print = True
        return instruction, response


class Seq2SeqCollator(object):
    def __init__(self, tokenizer, max_length, padding_side='right'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert padding_side == 'right', "Only right padding is supported for Seq2Seq models"

    def __call__(self, batch):
        sources = [ex[0] for ex in batch]
        targets = [ex[1] for ex in batch]

        inputs = self.tokenizer(
            sources,
            max_length=self.max_length,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        labels = self.tokenizer(
            targets,
            max_length=self.max_length,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels

        return inputs


class CausalLMDataset(Dataset):
    def __init__(self, tokenizer, sources, targets, max_length, template):
        super(CausalLMDataset, self).__init__()
        self.tokenizer = tokenizer
        self.sources = sources
        self.targets = targets
        self.max_length = max_length
        self.template = template
        # self.instruction_prompt = "Instruction: {instruction} Response: "
        # self.response_prompt = "{response}"
        self.has_print = False

    def _tokenize(self, text):
        return self.tokenizer(text, truncation=True, max_length=self.max_length)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        full_prompt = self.template['full_template'].format(instruction=self.sources[item], response=self.targets[item])
        user_prompt = self.template['user_template'].format(instruction=self.sources[item])
        # full_prompt = self.sources[item] + ' ' + self.targets[item]
        # user_prompt = self.sources[item]

        # set a prompt for inputs
        # full_prompt = self.instruction_prompt.format(instruction=self.sources[item]) + self.response_prompt.format(response=self.targets[item])
        # user_prompt = self.response_prompt.format(response=self.targets[item])

        if not self.has_print:
            rank = get_rank()
            if rank == 0 or -1:
                print(f"Full Prompt: {full_prompt}")
                print(f"User Prompt: {user_prompt}")
            self.has_print = True

        tokenized_full_prompt = self._tokenize(full_prompt)
        if (tokenized_full_prompt["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(tokenized_full_prompt["input_ids"]) < self.max_length):
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized_full_prompt["attention_mask"].append(1)

        tokenized_user_prompt = self._tokenize(user_prompt)["input_ids"]
        user_prompt_len = len(tokenized_user_prompt)
        labels = [-100 if i < user_prompt_len else token_id for i, token_id in enumerate(tokenized_full_prompt["input_ids"])]

        return torch.tensor(tokenized_full_prompt["input_ids"]), \
            torch.tensor(tokenized_full_prompt["attention_mask"]), \
            torch.tensor(labels)


class CausalLMChatDataset(Dataset):
    def __init__(self, tokenizer, messages_list, max_length, template):
        super(CausalLMChatDataset, self).__init__()
        self.tokenizer = tokenizer
        self.messages_list = messages_list
        self.max_length = max_length
        self.template = template
        self.has_print = False

    def __len__(self):
        return len(self.messages_list)

    def __getitem__(self, item):
        messages = self.messages_list[item]

        inputs = tokenize_conversation(
            messages,
            self.tokenizer,
            self.template,
            self.max_length
        )
        if is_main_process():
            if not self.has_print:
                conv = format_conversation(messages, self.template)
                print(conv.hightlight_with_to_loss())
                self.has_print = True

        return inputs['input_ids'].squeeze(dim=0), \
            inputs['attention_mask'].squeeze(dim=0), \
            inputs['labels'].squeeze(dim=0)


class CausalLMCollator(object):
    def __init__(self, tokenizer, max_length, padding_side='right'):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = tokenizer.eos_token_id
        self.max_length = max_length
        self.padding_side = padding_side

    def __call__(self, instances):
        input_ids = [e[0] for e in instances]
        attention_masks = [e[1] for e in instances]
        labels = [e[2] for e in instances]

        if self.padding_side == 'left':
            # pad all inputs from left side, this can help batch generation
            reversed_input_ids = [ids.flip(0) for ids in input_ids]
            reversed_attention_masks = [mask.flip(0) for mask in attention_masks]
            reversed_labels = [label.flip(0) for label in labels]

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_input_ids = padded_input_ids.flip(1)
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(reversed_attention_masks, batch_first=True, padding_value=0)
            padded_attention_masks = padded_attention_masks.flip(1)
            padded_labels = torch.nn.utils.rnn.pad_sequence(reversed_labels, batch_first=True, padding_value=-100)
            padded_labels = padded_labels.flip(1)
        elif self.padding_side == 'right':
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
            padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        else:
            raise RuntimeError("Padding side must 'left' or 'right'.")

        return {"input_ids": padded_input_ids, "attention_mask": padded_attention_masks, "labels": padded_labels}

    def _mask(self, lens, max_length):
        mask = torch.arange(max_length).expand(len(lens), max_length) < torch.tensor(lens).unsqueeze(1)
        return mask


if __name__ == '__main__':
    sources = ['List 5 reasons why learn to code.', 'what is it?']
    targets = ['Improve communication skills.', 'Not like a human.']
    tokenizer = transformers.AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-590M")
    dataset = CausalLMDataset(tokenizer, sources, targets, max_length=512)
    collator = CausalLMCollator(tokenizer, max_length=512)
    print(dataset[1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=4
    )
    for data in dataloader:
        print(data)

    instructions, responses = load_bias_data()
    dataset = CausalLMDataset(tokenizer, instructions, responses, max_length=512)
    print(dataset[1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=4
    )
    for data in dataloader:
        print(data)
        break



