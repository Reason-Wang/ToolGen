
from data.dataset import CausalLMDataset, CausalLMCollator, CausalLMChatDataset, Seq2SeqDataset, Seq2SeqCollator
from data.utils import load_chat_data, load_instruction_data


def load_datasets(chat, architecture, datasets, dataset_nums, tokenizer, max_length, template):
    if chat:
        # assert args.architecture == 'causal' # Only causal is supported for chat
        messages_list = load_chat_data(
            datasets,
            dataset_nums,
        )
        dataset = CausalLMChatDataset(tokenizer, messages_list, max_length=max_length, template=template)
        collator = CausalLMCollator(tokenizer, max_length=max_length)
    else:
        instructions, responses = load_instruction_data(
            datasets,
            dataset_nums,
        )
        # TODO: Support better template system
        if architecture == 'causal':
            dataset = CausalLMDataset(
                tokenizer, 
                instructions, 
                responses,
                max_length=max_length,
                template=template
            )
            # Currently max_length is not used in the collator
            collator = CausalLMCollator(tokenizer, max_length=max_length)
        elif architecture == 'seq2seq':
            dataset = Seq2SeqDataset(
                tokenizer,
                instructions,
                responses,
                max_length=max_length,
                template=template
            )
            collator = Seq2SeqCollator(tokenizer, max_length=max_length)
        else:
            raise ValueError(f"Architecture {architecture} not supported")
        
    return dataset, collator