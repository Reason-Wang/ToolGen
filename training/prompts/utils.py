import torch
import transformers
from prompts.conversations import Conversation, get_conv_template

def k2_tokenize(tokenizer, text, add_special_tokens=True, return_tensors=None):
    """
    Tokenize the prompt and return the input_ids and attention_mask
    To make </s> tokenized correctly, we split the text by "</s>" and tokenize each part separately.
    :param tokenizer:
    :param prompt:
    :param max_length:
    :return: input_ids, attention_mask
    """
    if add_special_tokens:
        input_ids = [tokenizer.bos_token_id]
        attention_mask = [1]
    else:
        input_ids = []
        attention_mask = []

    splited_texts = text.split("</s>")
    inputs = tokenizer(splited_texts[0], add_special_tokens=False)
    input_ids.extend(inputs['input_ids'])
    attention_mask.extend(inputs['attention_mask'])
    if len(splited_texts) > 1:
        for text in splited_texts[1:]:
            current_inputs = tokenizer(text, add_special_tokens=False)
            input_ids += [tokenizer.eos_token_id] + current_inputs['input_ids']
            attention_mask += [1] + current_inputs['attention_mask']
    if return_tensors == 'pt':
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.tensor([attention_mask])

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask
    )


def format_conversation(messages, conv_template):
    # When there is no 'loss', we set it to False
    for message in messages:
        if 'loss' not in message:
            message['loss'] = False

    human_role_set = {"human", "user"}
    ai_role_set = {"ai", "gpt", "assistant"}
    conv = get_conv_template(conv_template)
    if 'from' in messages[0]:
        role_label, content_label = "from", "value"
    elif 'role' in messages[0]:
        role_label, content_label = "role", "content"
    else:
        raise ValueError("Cannot find role label and content label in the data.")

    for message in messages:
        if message[role_label] == 'system':
            conv.set_system_message(message[content_label])
        else:
            conv.append_message(conv.roles[0] if message[role_label] in human_role_set else conv.roles[1], message[content_label], message['loss'])

    # conv.append_message(conv.roles[1], None)
    return conv

    
def tokenize_conversation(
    messages,
    tokenizer,
    conv_template,
    max_length,
):
    """
    We want to tokenize the whole conversation. But we can't just simply
    use get_prompt to get string prompt and tokenize it. Because the loss
    can only be computed on model's response. We want:
        input_ids
        attention_mask
        labels: should be -100 for user prompt and input id for model's response
        action_mask: should be 0 for user prompt and 1 for model's response
    :param messages:
    :param tokenizer:
    :param conv_template:
    :param max_length:
    :return: input_ids, attention_mask, labels, action_mask
    """
    conv = format_conversation(messages, conv_template)
    separate_prompts = conv.get_separate_prompt_with_to_loss()
    # print(separate_prompts)
    input_ids = []
    attention_mask = []
    labels = []
    action_mask = []
    for i, (prompt, to_loss) in enumerate(separate_prompts):
        if i == 0:
            if tokenizer.bos_token:
                prompt = tokenizer.bos_token + prompt

        if conv_template == 'k2':
            tmp_input_ids = k2_tokenize(tokenizer, prompt, add_special_tokens=False)['input_ids']
        else:
            tmp_input_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        if to_loss:
            tmp_target = tmp_input_ids.copy()
            tmp_action_mask = [1] * len(tmp_input_ids)
        else:
            tmp_target = [-100] * len(tmp_input_ids)
            tmp_action_mask = [0] * len(tmp_input_ids)
        # print(tmp_input_ids)
        input_ids.extend(tmp_input_ids)
        attention_mask.extend([1] * len(tmp_input_ids))
        labels.extend(tmp_target)
        action_mask.extend(tmp_action_mask)

    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]
    action_mask = action_mask[:max_length]

    # TODO: remove this check if everything is correct
    assert len(input_ids) == len(attention_mask) == len(labels) == len(action_mask)

    return dict(
        input_ids=torch.tensor([input_ids]),
        attention_mask=torch.tensor([attention_mask]),
        labels=torch.tensor([labels]),
        # action_mask=torch.tensor([action_mask])
    )

