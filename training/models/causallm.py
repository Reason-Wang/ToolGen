from typing import List
from prompts.conversations import get_conv_template
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import huggingface_hub
import torch
from prompts.utils import k2_tokenize
from models.utils import KeywordsStoppingCriteria, TextStoppingCriteria


class ChatCausalLM:
    def __init__(
        self,
        model_name,
        max_new_tokens=512,
        temperature=0.7,
        device="auto",
        system_prompt=None,
        cache_dir=None,
        conversation_template=None,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = "cuda" if device=="auto" else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=cache_dir
        )

        self.system_prompt = system_prompt
        self.conversation_history = []
        self.conversation_template = conversation_template

    def generate(self, messages, stop=None, print_prompt=False):
        human_role_set = {"user", "human"}
        ai_role_set = {"bot", "ai", "gpt", "assistant"}
        conv = get_conv_template(self.conversation_template)
        for message in messages:
            if message['role'] == 'system':
                conv.set_system_message(message['content'])
            else:
                conv.append_message(
                    conv.roles[0] if message['role'] in human_role_set else conv.roles[1],
                    message["content"]
                )
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if print_prompt:
            print(prompt)
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.conversation_template == 'k2':
            inputs = k2_tokenize(self.tokenizer, prompt, return_tensors="pt")
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        if self.conversation_template == 'k2':
            stop_criteria = StoppingCriteriaList([TextStoppingCriteria(stop, self.tokenizer, self.device)]) if stop else None
        else:
            stop_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop, self.tokenizer, self.device)]) if stop else None

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            stopping_criteria=stop_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        inputs_token_length = len(inputs['input_ids'][0])
        new_tokens = outputs[0][inputs_token_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop:
            for ending in stop:
                if text.endswith(ending):
                    text = text[:-len(ending)]
                    break

        return text.strip()

    def chat(self, text, stop=None, print_prompt=False):

        self.conversation_history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        messages.extend(self.conversation_history)
        response = self.generate(messages, stop=stop, print_prompt=print_prompt)
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        self.conversation_history = []

