#!/usr/bin/env python
# coding=utf-8
import time
from termcolor import colored
from typing import Optional, List
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from fastchat.conversation import get_conv_template
from evaluation.toolbench.utils import process_system_message
from evaluation.toolbench.model.model_adapter import get_conversation_template
from evaluation.toolbench.inference.utils import SimpleChatIO, generate_stream, react_parser
from copy import deepcopy


SystemWOTool = '''You are an AutoGPT, capable of utilizing numerous tools and functions to complete the given task.
1.First, I will provide you with the task description, and your task will commence.
2.At each step, you need to analyze the current status and determine the next course of action by executing a function call.
3.Following the call, you will receive the result, transitioning you to a new state. Subsequently, you will analyze your current status, make decisions about the next steps, and repeat this process.
4.After several iterations of thought and function calls, you will ultimately complete the task and provide your final answer.

Remember:
1.The state changes are irreversible, and you cannot return to a previous state.
2.Keep your thoughts concise, limiting them to a maximum of five sentences.
3.You can make multiple attempts. If you plan to try different conditions continuously, perform one condition per try.
4.If you believe you have gathered enough information, call the function "Finish: give_answer" to provide your answer for the task.
5.If you feel unable to handle the task from this step, call the function "Finish: give_up_and_restart".
Let's Begin!
Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call "Finish" function at the end of the task. And the final answer should contain enough information to show to the user. If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.'''


class ToolChatModel:
    def __init__(
            self, 
            model_name_or_path: str, 
            template:str="llama-3", 
            device: str="cuda", 
            cpu_offloading: bool=False,
            max_sequence_length: int=8192
        ) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.template = template
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            use_fast=False,
            model_max_length=self.max_sequence_length
        )
        # Only support llama-3 currently
        if self.template == "llama-3":
            self.tokenizer.eos_token = "<|eot_id|>"
            self.tokenizer.eos_token_id = 128009
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        # if self.tokenizer.pad_token_id == None:
        #     self.tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        #     self.model.resize_token_embeddings(len(self.tokenizer))
        self.use_gpu = (True if device == "cuda" else False)
        if (device == "cuda" and not cpu_offloading) or device == "mps":
            self.model.to(device)
        self.chatio = SimpleChatIO()

        self.logs = {
            "retry_times": {
                "actions": 0,
                "give_up": 0,
                "sorry": 0,
            },
            "actions": {
            }
        }

    # def prediction(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    #     with torch.no_grad():
    #         gen_params = {
    #             "model": "",
    #             "prompt": prompt,
    #             "temperature": 0.5,
    #             "max_new_tokens": 512,
    #             "stop": "</s>",
    #             "stop_token_ids": None,
    #             "echo": False
    #         }
    #         generate_stream_func = generate_stream
    #         output_stream = generate_stream_func(self.model, self.tokenizer, gen_params, "cuda", self.max_sequence_length, force_generate=True)
    #         outputs = self.chatio.return_output(output_stream)
    #         prediction = outputs.strip()
    #     return prediction
    def initialize(self):
        self.logs["actions"] = {}
        print("Agent initialized.")
        
    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*"*50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*"*50)

    def parse(
            self,
            tools: List,
            process_id,
            add_tools=True,
            do_retry=False,
            retry_action=False,
            retry_give_up=True,
            retry_sorry=True,
            print_prompt=False,
            **args
        ):
        # conv = get_conversation_template(self.template)
        # if self.template == "tool-llama":
        #     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        # elif self.template == "tool-llama-single-round" or self.template == "tool-llama-multi-rounds":
        #     roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "assistant": conv.roles[3]}
        
        functions = [tool['function'] for tool in tools.values()]
        conversation_history = deepcopy(self.conversation_history)
        conversation_gpt = []
        for turn in conversation_history:
            if turn["role"] == "system":
                conversation_gpt.append({"role": "system", "content": turn['content']})
            elif turn['role'] == 'user':
                conversation_gpt.append({"role": "user", "content": turn['content']})
            elif turn['role'] == 'assistant':
                # react_format = f"Thought: {turn['content']}\nAction: {turn['function_call']['name']}\nAction Input: {turn['function_call']['arguments']}"
                react_format = f"Thought: {turn['content']}\nAction: {turn['tool_calls'][0]['function']['name']}\nAction Input: {turn['tool_calls'][0]['function']['arguments']}"
                conversation_gpt.append({"role": "assistant", "content": react_format})
            # elif turn['role'] == 'function':
            elif turn['role'] == 'tool':
                conversation_gpt.append({"role": "tool", "content": turn['content']})
            else:
                raise ValueError(f"Unknown role: {turn['role']}")


        conv = get_conv_template(self.template)
        roles = {"user": conv.roles[0], "function": conv.roles[0], "tool": conv.roles[0], "assistant": conv.roles[1]}
        self.time = time.time()
        for turn in conversation_gpt:
            if turn["role"] == "system":
                if add_tools:
                    content = process_system_message(turn["content"], functions)
                    # content = process_system_message(turn["content"], functions) + " Answer **Correctly**"
                    conv.set_system_message(content)
                else:
                    conv.set_system_message(SystemWOTool)
            else:
                role = roles[turn["role"]]
                content = turn["content"]
                conv.append_message(role, content)
        conv.append_message(roles['assistant'], None)

        # print(conversation_gpt)
        if do_retry:
            max_retry_times = 5
        else:
            max_retry_times = 1
        retry_temperature = {
            "0": 0.0,
            "1": 0.6,
            "2": 1.0,
            "3": 1.0,
            "4": 1.0,
        }
        for retry in range(max_retry_times):
            print("Retry times: ", retry)
            temperature = retry_temperature[str(retry)]
            conv_try = conv.copy()
            
            prompt = conv_try.get_prompt()
            if print_prompt:
                print(prompt)
            # if functions != []:
            #     predictions = self.prediction(prompt)
            # else:
            #     predictions = self.prediction(prompt)
            # predictions = self.prediction(prompt)
            inputs = self.tokenizer(prompt, return_tensors='pt')
            for k, v in inputs.items():
                inputs[k] = v.to("cuda")
            temperature = 0.0
            do_sample = False
            outputs = self.model.generate(
                **inputs,
                do_sample=do_sample if temperature > 0.0 else False,
                temperature=temperature,
                max_new_tokens=512,
                eos_token_id=self.tokenizer.eos_token_id
            )
            input_length = inputs["input_ids"].shape[1]
            # Use -1 to avoid eos_token
            predictions = self.tokenizer.decode(outputs[0][input_length:-1])
            # print(f"Predictions: {predictions}")
            # inputs = self.tokenizer(predictions)
            # print(f"Inputs: {inputs}")
            decoded_token_len = len(self.tokenizer(predictions)['input_ids'])
            if process_id == 0:
                print(f"[process({process_id})]total tokens: {decoded_token_len}")
            thought, action, arguments = react_parser(predictions)

            # if action in self.token_to_toolbench_name:
            #     toolbench_name = self.token_to_toolbench_name[action]
            toolbench_name = action
            if toolbench_name != "Finish":
                if toolbench_name not in self.logs['actions']:
                    self.logs['actions'][toolbench_name] = [arguments.strip()]
                else:
                    if arguments not in self.logs['actions'][toolbench_name]:
                        self.logs['actions'][toolbench_name].append(arguments.strip())
                    else:
                        if retry_action:
                            print(f"Action {toolbench_name} with arguments {arguments.strip()} has been used before.")
                            self.logs['retry_times']['actions'] += 1
                            continue
            
            if toolbench_name == "Finish":
                if "give_up_and_restart" in arguments:
                    if retry_give_up:
                        self.logs['retry_times']['give_up'] += 1
                        print("Generated give up, will retry.")
                        continue
                if ("I'm sorry" in arguments) or ("apologize" in arguments):
                    if retry_sorry:
                        self.logs['retry_times']['sorry'] += 1
                        print("Generated sorry, will retry.")
                        continue
            break
        
        print(f"Retry times: {self.logs['retry_times']}")
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": None,
            "tool_calls": [{
                "id": None,
                "type": "function",
                "function": {
                    "name": action,
                    "arguments": arguments
                }
            }]
        }
        return message, 0, decoded_token_len


if __name__ == "__main__":
    # can accept all huggingface LlamaModel family
    llm = ToolChatModel("decapoda-research/llama-7b-hf")
    messages = [
        {'role': 'system', 'content': '''You are AutoGPT, you can use many tools(functions) to do
the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is , you can\'t go
back to the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\nLet\'s Begin!\nTask description: Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number=24. Each
step, you are only allowed to choose two of the left numbers to obtain a new number. For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.\nRemember:\n1.all of the number must be used , and must be used ONCE. So Only when left numbers is exact 24, you will win. So you don\'t succeed when left number = [24, 5]. You succeed when left number = [24]. \n2.all the try takes exactly 3 steps, look
at the input format'''}, 
{'role': 'user', 'content': '\nThe real task input is: [1, 2, 4, 7]\nBegin!\n'}
]
    functions = [{'name': 'play_24', 'description': '''make your current conbine with the format "x operation y = z (left: aaa) " like "1+2=3, (left: 3 5 7)", then I will tell you whether you win. This is the ONLY way
to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1''','parameters':{'type': 'object', 'properties':{}}}]#, 'parameters': {'type': 'object', 'properties': {'input': {'type': 'string', 'description': 'describe what number you want to conbine, and how to conbine.'}}, 'required': ['input']}}]

    llm.change_messages(messages)
    output = llm.parse(functions=functions)
    print(output)