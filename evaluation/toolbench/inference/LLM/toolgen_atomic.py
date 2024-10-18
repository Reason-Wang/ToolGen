#!/usr/bin/env python
# coding=utf-8
from copy import deepcopy
import json
import re
import time
from termcolor import colored
from typing import Optional, List
from evaluation.toolbench.utils import change_name, standardize
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    LogitsProcessor
)
# from toolbench.utils import process_system_message
from evaluation.toolbench.model.model_adapter import get_conversation_template
from evaluation.toolbench.inference.utils import SimpleChatIO, generate_stream, react_parser
from unidecode import unidecode
from fastchat.conversation import get_conv_template


class AllowTokenIdsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[:, self.allowed_token_ids] = False
        scores = scores.masked_fill(mask, -1e10)

        return scores
    

SystemPrompt = '''You are an AutoGPT, capable of utilizing numerous tools and functions to complete the given task.
1.First, I will provide you with the task description, and your task will commence.
2.At each step, you need to determine the next course of action by generating an action token.
3.Following the token, you will receive the documentation of the action corresponding to the token. You need to generate the input of the action, transitioning you to a new state. Subsequently, you will make decisions about the next steps, and repeat this process.
4.After several iterations of generating actions and inputs, you will ultimately complete the task and provide your final answer.

Remember:
1.The state changes are irreversible, and you cannot return to a previous state.
2.Keep your actions concise, limiting them to best suits the current query.
3.You can make multiple attempts. If you plan to try different conditions continuously, perform one condition per try.
4.If you believe you have gathered enough information, generate the action "<<Finish>> with argument give_answer" to provide your answer for the task.
5.If you feel unable to handle the task from this step, generate the action "<<Finish>> with argument give_up_and_restart".
Let's Begin!
Task description: You should use actions to help handle the real time user querys. Remember:
1.ALWAYS generate "<<Finish>>" at the end of the task. And the final answer should contain enough information to show to the user. If you can't handle the task, or you find that actions always fail(the function is not valid now), use action <<Finish>> with give_up_and_restart.
2.Only generate actions and inputs.'''

SystemPromptTokens = '''You are an AutoGPT, capable of utilizing numerous tools and functions to complete the given task.
    1.First, I will provide you with the task description, and your task will commence.
    2.At each step, you need to determine the next course of action by generating an action token.
    3.Following the token, you will receive the documentation of the action corresponding to the token. You need to generate the input of the action, transitioning you to a new state. Subsequently, you will make decisions about the next steps, and repeat this process.
    4.After several iterations of generating actions and inputs, you will ultimately complete the task and provide your final answer.
    
    Remember:
    1.The state changes are irreversible, and you cannot return to a previous state.
    2.Keep your actions concise, limiting them to best suits the current query.
    3.You can make multiple attempts. If you plan to try different conditions continuously, perform one condition per try.
    4.If you believe you have gathered enough information, generate the action "{finish} with argument give_answer" to provide your answer for the task.
    5.If you feel unable to handle the task from this step, generate the action "{finish} with argument give_up_and_restart".
    Let's Begin!
    Task description: You should use actions to help handle the real time user querys. Remember:
    1.ALWAYS generate "{finish}" at the end of the task. And the final answer should contain enough information to show to the user. If you can't handle the task, or you find that actions always fail(the function is not valid now), use action {finish} with give_up_and_restart.
    2.I may or may not provide your some actions that you need to use. If actions are provided, you must take one action from provided actions at each step. If actions are not provided, you must come up actions by yourself.
    
    Actions: {actions}'''

def load_virtual_tokenizer(model_name_or_path, cache_dir=None):
        print(f"Loading Llama-3 tokenizer with virtual tokens.")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", 
            cache_dir=cache_dir,
        )
        
        with open('data/virtual_tokens.txt', 'r') as f:
            virtual_tokens = f.readlines()
            virtual_tokens = [unidecode(vt.strip()) for vt in virtual_tokens]
        tokenizer.add_tokens(new_tokens=virtual_tokens, special_tokens=False)
        print(f"Added {len(virtual_tokens)} virtual tokens")
        return tokenizer


def load_tool_documentation():
    # Build token api document
    with open("data/toolenv/tools"+"/"+"tools.json", 'r') as f:
        all_tools = json.load(f)
    print(len(all_tools))
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
        "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
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
    toolbench_virtual_token_to_document["Invalid Action"] = "No documentation, try again."
    print(len(toolbench_virtual_token_to_document))
    return toolbench_virtual_token_to_document


def get_toolbench_name(tool_name, api_name):
    tool_name = standardize(tool_name)
    api_name = change_name(standardize(api_name))
    toolbench_name = api_name+f"_for_{tool_name}"
    toolbench_name = toolbench_name[-64:]
    return toolbench_name
    

def load_virtual_token_toobench_name_conversion():
    with open('data/virtual_tokens.txt', 'r') as f:
        virtual_tokens = f.readlines()
    virtual_tokens = [vt.strip() for vt in virtual_tokens]


    # def remove_symbol(name):
    #     # name = name.split()
    #     # Replace all special symbols with space
    #     name = re.sub('[^a-zA-Z0-9 \n\.]', ' ', name)
    #     # symbols = ['+', '|', '/', '-']
    #     # for s in symbols:
    #     #     if s in name:
    #     #         name.remove(s)
    #     return name


    toolbench_name_to_virtual_token_dict = {}
    virtual_token_to_toolbench_name_dict = {}
    for vt in virtual_tokens:
        if vt == "<<Finish>>":
            toolbench_name_to_virtual_token_dict["Finish"] = vt
            virtual_token_to_toolbench_name_dict[vt] = "Finish"
        else:
            names = vt[2:-2].split("&&")
            tool_name = names[0]
            api_name = names[1]
            # tool_name = remove_symbol(tool_name)
            # api_name = remove_symbol(api_name)
            # toolbench_name = "_".join(" ".join([api_name, "for", tool_name]).split())
            # toolbench_name_to_virtual_token_dict[toolbench_name] = vt
            toolbench_name = get_toolbench_name(tool_name, api_name)
            virtual_token_to_toolbench_name_dict[unidecode(vt)] = toolbench_name
            toolbench_name_to_virtual_token_dict[toolbench_name] = unidecode(vt)

    toolbench_name_to_virtual_token_dict["invalid_hallucination_function_name"] = "Invalid Action"
    return virtual_token_to_toolbench_name_dict, toolbench_name_to_virtual_token_dict

class ToolGenAtomic:
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
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, model_max_length=self.max_sequence_length)
        self.tokenizer = load_virtual_tokenizer(model_name_or_path)
        # Only support llama-3 currently
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

        self.tool_documentation = load_tool_documentation()
        self.token_to_toolbench_name, self.toolbench_name_to_token = load_virtual_token_toobench_name_conversion()
        
        self.retrieved_actions = None
        self.relevant_actions_documentations = None
        self.allowed_action_ids = [128009] + list(range(128256, 128256 + 46985))

        self.logs = {
            "retry_times": {
                "actions": 0,
                "give_up": 0,
                "sorry": 0,
            },
            "actions": {
            }
        }

    def generate(self, text, do_sample, temperature=0.6, restrict_actions=False, allowed_action_ids=None):
        inputs = self.tokenizer(text, return_tensors='pt')
        input_length = inputs["input_ids"].shape[1]
        for k, v in inputs.items():
            inputs[k] = v.to("cuda")
        if restrict_actions:
            if allowed_action_ids is None:
                logits_processor = LogitsProcessorList([
                        AllowTokenIdsProcessor(self.allowed_action_ids)
                    ])
            else:
                logits_processor = LogitsProcessorList([
                        AllowTokenIdsProcessor(allowed_action_ids)
                    ])
        else:
            logits_processor = None
        outputs = self.model.generate(
            **inputs, 
            do_sample=do_sample if temperature > 0.0 else False,
            max_new_tokens=512,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=logits_processor,
            temperature=temperature
        )
        output_ids = outputs[0][input_length:-1]
        generated_text = self.tokenizer.decode(output_ids)
        return {
            "output_ids": output_ids,
            "generated_text": generated_text
        }
        

    def initialize(self):
        self.retrieved_actions = None
        self.relevant_actions_documentations = None
        self.logs["actions"] = {}
        print("Agent initialized.")

    # def prediction(self, prompt: str, stop: Optional[List[str]] = None, do_sample=True) -> str:
    #         inputs = self.tokenizer(prompt, return_tensors='pt')
    #         for k, v in inputs.items():
    #             inputs[k] = v.to("cuda")
    #         outputs = self.model.generate(**inputs, do_sample=do_sample, max_new_tokens=512, eos_token_id=self.tokenizer.eos_token_id)
    #         input_length = inputs["input_ids"].shape[1]
    #         # Use -1 to avoid eos_token
            
    #         # print(f"Predictions: {predictions}")
            
    #         return 
            
        
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

    def get_relevant_actions(self, conversation_gpt):
        # System prompt, query, retrieval_hint
        conversation_for_retrieval = conversation_gpt[:3]
        conv = get_conv_template(self.template)
        roles = {"user": conv.roles[0], "function": conv.roles[0], "assistant": conv.roles[1]}
        self.time = time.time()
        for turn in conversation_for_retrieval:
            if turn["role"] == "system":
                # content = process_system_message(turn["content"], functions)
                content = SystemPrompt
                conv.set_system_message(content)
            else:
                role = roles[turn["role"]]
                content = turn["content"]
                conv.append_message(role, content)
        conv.append_message(roles['assistant'], None)
        prompt = conv.get_prompt()
        output_ids = self.generate(prompt, do_sample=True, restrict_actions=True)['output_ids']
        actions = self.tokenizer.convert_ids_to_tokens(output_ids)

        return actions

    def get_token_ids_by_actions(self, tools):
        ## Get relevant_actions ids
        virtual_tool_ids = []
        for tool in tools:
            if tool in self.toolbench_name_to_token:
                tool_id = self.tokenizer(self.toolbench_name_to_token[tool])['input_ids'][1]
            virtual_tool_ids.append(tool_id)

        # We also need to add the finish token and eos_token
        tool_id = self.tokenizer("<<Finish>>")['input_ids'][1]
        virtual_tool_ids.append(tool_id)
        # virtual_tool_ids.append(128009)
        print(f"virtual_tool_ids: {virtual_tool_ids}")
        return virtual_tool_ids
    
    def convert_conversation_history_to_gpt_format(
            self,
            tools,
            add_retrieval=False,
        ):
        conversation_history = deepcopy(self.conversation_history)
        conversation_gpt = []
        for i, turn in enumerate(conversation_history):
            if turn["role"] == "system":
                conversation_gpt.append({"role": "system", "content": turn['content']})
            elif turn['role'] == 'user':
                conversation_gpt.append({"role": "user", "content": turn['content']})
                if i == 1 and add_retrieval:
                    conversation_gpt.append({"role": "user", "content": "Please generate relevant actions."})
                    if not self.retrieved_actions:
                        # print(conversation_gpt)
                        ## We use ground truth actions

                        virtual_token_ids = self.get_token_ids_by_actions(tools)
                        self.relevant_actions = self.tokenizer.convert_ids_to_tokens(virtual_token_ids)
                        
                        print(f"Relevant_actions: {self.relevant_actions}")
                        ## We use the LLM to retrieve the relevant actions
                        # self.relevant_actions = self.get_relevant_actions(conversation_gpt)
                        self.relevant_actions_documentations = []
                        for vt in self.relevant_actions:
                            # print(self.tool_documentation[vt])
                            self.relevant_actions_documentations.append(
                                {
                                    "name": vt, 
                                    "description": self.tool_documentation[vt]['description']
                                }
                            )
                    conversation_gpt.append({"role": "assistant", "content": "".join(self.relevant_actions)})
                    doc_format = f"Here are descriptions: {json.dumps(self.relevant_actions_documentations)}"
                    conversation_gpt.append(
                        {"role": "user", "content": doc_format}
                    )
            
            elif turn['role'] == 'assistant':
                if turn['content'] is not None:
                    conversation_gpt.append({"role": "assistant", "content": turn['content']})
                    conversation_gpt.append({"role": "user", "content": "Please generate the action."})
                    # assert 'function_call' not in turn.keys()
                # else:
                # virtual_token = self.toolbench_name_to_token[turn['function_call']['name']]
                virtual_token = self.toolbench_name_to_token[turn['tool_calls'][0]['function']['name']]
                action_format = f"{virtual_token}"
                conversation_gpt.append({"role": "assistant", "content": action_format})
                documentation_format = f"Please give the input. Here is the documentation: {self.tool_documentation[virtual_token]}"
                conversation_gpt.append({"role": "user", "content": documentation_format})
                action_input_format = f"{turn['tool_calls'][0]['function']['arguments']}"
                conversation_gpt.append({"role": "assistant", "content": action_input_format})
            elif turn['role'] == 'function':
                conversation_gpt.append({"role": "function", "content": turn['content']})
            elif turn['role'] == 'tool':
                conversation_gpt.append({"role": "tool", "content": turn['content']})
            else:
                raise ValueError(f"Unknown role: {turn['role']}")
            
        return conversation_gpt
    
    def convert_to_fastchat_format(
            self,
            tools,
            conversation_gpt,
            add_relevant_tokens,
        ):
        
        self.time = time.time()

        conv = get_conv_template(self.template)
        roles = {"user": conv.roles[0], "function": conv.roles[0], "tool": conv.roles[0], "assistant": conv.roles[1]}

        for turn in conversation_gpt:
            if turn["role"] == "system":
                # content = process_system_message(turn["content"], functions)
                virtual_token_ids = self.get_token_ids_by_actions(tools)
                self.relevant_actions = self.tokenizer.convert_ids_to_tokens(virtual_token_ids)
                if add_relevant_tokens:
                    content = SystemPromptTokens.format(
                        finish="<<Finish>>",
                        actions=str(self.relevant_actions)
                    )
                else:
                    content = SystemPrompt
                conv.set_system_message(content)
            else:
                role = roles[turn["role"]]
                content = turn["content"]
                conv.append_message(role, content)
        return conv, roles

    
    def planning(self, tools, conv, roles, planning, add_planning_prefix=False, temperature=0.1):
        conv.append_message(roles['assistant'], None)
        prompt = conv.get_prompt()
        if add_planning_prefix:
            tool_tokens = []
            assert planning is True
            for tool in tools:
                if tool in self.toolbench_name_to_token:
                    virtual_token = self.toolbench_name_to_token[tool]
                    tool_tokens.append(virtual_token)
            # Add finish token
            tool_tokens.append("<<Finish>>")
            planning_prefix_format = "I am using one of the following tools: {tool_tokens}."
            planning_prefix = planning_prefix_format.format(tool_tokens=tool_tokens)
            prompt = prompt + planning_prefix

        if planning:
            thought = self.generate(prompt, do_sample=True, temperature=temperature, restrict_actions=False)['generated_text']
            if add_planning_prefix:
                thought = planning_prefix + " " + thought
            conv.messages[-1] = (roles['assistant'], thought)
        else:
            thought = None

        return thought, conv
    
    def acting(self, tools, conv, roles, restrict_to_ground_truth_tools, restrict_actions, temperature=0.1):
        conv.append_message(roles['user'], "Generate the action.")
        conv.append_message(roles['assistant'], None)
        prompt = conv.get_prompt()
        
        if restrict_to_ground_truth_tools:
            virtual_tool_ids = self.get_token_ids_by_actions(tools)
        else:
            virtual_tool_ids = None
        
        action = self.generate(
            prompt, 
            do_sample=True, 
            temperature=temperature,
            restrict_actions=restrict_actions,
            allowed_action_ids=virtual_tool_ids
        )['generated_text']
        conv.messages[-1] = (roles['assistant'], action)

        return action, conv

    def calling(self, action, roles, conv, process_id, print_prompt=False, temperature=0.1):
        virtual_token = action
        if virtual_token in self.tool_documentation:
            documentation_format = f"Please give the input. Here is the documentation: {self.tool_documentation[virtual_token]}"
            conv.append_message(roles['assistant'], documentation_format)
            conv.append_message(roles['assistant'], None)
            prompt = conv.get_prompt()
            if print_prompt:
                print(prompt)
            action_input = self.generate(prompt, do_sample=True, temperature=temperature, restrict_actions=False)['generated_text']
            decoded_token_len = len(self.tokenizer(action_input)[0]) + 1
            if process_id == 0:
                print(f"[process({process_id})]total tokens: {decoded_token_len}")

            
            # arguments = action_input[action_input.find("Action Input: ") + len("Action Input: "):]
            arguments = action_input
        else:
            # raise RuntimeError("Invalid action.")
            # toolbench_name = "Invalid Action"
            documentation_format = f"Action does not exist. Please try again."
            conv.append_message(roles['assistant'], documentation_format)
            conv.append_message(roles['assistant'], None)
            prompt = conv.get_prompt()
            if print_prompt:
                print(prompt)
            arguments = "{}"
            decoded_token_len = 0

        return arguments, decoded_token_len


    def parse(
        self,
        process_id,
        tools,
        planning=True,
        temperature=0.0,
        add_retrieval=False,
        restrict_actions=False,
        restrict_to_ground_truth_tools=False,
        add_planning_prefix=False,
        add_relevant_tokens=False,
        do_retry=False,
        retry_action=False,
        retry_give_up=True,
        retry_sorry=True,
        print_prompt=False,
        
    ):
        
        conversation_gpt = self.convert_conversation_history_to_gpt_format(tools, add_retrieval=add_retrieval)

        conv, roles = self.convert_to_fastchat_format(
            tools,
            conversation_gpt,
            add_relevant_tokens=add_relevant_tokens,
        )      
        
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
            thought, conv_try = self.planning(
                tools,
                conv_try,
                roles,
                planning,
                add_planning_prefix=add_planning_prefix,
                temperature=temperature
            )

            action, conv_try = self.acting(
                tools,
                conv_try,
                roles,
                restrict_to_ground_truth_tools,
                restrict_actions,
                temperature=temperature
            )
            
            arguments, decoded_token_len = self.calling(
                action,
                roles,
                conv_try,
                process_id=process_id,
                temperature=temperature,
                print_prompt=print_prompt
            )

            
            if action in self.token_to_toolbench_name:
                toolbench_name = self.token_to_toolbench_name[action]
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
            
            
        # Test if the action is valid
        if action in self.token_to_toolbench_name:
            toolbench_name = self.token_to_toolbench_name[action]
        # If the action is invalid, we use the invalid action token, which will be handled by the execution program
        else:
            toolbench_name = action

            
        
        message = {
            "role": "assistant",
            "content": thought,
            "function_call": None,
            "tool_calls": [{
                "id": None,
                "type": "function",
                "function": {
                    "name": toolbench_name,
                    "arguments": arguments
                }
            }]
        }

        print(self.logs['retry_times'])
        return message, 0, decoded_token_len


if __name__ == "__main__":
    # can accept all huggingface LlamaModel family
#     llm = ToolLLaMA("decapoda-research/llama-7b-hf")
#     messages = [
#         {'role': 'system', 'content': '''You are AutoGPT, you can use many tools(functions) to do
# the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is , you can\'t go
# back to the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\nLet\'s Begin!\nTask description: Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number=24. Each
# step, you are only allowed to choose two of the left numbers to obtain a new number. For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.\nRemember:\n1.all of the number must be used , and must be used ONCE. So Only when left numbers is exact 24, you will win. So you don\'t succeed when left number = [24, 5]. You succeed when left number = [24]. \n2.all the try takes exactly 3 steps, look
# at the input format'''}, 
# {'role': 'user', 'content': '\nThe real task input is: [1, 2, 4, 7]\nBegin!\n'}
# ]
#     functions = [{'name': 'play_24', 'description': '''make your current conbine with the format "x operation y = z (left: aaa) " like "1+2=3, (left: 3 5 7)", then I will tell you whether you win. This is the ONLY way
# to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1''','parameters':{'type': 'object', 'properties':{}}}]#, 'parameters': {'type': 'object', 'properties': {'input': {'type': 'string', 'description': 'describe what number you want to conbine, and how to conbine.'}}, 'required': ['input']}}]

#     llm.change_messages(messages)
#     output = llm.parse(functions=functions)
#     print(output)
    pass