#!/usr/bin/env python
# coding=utf-8
from copy import deepcopy
import json
import re
import time
from termcolor import colored
from typing import Optional, List
# from toolbench.utils import change_name, standardize
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
)
from unidecode import unidecode
from ..base import SingleChainAgent
from fastchat.conversation import get_conv_template
from huggingface_hub import hf_hub_download
from .utils import get_toolbench_name
from .inference import AllowKeyWordsProcessor, DisjunctiveTrie


SystemPrompt = '''You are an AutoGPT, capable of utilizing numerous tools and functions to complete the given task.
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

# def load_tokenizer(model_name_or_path, cache_dir=None, indexing=None):
#     if "llama-3" in model_name_or_path.lower():
#         tokenizer = AutoTokenizer.from_pretrained(
#             "meta-llama/Meta-Llama-3-8B", 
#             cache_dir=cache_dir,
#         )
#     else:
#         raise ValueError(f"Not supported for tokenizer {model_name_or_path}")
#     if indexing == "Atomic":
#         with open('data/virtual_tokens.txt', 'r') as f:
#             virtual_tokens = f.readlines()
#             virtual_tokens = [unidecode(vt.strip()) for vt in virtual_tokens]
#         tokenizer.add_tokens(new_tokens=virtual_tokens, special_tokens=False)
#         print(f"Added {len(virtual_tokens)} virtual tokens")

#     return tokenizer


def load_tool_documentation(model_name_or_path, toolbench_name_to_toolgen_tokens_dict):
    # Build token api document
    # with open("data/toolenv/tools"+"/"+"tools.json", 'r') as f:
    #     all_tools = json.load(f)
    tools_path = hf_hub_download(repo_id=model_name_or_path, filename="tools.json")
    with open(tools_path, 'r') as f:
        all_tools = json.load(f)
    print(len(all_tools))

    toolbench_virtual_token_to_document = {}
    keyerror_num = 0
    for tool in all_tools:
        if 'name' in tool:
            tool_name = tool['name']
        elif 'tool_name' in tool:
            tool_name = tool['tool_name']
        else:
            raise RuntimeError
        for api in tool['api_list']:
            # token = f"<<{tool_name}&&{api['name']}>>"
            api_description = api['description']
            if api_description is None:
                api_description = "None."
            else:
                api_description = api_description.strip()
                if api_description == "":
                    api_description = "None."
            toolbench_name = get_toolbench_name(tool_name, api['name'])

            try:
                toolgen_tokens = toolbench_name_to_toolgen_tokens_dict[toolbench_name]
            except KeyError:
                keyerror_num += 1
                

            
            toolbench_virtual_token_to_document[toolgen_tokens] = {
                "name": toolgen_tokens,
                "description": api_description,
                "required": api['required_parameters'],
                "optional": api['optional_parameters']
            }
    print(f"keyerror_num: {keyerror_num}")
    finish_tokens = toolbench_name_to_toolgen_tokens_dict['Finish']
    # document for finish action
    toolbench_virtual_token_to_document[finish_tokens] = {
        "name": finish_tokens,
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
    

def load_toolgen_token_toobench_name_conversion(model_name_or_path, indexing):
    # with open(f"data/toolenv/Tool2{indexing}Id.json", 'r') as f:
    #     toolbench_name_to_toolgen_tokens_dict = json.load(f)
    tool2id_path = hf_hub_download(repo_id=model_name_or_path, filename="Tool2Id.json")
    with open(tool2id_path, 'r') as f:
        toolbench_name_to_toolgen_tokens_dict = json.load(f)

    toolgen_tokens_to_toolbench_name_dict = {}
    finish_to_tokens_dict = {
        "Semantic": "Finish",
        "Numeric": "4 6 7 3 2",
        "Hierarchical": "9 9 9 9 3",
        "Atomic": "<<Finish>>"
    }
    toolbench_name_to_toolgen_tokens_dict['Finish'] = finish_to_tokens_dict[indexing]
    for k, v in toolbench_name_to_toolgen_tokens_dict.items():
        toolgen_tokens_to_toolbench_name_dict[v] = k

    toolbench_name_to_toolgen_tokens_dict["invalid_hallucination_function_name"] = "Invalid Action"
    return toolgen_tokens_to_toolbench_name_dict, toolbench_name_to_toolgen_tokens_dict


class ToolGen(SingleChainAgent):
    def __init__(
            self,
            model_name_or_path: str,
            tools,
            template:str="llama-3",
            indexing: str="Atomic",
            device: str="cuda", 
            cpu_offloading: bool=False, 
            max_sequence_length: int=8192
        ) -> None:
        super(ToolGen, self).__init__(tools)

        self.model_name = model_name_or_path
        self.indexing = indexing
        self.template = template
        self.max_sequence_length = max_sequence_length
        # self.tokenizer = load_tokenizer(model_name_or_path, indexing=indexing)
        # Only support llama-3 currently
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if template == "llama-3":
            self.tokenizer.eos_token = "<|eot_id|>"
            self.tokenizer.eos_token_id = 128009
        elif template == "qwen-7b-chat":
            self.tokenizer.eos_token = "<|im_end|>"
            self.tokenizer.eos_token_id = 151645
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
        )

        self.use_gpu = (True if device == "cuda" else False)
        if (device == "cuda" and not cpu_offloading) or device == "mps":
            self.model.to(device)

        self.token_to_toolbench_name, self.toolbench_name_to_token = load_toolgen_token_toobench_name_conversion(model_name_or_path, indexing=indexing)
        self.tool_documentation = load_tool_documentation(model_name_or_path, self.toolbench_name_to_token)
        self.retrieved_actions = None
        self.relevant_actions_documentations = None
        
        # Used for constrained generation to only allow the action tokens
        candidate_actions = list(self.token_to_toolbench_name.keys())
        candidate_actions_ids = self.tokenizer(candidate_actions, add_special_tokens=False)['input_ids']
        # print(f"candidate_actions_ids: {candidate_actions_ids[:10]}")
        candidate_actions_ids = [ids + [self.tokenizer.eos_token_id] for ids in candidate_actions_ids]
        self.trie = DisjunctiveTrie(candidate_actions_ids)
    
    def generate(self, text, do_sample, temperature=0.6, restrict_actions=False, allowed_action_ids=None):
        inputs = self.tokenizer(text, return_tensors='pt')
        input_length = inputs["input_ids"].shape[1]
        for k, v in inputs.items():
            inputs[k] = v.to("cuda")
        if restrict_actions:
            if allowed_action_ids is None:
                logits_processor = LogitsProcessorList([
                        AllowKeyWordsProcessor(self.tokenizer, self.trie, inputs["input_ids"])
                    ])
            else:
                trie_allowed_action_ids = [ids + self.tokenizer.eos_token_id for ids in allowed_action_ids]
                trie = DisjunctiveTrie(trie_allowed_action_ids)
                logits_processor = LogitsProcessorList([
                        AllowKeyWordsProcessor(self.tokenizer, trie, inputs["input_ids"])
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
        output_length = output_ids.shape[0]
        generated_text = self.tokenizer.decode(output_ids)
        return {
            "output_ids": output_ids,
            "output_length": output_length,
            "generated_text": generated_text
        }       
        
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
        tool_id = self.tokenizer(self.toolbench_name_to_token["Finish"])['input_ids'][1]
        virtual_tool_ids.append(tool_id)
        # virtual_tool_ids.append(128009)
        print(f"virtual_tool_ids: {virtual_tool_ids}")
        return virtual_tool_ids
    
    def convert_conversation_history_to_gpt_format(
            self,
            tools,
        ):
        conversation_history = deepcopy(self.conversation_history)
        conversation_gpt = []
        for i, turn in enumerate(conversation_history):
            if turn["role"] == "system":
                conversation_gpt.append({"role": "system", "content": turn['content']})
            elif turn['role'] == 'user':
                conversation_gpt.append({"role": "user", "content": turn['content']})
            elif turn['role'] == 'assistant':
                if turn['content'] is not None:
                    conversation_gpt.append({"role": "assistant", "content": turn['content']})
                    conversation_gpt.append({"role": "user", "content": "Generate the action."})

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
                # Deprecated, do not use
                if add_relevant_tokens:
                    virtual_token_ids = self.get_token_ids_by_actions(tools)
                    self.relevant_actions = self.tokenizer.convert_ids_to_tokens(virtual_token_ids)
                    content = SystemPromptTokens.format(
                        finish=self.toolbench_name_to_token["Finish"],
                        actions=str(self.relevant_actions)
                    )
                else:
                    content = SystemPrompt.format(finish=self.toolbench_name_to_token["Finish"])
                conv.set_system_message(content)
            else:
                role = roles[turn["role"]]
                content = turn["content"]
                conv.append_message(role, content)
        return conv, roles

    
    def planning(self, tools, conv, roles, planning, temperature=0.1):
        conv.append_message(roles['assistant'], None)
        prompt = conv.get_prompt()

        if planning:
            outputs = self.generate(prompt, do_sample=True, temperature=temperature, restrict_actions=False)
            thought = outputs['generated_text']
            decoded_length = outputs['output_length']
            conv.messages[-1] = (roles['assistant'], thought)
        else:
            thought = None
            decoded_length = 0

        return thought, conv, decoded_length
    
    def acting(self, tools, conv, roles, restrict_to_ground_truth_tools, restrict_actions, temperature=0.1):
        conv.append_message(roles['user'], "Generate the action.")
        conv.append_message(roles['assistant'], None)
        prompt = conv.get_prompt()
        
        if restrict_to_ground_truth_tools:
            virtual_tool_ids = self.get_token_ids_by_actions(tools)
        else:
            virtual_tool_ids = None
        
        outputs = self.generate(
            prompt, 
            do_sample=True, 
            temperature=temperature,
            restrict_actions=restrict_actions,
            allowed_action_ids=virtual_tool_ids
        )
        action = outputs['generated_text']
        decoded_length = outputs['output_length']

        conv.messages[-1] = (roles['assistant'], action)

        return action, conv, decoded_length

    def calling(self, action, roles, conv, process_id, print_prompt=False, temperature=0.1):
        virtual_token = action
        if virtual_token in self.tool_documentation:
            documentation_format = f"Please give the input. Here is the documentation: {self.tool_documentation[virtual_token]}"
            conv.append_message(roles['assistant'], documentation_format)
            conv.append_message(roles['assistant'], None)
            prompt = conv.get_prompt()
            if print_prompt:
                print(prompt)
            outputs = self.generate(prompt, do_sample=True, temperature=temperature, restrict_actions=False)
            action_input = outputs['generated_text']
            decoded_length = outputs['output_length']
            if process_id == 0:
                print(f"[process({process_id})]total tokens: {decoded_length}")

            
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
            decoded_length = 0

        return arguments, decoded_length


    def parse(
        self,
        process_id,
        tools,
        planning=True,
        temperature=0.6,
        restrict_actions=True,
        restrict_to_ground_truth_tools=False,
        add_relevant_tokens=False,
        print_prompt=False,
    ):
        conversation_gpt = self.convert_conversation_history_to_gpt_format(tools)

        conv, roles = self.convert_to_fastchat_format(
            tools,
            conversation_gpt,
            add_relevant_tokens=add_relevant_tokens,
        )      

        num_tokens = 0
        thought, conv, decoded_length = self.planning(
            tools,
            conv,
            roles,
            planning,
            temperature=temperature
        )
        num_tokens += decoded_length

        action, conv, decoded_length= self.acting(
            tools,
            conv,
            roles,
            restrict_to_ground_truth_tools,
            restrict_actions,
            temperature=temperature
        )
        num_tokens += decoded_length
        
        arguments, decoded_length = self.calling(
            action,
            roles,
            conv,
            process_id=process_id,
            temperature=temperature,
            print_prompt=print_prompt
        )
        num_tokens += decoded_length
        
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
        return message, 0, num_tokens