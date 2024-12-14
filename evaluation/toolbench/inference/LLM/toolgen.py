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

def load_tokenizer(model_name_or_path, cache_dir=None, indexing=None):
    if "llama-3" in model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B", 
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Not supported for tokenizer {model_name_or_path}")
    if indexing == "Atomic":
        with open('data/virtual_tokens.txt', 'r') as f:
            virtual_tokens = f.readlines()
            virtual_tokens = [unidecode(vt.strip()) for vt in virtual_tokens]
        tokenizer.add_tokens(new_tokens=virtual_tokens, special_tokens=False)
        print(f"Added {len(virtual_tokens)} virtual tokens")

    return tokenizer


def load_tool_documentation(toolbench_name_to_toolgen_tokens_dict):
    # Build token api document
    with open("data/toolenv/tools"+"/"+"tools.json", 'r') as f:
        all_tools = json.load(f)
    print(len(all_tools))

    toolbench_virtual_token_to_document = {}
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
            toolgen_tokens = toolbench_name_to_toolgen_tokens_dict[toolbench_name]
            toolbench_virtual_token_to_document[toolgen_tokens] = {
                "name": toolgen_tokens,
                "description": api_description,
                "required": api['required_parameters'],
                "optional": api['optional_parameters']
            }

    finish_tokens = toolbench_name_to_toolgen_tokens_dict['Finish']
    # document for finish action
    toolbench_virtual_token_to_document[finish_tokens] = {
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
    

def load_toolgen_token_toobench_name_conversion(indexing):
    # with open('data/virtual_tokens.txt', 'r') as f:
    #     virtual_tokens = f.readlines()
    # virtual_tokens = [vt.strip() for vt in virtual_tokens]

    # toolbench_name_to_virtual_token_dict = {}
    # virtual_token_to_toolbench_name_dict = {}
    # for vt in virtual_tokens:
    #     if vt == "<<Finish>>":
    #         toolbench_name_to_virtual_token_dict["Finish"] = vt
    #         virtual_token_to_toolbench_name_dict[vt] = "Finish"
    #     else:
    #         names = vt[2:-2].split("&&")
    #         tool_name = names[0]
    #         api_name = names[1]
    #         toolbench_name = get_toolbench_name(tool_name, api_name)
    #         virtual_token_to_toolbench_name_dict[unidecode(vt)] = toolbench_name
    #         toolbench_name_to_virtual_token_dict[toolbench_name] = unidecode(vt)
    with open(f"data/toolenv/Tool2{indexing}Id.json", 'r') as f:
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


class ToolGen:
    def __init__(
            self, 
            model_name_or_path: str, 
            template:str="llama-3",
            indexing: str="Semantic",
            device: str="cuda", 
            cpu_offloading: bool=False, 
            max_sequence_length: int=8192
        ) -> None:
        super().__init__()
        self.model_name = model_name_or_path
        self.indexing = indexing
        self.template = template
        self.max_sequence_length = max_sequence_length
        self.tokenizer = load_tokenizer(model_name_or_path, indexing=indexing)
        # Only support llama-3 currently
        self.tokenizer.eos_token = "<|eot_id|>"
        self.tokenizer.eos_token_id = 128009
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        self.use_gpu = (True if device == "cuda" else False)
        if (device == "cuda" and not cpu_offloading) or device == "mps":
            self.model.to(device)
        self.chatio = SimpleChatIO()

        self.token_to_toolbench_name, self.toolbench_name_to_token = load_toolgen_token_toobench_name_conversion(indexing=indexing)
        self.tool_documentation = load_tool_documentation(self.toolbench_name_to_token)
        
        
        self.retrieved_actions = None
        self.relevant_actions_documentations = None
        # self.allowed_action_ids = [128009] + list(range(128256, 128256 + 46985))
        
        # Used for constrained generation to only allow the action tokens
        candidate_actions = list(self.token_to_toolbench_name.keys())
        candidate_actions_ids = self.tokenizer(candidate_actions, add_special_tokens=False)['input_ids']
        print(f"candidate_actions_ids: {candidate_actions_ids[:10]}")
        candidate_actions_ids = [ids + [self.tokenizer.eos_token_id] for ids in candidate_actions_ids]
        self.trie = DisjunctiveTrie(candidate_actions_ids)

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
                        AllowKeyWordsProcessor(self.tokenizer, self.trie, inputs["input_ids"])
                    ])
            else:
                trie_allowed_action_ids = [ids + self.tokenizer.eos_token_id for ids in allowed_action_ids]
                trie = DisjunctiveTrie(allowed_action_ids)
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
            tool_tokens.append(self.toolbench_name_to_token["Finish"])
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
        print(f"Retry times: {self.logs['retry_times']}")
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
        return message, 0, decoded_token_len
