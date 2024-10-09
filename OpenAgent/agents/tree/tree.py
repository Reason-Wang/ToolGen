from termcolor import colored
import numpy as np
from copy import deepcopy
import math

class Tree:
    def __init__(self):
        self.root = TreeNode()
        self.now_deal_node = self.root


    def to_json_recursive(self,use_messages=False):
        tree_structure =  self.root.to_json_recursive(use_messages=use_messages)
        js_obj = {
            "size": self.root.get_size(),
            "max_length":self.root.get_max_depth(),
            "tree": tree_structure,
        }
        return js_obj


class TreeNode:

    def __init__(self):
        self.is_terminal = False
        self.pruned = False
        # self.finished = False
        self.node_type = None
        self.description = ""
        self.observation = ""
        self.observation_code = None
        self.father = None
        self.children = []
        # self.io_state = None

        # openai-messages of this node
        self.messages = []


    def get_depth(self):
        if self.father == None:
            return 0
        return self.father.get_depth() + 1

    def print(self,process_id = 0):
        if process_id != 0:
            return
        color_converter = {"Thought":"red", "Action": "blue", "Action Input": "cyan","Final Answer": "green","Reflection":"blue"}
        print(colored(f"{self.node_type}: {self.description}",color = color_converter[self.node_type]))
        if self.observation != "":
            if len(self.observation) < 1536:
                print(colored(f"Observation: {self.observation}",color="yellow"))
            else:
                print(colored(f"Observation: {self.observation[:1536]}......(len={len(self.observation)})",color="yellow"))
    

    def to_json_recursive(self,use_messages=False):
        js_obj = self.to_json(use_messages=use_messages)
        js_obj["children"] = []
        for child in self.children:
            js_obj["children"].append(child.to_json_recursive())
        return js_obj


    def get_chain_result_from_this_node(self,use_messages=False):
        '''
        Returns chained results, starting from this node up to the root node
        '''
        now_node = self
        result = []
        while now_node.father != None:
            result = [now_node.to_json(use_messages=use_messages)] + result
            now_node = now_node.father
        return result

    def to_json(self, use_messages=False):
        
        json_obj = {}
        json_obj["is_terminal"] = self.is_terminal
        json_obj["pruned"] = self.pruned

        json_obj["depth"] = self.get_depth()
        json_obj["node_type"] = self.node_type
        json_obj["description"] = self.description
        if self.observation != "":
            json_obj["observation"] = self.observation
        if self.observation_code != None:
            json_obj["observation_code"] = self.observation_code
        json_obj["child_count"] = len(self.children)

        # if self.io_state != None and self.node_type == "Action Input":
        #     json_obj["io_state"] = self.io_state.to_json()

            
        if use_messages:
            json_obj["messages"] = []
            for message in self.messages:
                if not ("valid" in message.keys() and message["valid"] == False):
                    json_obj["messages"].append(message["role"])
                else:
                    json_obj["messages"].append(message["role"] + "_invalid")

        return json_obj