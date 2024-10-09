from abc import abstractmethod
import re
from .tree.tree import TreeNode, Tree
from copy import deepcopy


class SingleChainAgent():
    """Implement of CoT method
    """
    def __init__(self, io_func, extra_prefix="", process_id=0, start_message_list=None):
        """extra_prefix and start_message_list is used in Reflection Algo"""
        self.io_func = io_func
        self.extra_prefix = extra_prefix
        self.start_message_list = start_message_list
        self.process_id = process_id

        self.restart()


    def restart(self):
        self.status = 0
        # self.try_list = []
        self.chain = []
        self.terminal_node = []

        self.query_count = 0 # number of interactions with openai
        self.total_tokens = 0
        self.success_count = 0

    def to_json(self):
        # if process:
        json_obj = {
            "finish": self.status == 1,
            "chain": self.chain,
            "forward_args":self.forward_args,
        }

        return json_obj

    def start(self, single_chain_max_step, pass_at=1, answer=1, start_messages=None):
        if start_messages:
            self.start_messages = start_messages
            
        self.forward_args = locals()
        if "self" in self.forward_args.keys():
            self.forward_args.pop("self")

        for i in range(pass_at):
            if self.process_id == 0:
                print(f"[single_chain]try for the {i+1} time")
            self.tree = Tree()
            self.tree.root.node_type = "Action Input"
            self.tree.root.io_state = deepcopy(self.io_func)
            out_node = self.do_chain(self.tree.root, single_chain_max_step)
            self.chain = out_node.get_chain_result_from_this_node()
            if out_node.io_state.check_success() == 1:
                self.status = 1
                self.success_count += 1
                if self.success_count >= answer:
                    return 1
        return 0
    
    @abstractmethod
    def change_messages(self, messages):
        raise NotImplementedError
    
    @abstractmethod
    def parse(self, tools, process_id, **args):
        raise NotImplementedError

    def get_agent_response(self, node):
        self.change_messages(node.messages)
        new_message, error_code, total_tokens = self.parse(tools=self.io_func.tools,
                                                                process_id=self.process_id)
        self.total_tokens += total_tokens
        self.query_count += 1
        assert new_message["role"] == "assistant"
        return new_message, error_code, total_tokens

    def parse_planning(self, message, node, error_code):
        if "content" in message.keys() and message["content"] != None:
            temp_node = TreeNode()
            temp_node.node_type = "Thought"
            temp_node.description = message["content"]
            child_io_state = deepcopy(node.io_state)
            
            temp_node.io_state = child_io_state
            temp_node.is_terminal = child_io_state.check_success() != 0
            print("is_terminal:",temp_node.is_terminal)
            temp_node.messages = node.messages.copy()
            temp_node.father = node
            node.children.append(temp_node)
            temp_node.print(self.process_id)

            if error_code != 0:
                temp_node.observation_code = error_code
                temp_node.pruned = True

            return temp_node
        else:
            return None

    def take_action(self, tool_call, node):
        function_name = tool_call["function"]["name"]
        temp_node = TreeNode()
        temp_node.node_type = "Action"
        temp_node.description = function_name
        child_io_state = deepcopy(node.io_state)

        temp_node.io_state = child_io_state
        temp_node.is_terminal = child_io_state.check_success() != 0 
        temp_node.messages = node.messages.copy()
        temp_node.father = node
        node.children.append(temp_node)

        temp_node.print(self.process_id)
        node = temp_node

        function_input = tool_call["function"]["arguments"]
        temp_node = TreeNode()
        temp_node.node_type = "Action Input"
        temp_node.description = function_input
        child_io_state = deepcopy(node.io_state)


        observation, status = child_io_state.call(action_name=node.description, action_input=function_input)
        temp_node.observation = observation
        temp_node.observation_code = status

        temp_node.io_state = child_io_state
        temp_node.is_terminal = child_io_state.check_success() != 0 
        temp_node.messages = node.messages.copy()
        temp_node.father = node
        node.children.append(temp_node)
        temp_node.print(self.process_id)

        return temp_node, status
    

    def do_chain(self, now_node, single_chain_max_step):
        if self.start_messages:
            """In Reflection Algo, we startswith former trials and reflections, so the caller will give the start messages"""
            self.tree.root.messages = self.start_messages
        
        now_node = self.tree.root
        while True:
            # recursively parse message into nodes
            new_message, error_code, total_tokens = self.get_agent_response(now_node)
            temp_node = self.parse_planning(new_message, now_node, error_code)
            if temp_node:
                now_node = temp_node
            
            now_node.messages.append(new_message)

            if "tool_calls" in new_message.keys() and new_message["tool_calls"] != None and len(new_message["tool_calls"]) > 0:
                tool_calls = new_message["tool_calls"]
                if self.process_id == 0:
                    print("number of parallel calls:",len(tool_calls))

                # lastnode = now_node
                for i in range(len(tool_calls)):
                    tool_call = tool_calls[i]
                    temp_node, status = self.take_action(tool_call, now_node)
                    now_node = temp_node
                    
                    if status != 0:
                        # return code refers to Downstream_tasks/rapidapi
                        if status == 4:
                            now_node.pruned = True
                        elif status == 1: # hallucination api name
                            assert "tool_calls" in new_message.keys() and len(new_message["tool_calls"]) > 0
                            tool_calls[i]["function"]["name"] = "invalid_hallucination_function_name"

                    if now_node.node_type == "Action Input":
                        now_node.messages.append({
                            "role":"tool",
                            "name": tool_calls[i]["function"]["name"],
                            "content": now_node.observation,
                            "tool_call_id": tool_calls[i]['id'],
                        })
            else:
                now_node.messages.append(new_message)

            if now_node.get_depth() >= single_chain_max_step and not (now_node.is_terminal):
                now_node.pruned = True
            
            if now_node.pruned or now_node.is_terminal:
                return now_node
