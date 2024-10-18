import os
from openai import OpenAI
import httpx
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import time
import json
import traceback

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(key, base_url, messages, tools=None, tool_choice=None, key_pos=None,
                            model="gpt-3.5-turbo", stop=None, process_id=0, **args):
    use_messages = []
    for message in messages:
        if not ("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)
    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        json_data.update({"stop": stop})
    # if functions is not None:
    #     # Funcall calling request a list of functions
    #     if isinstance(functions, dict):
    #         functions = list(functions.values())
    #     json_data.update({"functions": functions})
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})

    
    # print(json_data)
    ## TODO: This error processing sucks, should be refined
    try:
        # openai.api_key = key
        # openai_response = openai.ChatCompletion.create(
        #     **json_data,
        # )
        # json_data = json.loads(str(openai_response))
        # return json_data 
        if model.startswith("gpt"):
            client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)
        else:
            raise NotImplementedError("Model not supported")
        # print(json_data)
        # print(json_data)
        openai_response = client.chat.completions.create(**json_data)
        json_data = openai_response.dict()
        return json_data
    
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        # print(f"OpenAI calling Exception: {e}")
        # traceback.print_exc()
        # import pdb;  pdb.set_trace()
        raise e

class ChatGPTFunction:
    def __init__(self, model="gpt-4-turbo-2024-04-09", openai_key="", base_url=None):
        self.model = model
        self.conversation_history = []
        self.openai_key = openai_key
        self.base_url = base_url
        self.time = time.time()
        self.TRY_TIME = 6
        self.logs = {
            "retry_times": {
                "actions": 0,
                "give_up": 0,
                "sorry": 0,
            },
            "actions": {
            }
        }

    def initialize(self):
        self.logs["actions"] = {}
        print("Agent initialized.")

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self, messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print"+"*" * 50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            if 'tool_calls' in message.keys():
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
                print_obj = print_obj + f"number of tool calls: {len(message['tool_calls'])}"
            if detailed:
                print_obj = print_obj + f"function_call: {message['function_call']}"
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
                print_obj = print_obj + f"function_call_id: {message['function_call_id']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print"+"*" * 50)

    def parse(
            self,
            tools,
            process_id,
            key_pos=None,
            do_retry=False,
            retry_action=False,
            retry_give_up=True,
            retry_sorry=True,
            print_prompt=False,
            **args):
        
        if isinstance(tools, dict):
            tools = list(tools.values())
        self.time = time.time()
        conversation_history = self.conversation_history
        if print_prompt:
            print(conversation_history)
        # for _ in range(self.TRY_TIME):
        #     if _ != 0:
        #         time.sleep(15)
                
            # if functions != []:
            #     json_data = chat_completion_request(
            #         self.openai_key, conversation_history, 
            #         model=self.model,
            #         functions=functions,process_id=process_id, key_pos=key_pos,**args
            #     )
        if do_retry:
            max_retry_times = 5
        else:
            max_retry_times = 1
        for retry in range(max_retry_times):
            if tools != []:
                # We have changed tools into a dict, but the model accept a list
                # So we need to convert it back to a list
                response = chat_completion_request(
                    self.openai_key, self.base_url, conversation_history, tools=tools, process_id=process_id, key_pos=key_pos,
                    model=self.model, **args
                )
            else:
                # json_data = chat_completion_request(
                #     self.openai_key, conversation_history,
                #     model=self.model,
                #     process_id=process_id,key_pos=key_pos, **args
                # )
                response = chat_completion_request(
                    self.openai_key, self.base_url, conversation_history, process_id=process_id, key_pos=key_pos, model=self.model, **args
                )
            try:
            # total_tokens = json_data['usage']['total_tokens']
            # message = json_data["choices"][0]["message"]
            # if process_id == 0:
            #     print(f"[process({process_id})]total tokens: {json_data['usage']['total_tokens']}")
                total_tokens = response['usage']['total_tokens']
                message = response["choices"][0]["message"]

                # if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                #     message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]
                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {total_tokens}")

                thought = message["content"]
                if message['tool_calls']:
                    action = message['tool_calls'][0]['function']['name']
                    arguments = message['tool_calls'][0]['function']['arguments']
                else:
                    action = None
                    arguments = None

                if action:
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

                print(f"Retry times: {self.logs['retry_times']}")
                return message, 0, total_tokens
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                traceback.print_exc()
                if response is not None:
                    print(f"[process({process_id})]OpenAI return: {response}")

                # Encountering error, retry the whole process
                break
            

        return {"role": "assistant", "content": str(response)}, -1, 0
    
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


if __name__ == "__main__":
#     openai.api_base = "" 
#     llm = ChatGPTFunction(openai_key='')
#     prompt = '''下面这句英文可能有语病，能不能把语病都改掉？
# If you think you get the result which can answer the task, call this function to give the final answer. Or, if you think you can't handle the task from this status, call this function to restart. Remember: you should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be showed to user, so final answer should contain enough information.
# 没语病的形式：
# '''
#     messages = [
#         {"role":"system","content":""},
#         {"role":"user","content":prompt},
#     ]
#     llm.change_messages(messages)
#     output,error_code,token_usage = llm.parse(functions=[],process_id=0)
#     print(output)
    llm = ChatGPTFunction(openai_key="", model="gpt-3.5-turbo")
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    llm.change_messages(messages)
    output, error_code, token_usage = llm.parse(tools=tools, process_id=0)
    print(output)
