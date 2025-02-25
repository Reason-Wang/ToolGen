import os
from openai import OpenAI, AzureOpenAI
import httpx
from .base import SingleChainAgent
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import time
import json
import traceback


class OpenAIFunctionAgent(SingleChainAgent):
    def __init__(self, model, tools, openai_key="", openai_api_base=None, openai_api_version=None, start_messages=None, base_url=None):
        super(OpenAIFunctionAgent, self).__init__(tools, start_message_list=start_messages)
        if openai_api_base is None:
            self.client = OpenAI(api_key=openai_key)
        else:
            print("Using AzureOpenAI")
            self.client = AzureOpenAI(
                api_key=openai_key,
                azure_endpoint=openai_api_base,
                api_version=openai_api_version
            )
        self.openai_key = openai_key
        self.model = model
        self.conversation_history = []
        self.base_url = base_url
        self.time = time.time()
        self.TRY_TIME = 6

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

    def parse(self, tools, process_id, key_pos=None, **args):
        self.time = time.time()
        conversation_history = self.conversation_history
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(15)
            
            if tools != []:
                # We have changed tools into a dict, but the model accept a list
                # So we need to convert it back to a list
                if isinstance(tools, dict):
                    tools = list(tools.values())
                # print(tools)
                response = chat_completion_request(
                    self.openai_key, self.base_url, conversation_history, tools=tools, process_id=process_id, key_pos=key_pos,
                    model=self.model, **args
                )
            else:
                 response = chat_completion_request(
                    self.openai_key, self.base_url, conversation_history, process_id=process_id, key_pos=key_pos, model=self.model, **args
                )
            try:
                total_tokens = response['usage']['total_tokens']
                message = response["choices"][0]["message"]

                if process_id == 0:
                    print(f"[process({process_id})]total tokens: {total_tokens}")

                # print(message)
                return message, 0, total_tokens
            except BaseException as e:
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                traceback.print_exc()
                if response is not None:
                    print(f"[process({process_id})]OpenAI return: {response}")

        return {"role": "assistant", "content": str(json_data)}, -1, 0
    
    
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(key, base_url, messages, tools=None, tool_choice=None, key_pos=None, model="gpt-3.5-turbo", stop=None, process_id=0, **args):
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
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})

    try:
        if model.startswith("gpt"):
            client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)
        else:
            raise NotImplementedError("Model not supported")
        # print(json_data)
        openai_response = client.chat.completions.create(**json_data)
        json_data = openai_response.dict()
        return json_data
    
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        raise e
    
    
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
    llm = OpenAIFunctionAgent(openai_key="", model="gpt-3.5-turbo")
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
