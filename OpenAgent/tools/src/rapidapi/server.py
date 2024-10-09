
import random
import time
from typing import Union
from pydantic import BaseModel
import requests
from termcolor import colored
from .utils import change_name, standardize
import os
import json

class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str

def prepare_tool_name_and_url(tools_root, info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")
    
    tool_name = info.tool_name
    api_name = change_name(standardize(info.api_name))
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        code_string = f"""from {tools_root}.{standard_category}.{tool_name}.api import {api_name}"""
        tool_name += f"_for_{standard_category}"
    else:
        tmp_tool_name = standardize(tool_name.replace(f"_for_{standard_category}", ""))
        code_string = f"""from {tools_root}.{standard_category}.{tmp_tool_name}.api import {api_name}"""
    return tool_name, standard_category, api_name, code_string

def process_error(response):
    save_cache_flag = False
    switch_flag = False
    if "The request to the API has timed out. Please try again later, or if the issue persists" in str(response):
        return_dict = {"error": "API temporarily not working error...", "response": response}

    if "Your Client (working) ---> Gateway (working) ---> API (not working)" in str(response):
        return_dict = {"error": "API not working error...", "response": response}
        
    elif "Unauthorized" in str(response) or "unauthorized" in str(response):
        save_cache_flag = True
        return_dict = {"error": "Unauthorized error...", "response": response}
    
    elif "You are not subscribed to this API." in str(response):
        switch_flag = True
        return_dict = {"error": "Unsubscribed error...", "response": response}
    
    elif "Too many requests" in str(response):
        switch_flag = True
        return_dict = {"error": "Too many requests error...", "response": response}

    elif "You have exceeded" in str(response) or "you are being rate limited"  in str(response):
        switch_flag = True
        return_dict = {"error": "Rate limit error...", "response": response}

    elif "Access restricted. Check credits balance or enter the correct API key." in str(response):
        switch_flag = True
        return_dict = {"error": "Rate limit error...", "response": response}
    
    elif "Oops, an error in the gateway has occurred." in str(response):
        switch_flag = True
        return_dict = {"error": "Gateway error...", "response": response}

    elif "Blocked User. Please contact your API provider." in str(response):
        switch_flag = True
        return_dict = {"error": "Blocked error...", "response": response}
    
    elif "error" in str(response):
        return_dict = {"error": "Message error...", "response": response}

    else:
        save_cache_flag = True
        return_dict = {"error": "", "response": response}
    return return_dict, save_cache_flag, switch_flag

def run(toolbench_code_string, toolbench_api_name, toolbench_input_params_str):
    # get observation
    success_flag = False
    switch_flag = False
    save_cache = False
    exec(toolbench_code_string)
    try:
        eval_func_str = f"{toolbench_api_name}({toolbench_input_params_str})"
        new_func = eval(eval_func_str)
        response, save_cache, switch_flag = process_error(new_func)
        success_flag = True
    except Exception as e:
        response = {"error": f"Function executing {toolbench_code_string} error...\n{e}", "response": ""}
        save_cache = False
    return success_flag, switch_flag, response, save_cache


def dict_shorten(origin: dict, schema: dict):
    for key, value in list(origin.items()):
        if key not in schema:
            del origin[key]
        else:
            if isinstance(value, dict):
                dict_shorten(value, schema[key]) # schema[key] should be a dict
            elif isinstance(value, list):
                if value:
                    if isinstance(value[0], dict):
                        for item in value:
                            dict_shorten(item, schema[key][0]) # schema[key] should be a list with only one dict element
    return origin

def observation_shorten(schema_root, response_dict, category, tool_name, api_name, strip_method):
    print(random.random())
    if strip_method == "filter" or (strip_method == "random" and random.random() > 0.5):
        if isinstance(response_dict["response"], dict):
            if os.path.exists(os.path.join(schema_root, category)):
                if os.path.exists(os.path.join(schema_root, category, tool_name+".json")):
                    schema_dicts = json.load(open(os.path.join(schema_root, category, tool_name+".json"), "r"))
                    api_list = schema_dicts["api_list"]
                    schema = None
                    for schema_dict in api_list:
                        schema_api_name = change_name(standardize(schema_dict["name"]))
                        if schema_api_name == api_name and len(schema_dict["schema"]) > 0:
                            schema = schema_dict["schema"]
                            break
                    if schema is not None:
                        response_dict["response"] = dict_shorten(response_dict["response"], schema)
    return str(response_dict["response"])


def get_rapidapi_response(input_dict: dict, api_customization: bool=False, tools_root: str="data.toolenv.tools", schema_root: str="data/toolenv/response_examples"):
    info = Info
    info.category = input_dict['category']
    info.tool_name = input_dict['tool_name']
    info.api_name = input_dict['api_name']
    info.tool_input = input_dict['tool_input']
    info.strip = input_dict['strip']
    rapidapi_key = input_dict['rapidapi_key']

    tool_name, standard_category, api_name, code_string = prepare_tool_name_and_url(tools_root, info)
    tool_input = info.tool_input
    
    strip_method = info.strip
    
    try:
        tool_input = json.loads(tool_input)
    except Exception as e:
        if tool_input == "":
            tool_input = {}
        else:
            print(f"Can not parse tool input into json: {tool_input}")
            response_dict = {"error": f"Tool input parse error...\n", "response": ""}
            return response_dict
    
    input_params_str = ""
    if len(tool_input) > 0:
        for key, value in tool_input.items():
            if isinstance(value, str):
                input_params_str += f'{key}="{value}", '
            else:
                input_params_str += f'{key}={value}, '
    if not api_customization:
        input_params_str += f"toolbench_rapidapi_key='{rapidapi_key}'"
    success_flag, switch_flag, response_dict, save_cache = run(code_string, api_name, input_params_str)
    observation = observation_shorten(schema_root, response_dict, standard_category, tool_name.replace(f"_for_{standard_category}", ""), api_name, strip_method)
    result = str(observation)[:2048]
    return {"error": response_dict['error'], "response": result}


def query_rapidapi(
    payload,
    process_id,
    service_url,
    rapidapi_key,
    toolbench_key,
    use_rapidapi_key,
    api_customization
):
    if process_id == 0:
        print(colored(f"query to {payload['category']}-->{payload['tool_name']}-->{payload['api_name']}",color="yellow"))
    if use_rapidapi_key or api_customization:
        payload["rapidapi_key"] = rapidapi_key
        response = get_rapidapi_response(payload, api_customization=api_customization)
    else:
        time.sleep(2) # rate limit: 30 per minute
        headers = {"toolbench_key": toolbench_key}
        timeout = None if service_url.endswith("virtual") else 15
        try:
            response = requests.post(service_url, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.Timeout:
            return json.dumps({"error": f"Timeout error...", "response": ""}), 5
        if response.status_code != 200:
            return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
        try:
            response = response.json()
        except:
            # print(response)
            return json.dumps({"error": f"request invalid, data error", "response": ""}), 12
    
    if response["error"] == "API not working error...":
        status_code = 6
    elif response["error"] == "Unauthorized error...":
        status_code = 7
    elif response["error"] == "Unsubscribed error...":
        status_code = 8
    elif response["error"] == "Too many requests error...":
        status_code = 9
    elif response["error"] == "Rate limit per minute error...":
        print("Reach api calling limit per minute, sleeping...")
        time.sleep(10)
        status_code = 10
    elif response["error"] == "Message error...":
        status_code = 11
    else:
        status_code = 0
        return json.dumps(response), status_code
