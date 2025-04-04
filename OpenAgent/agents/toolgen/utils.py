import json
import re

import requests

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name

def get_toolbench_name(tool_name, api_name):
    tool_name = standardize(tool_name)
    api_name = change_name(standardize(api_name))
    toolbench_name = api_name+f"_for_{tool_name}"
    toolbench_name = toolbench_name[-64:]
    return toolbench_name


def toolgen_request(endpoint_url, query, system_prompt=None):
    payload = {
        "query": query,
        "system_prompt": system_prompt
    }

    try:
        response = requests.post(endpoint_url, json=payload, stream=True)  # Enable streaming
        response.raise_for_status()  # Raise an error for HTTP errors
        for line in response.iter_lines(decode_unicode=True): 
            if line:  # Filter out keep-alive new lines
                yield json.loads(line)  # Parse each line as JSON
    except requests.exceptions.RequestException as e:
        print(f"Error calling ToolGen model: {e}")
        yield {"error": str(e)}