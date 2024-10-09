import json
import re

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


def finish(action_input):
    try:
        json_data = json.loads(action_input, strict=False)
    except:
        json_data = {}
        if '"return_type": "' in action_input:
            if '"return_type": "give_answer"' in action_input:
                return_type = "give_answer"
            elif '"return_type": "give_up_and_restart"' in action_input:
                return_type = "give_up_and_restart"
            else:
                return_type = action_input[action_input.find('"return_type": "')+len('"return_type": "'):action_input.find('",')]
            json_data["return_type"] = return_type
        if '"final_answer": "' in action_input:
            final_answer = action_input[action_input.find('"final_answer": "')+len('"final_answer": "'):]
            json_data["final_answer"] = final_answer
    if "return_type" not in json_data.keys():
        return "{error:\"must have \"return_type\"\"}", 2
    if json_data["return_type"] == "give_up_and_restart":
        return "{\"response\":\"chose to give up and restart\"}",4
    elif json_data["return_type"] == "give_answer":
        if "final_answer" not in json_data.keys():
            return "{error:\"must have \"final_answer\"\"}", 2
        
        return "{\"response\":\"successfully giving the final answer.\"}", 3
    else:
        return "{error:\"\"return_type\" is not a valid choice\"}", 2

