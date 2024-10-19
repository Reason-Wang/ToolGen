import os
import json
from tqdm import tqdm
from .utils import standardize, change_name, finish
from .server import query_rapidapi
from huggingface_hub import hf_hub_download

# For pipeline environment preparation
def get_white_list(tool_root_dir):
    # print(tool_root_dir)
    white_list_dir = os.path.join(tool_root_dir)
    white_list = {}
    for cate in tqdm(os.listdir(white_list_dir)):
        if not os.path.isdir(os.path.join(white_list_dir,cate)):
            continue
        for file in os.listdir(os.path.join(white_list_dir,cate)):
            if not file.endswith(".json"):
                continue
            standard_tool_name = file.split(".")[0]
            # with open("data/test_names.txt", 'a') as f:
            #     f.write(standard_tool_name + '\n')
            # print(standard_tool_name)
            with open(os.path.join(white_list_dir,cate,file)) as reader:
                js_data = json.load(reader)
            origin_tool_name = js_data["tool_name"]
            white_list[standardize(origin_tool_name)] = {"description": js_data["tool_description"], "standard_tool_name": standard_tool_name}
    return white_list

def contain(candidate_list, white_list):
    output = []
    for cand in candidate_list:
        if cand not in white_list.keys():
            return False
        output.append(white_list[cand])
    return output
    

def fetch_api_json_from_tool(tool_json):
    data_dict = {"api_list":[]}
    cate_name = tool_json["category_name"]
    # tool_name = standardize(tool_json["tool_name"])
    api_dict_names = []
    for api_dict in tool_json["api_list"]:
        api_dict_names.append(api_dict["name"])
        # pure_api_name = change_name(standardize(api_dict["name"]))
        # if pure_api_name != api_name:
            # continue
        api_json = {}
        api_json["category_name"] = cate_name
        api_json["api_name"] = api_dict["name"]
        api_json["api_description"] = api_dict["description"]
        api_json["required_parameters"] = api_dict["required_parameters"]
        api_json["optional_parameters"] = api_dict["optional_parameters"]
        api_json["tool_name"] = tool_json["tool_name"]
        data_dict["api_list"].append(api_json)

    return data_dict


def api_json_to_openai_json(api_json, standard_tool_name):
    description_max_length=256
    function_templete = {
        "type": "function",
        "function": {
            "name": "",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": [""],
                "optional": [""],
            }
        }
    }
    templete = function_templete['function']
    
    map_type = {
        "NUMBER": "integer",
        "STRING": "string",
        "BOOLEAN": "boolean"
    }

    pure_api_name = change_name(standardize(api_json["api_name"]))
    templete["name"] = pure_api_name+ f"_for_{standard_tool_name}"
    # if "onboarding_project" in templete["name"]:
    #     print(templete["name"])
    templete["name"] = templete["name"][-64:]

    templete["description"] = f"This is the subfunction for tool \"{standard_tool_name}\", you can use this tool."
    
    if api_json["api_description"].strip() != "":
        tuncated_description = api_json['api_description'].strip().replace(api_json['api_name'],templete['name'])[:description_max_length]
        templete["description"] = templete["description"] + f"The description of this function is: \"{tuncated_description}\""
    if "required_parameters" in api_json.keys() and len(api_json["required_parameters"]) > 0:
        for para in api_json["required_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"
            prompt = {
                "type":param_type,
                "description":para["description"][:description_max_length],
            }

            default_value = para['default']
            if len(str(default_value)) != 0:    
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["required"].append(name)
        for para in api_json["optional_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"

            default_value = para['default']
            if len(str(default_value)) != 0:    
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["optional"].append(name)

    return function_templete, api_json["category_name"],  pure_api_name

def load_tool_package():
    tools_path = hf_hub_download(repo_id="reasonwang/ToolGen-Llama-3-8B", filename="tools.json")
    with open(tools_path, 'r') as f:
        all_tools = json.load(f)

    data_dict = {'api_list':[]}
    for tool in all_tools:
        data_dict['api_list'].extend(fetch_api_json_from_tool(tool)["api_list"])

    functions = {}
    api_name_reflect = {}
    tool_names = {}
    cate_names = {}
    for k,api_json in enumerate(data_dict['api_list']):
        standard_tool_name = standardize(api_json["tool_name"])
        openai_function_json, cate_name, pure_api_name = api_json_to_openai_json(api_json, standard_tool_name)
        functions[openai_function_json['function']["name"]] = openai_function_json

        api_name_reflect[openai_function_json['function']["name"]] = pure_api_name
        tool_names[openai_function_json['function']["name"]] = standard_tool_name
        cate_names[openai_function_json['function']["name"]] = cate_name
    all_tool_package = {
        "functions": functions,
        "api_name_reflect": api_name_reflect,
        "tool_names": tool_names,
        "cate_names": cate_names,
        "data_dict": data_dict
    }

    return all_tool_package


# rapidapi env wrapper
class RapidAPIWrapper:
    def __init__(
        self,
        toolbench_key,
        rapidapi_key,
        use_rapidapi_key=False,
        api_customization=False,
        max_observation_length=1024,
        observ_compress_method="truncate",
        process_id=0
    ):
        self.toolbench_key = toolbench_key
        self.rapidapi_key = rapidapi_key
        self.use_rapidapi_key = use_rapidapi_key
        self.api_customization = api_customization
        # StableToolBench url
        self.service_url = os.getenv("SERVICE_URL", "http://0.0.0.0:8080/virtual")
        self.max_observation_length = max_observation_length
        self.observ_compress_method = observ_compress_method
        self.process_id = process_id

        self.tool_names = {}
        self.cate_names = {}
        self.functions = {}
        self.api_name_reflect = {}


        tool_package = load_tool_package()
        data_dict = tool_package["data_dict"]
        print("using all tools")
        
        self.functions = tool_package["functions"]
        self.api_name_reflect = tool_package["api_name_reflect"]
        self.tool_names = tool_package["tool_names"]
        self.cate_names = tool_package["cate_names"]
        # We define tools simply for compatibility with the parse method
        self.tools = data_dict['api_list']

        finish_func = {
            "type": "function",
            "function": {
                "name": "Finish",
                "description": "If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "return_type": {
                            "type": "string",
                            "enum": ["give_answer","give_up_and_restart"],
                        },
                        "final_answer": {
                            "type": "string",
                            "description": "The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"",
                        }
                    },
                    "required": ["return_type"],
                },
            }
        }

        self.functions['Finish'] = finish_func
        self.CALL_MAX_TIME = 3

        self.success = 0

    def check_success(self):
        return self.success

    def to_json(self):
        return {}

    def restart(self):
        pass

    def get_score(self):
        return 0.0

    def call(self,**args):
        obs, code = self._call(**args)
        if len(obs) > self.max_observation_length:
            obs = obs[:self.max_observation_length] + "..."
        return obs, code

    def _call(self, action_name="", action_input=""):
        """Need to return an observation string and status code:
            0 means normal response
            1 means there is no corresponding api name
            2 means there is an error in the input
            3 represents the end of the generation and the final answer appears
            4 means that the model decides to pruning by itself
            5 represents api call timeout
            6 for 404
            7 means not subscribed
            8 represents unauthorized
            9 represents too many requests
            10 stands for rate limit
            11 message contains "error" field
            12 error sending request
        """
        if action_name == "Finish":
            response, status_code = finish(action_input)
            if status_code == 3:
                self.success = 1
            return response, status_code
        else:
            if action_name in self.functions:
                function = self.functions[action_name]
                pure_api_name = self.api_name_reflect[function['function']["name"]]
                payload = {
                    "category": self.cate_names[action_name],
                    "tool_name": self.tool_names[action_name],
                    "api_name": pure_api_name,
                    "tool_input": action_input,
                    "strip": self.observ_compress_method,
                    "toolbench_key": self.toolbench_key
                }

                response, status_code = query_rapidapi(
                    payload, 
                    process_id=self.process_id,
                    service_url=self.service_url,
                    rapidapi_key=self.rapidapi_key,
                    toolbench_key=self.toolbench_key,
                    use_rapidapi_key=self.use_rapidapi_key, 
                    api_customization=self.api_customization
                )
                return response, status_code
        
            return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1