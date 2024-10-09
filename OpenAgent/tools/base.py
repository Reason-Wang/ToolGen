import json
from termcolor import colored

class BaseTool():
    def __init__(self, tools, tools_map, max_observation_length=1024):
        self.tools = {}
        for tool in tools:
            self.tools[tool['function']['name']] = tool
        # self.tools = tools
        self.tools_map = tools_map
        self.max_observation_length = max_observation_length
        self.success = 0

    def call(self, action_name, action_input):
        # print(f"Calling {action_name} with input: {action_input}")
        if action_name in self.tools:
            obs, code = self._call(action_name, action_input)
            if len(obs) > self.max_observation_length:
                obs = obs[:self.max_observation_length] + "..."
            return obs, code
        else:
            return {"error": f"No such tool name: {action_name}"}, 0

    def check_success(self):
        return self.success

    def _call(self, action_name, action_input):
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
        json_data = json.loads(action_input)
        if action_name in self.tools:
            function = self.tools_map[action_name]
            # print(function)
            print(colored(f"Querying: {action_name}", color="yellow"))
            
            response = function(**json_data)
        else:
            response = {
                "error": "invalid hallucation of function name."
            }
            status_code = 0
            return json.dumps(response), status_code

        if isinstance(response, dict) and "status_code" in response:
            status_code = response['status_code']
            del response['status_code']
            # whether generated the final answer
            if status_code == 3:
                self.success = 1
        else:
            status_code = 0

        return json.dumps(response), status_code
        
    def to_json(self):
        return {}
