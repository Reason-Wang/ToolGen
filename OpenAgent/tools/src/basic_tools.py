

import json


BasicTools = [
    # Finish function
    {
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
]


TestTools = [
    {
      "type": "function",
      "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature for a specific location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g., San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["Celsius", "Fahrenheit"],
              "description": "The temperature unit to use. Infer this from the user's location."
            }
          },
          "required": ["location", "unit"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_rain_probability",
        "description": "Get the probability of rain for a specific location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g., San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    },
    {
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
]

def finish(return_type=None, final_answer=None):

      if return_type is None:
          response =  {"error": "must have \"return_type\""}
          status = 2
      if return_type == "give_up_and_restart":
          response = {"response": "chose to give up and restart"}
          status = 4
      elif return_type == "give_answer":
          if final_answer is None:
              response = {"error": "must have \"final_answer\""}
              status = 2
          else:
              response = {"response": "successfully giving the final answer."}
              status = 3
      else:
          response = {"error": "\"return_type\" is not a valid choice\""}
          status = 2

      response['status_code'] = status
      return response


def get_temperature(location, unit):
    return 75

def get_rain_probability(location):
    return 0.2

TestToolsMap = {
    "get_current_temperature": get_temperature,
    "get_rain_probability": get_rain_probability,
    "Finish": finish
}
