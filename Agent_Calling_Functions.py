from openai import OpenAI
import json
import requests

### Following example from Open AI Documentation: https://platform.openai.com/docs/guides/function-calling?api-mode=responses

client = OpenAI()

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']

def send_email(recipient_name: str, content: str):
    print(f"Email Sever: LLM send email to {recipient_name}, with content: {content}")
    return f"Email Server: Sucessfully sent email to {recipient_name}"

tools = [
    {
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
    }, 
    {
    "type": "function",
    "name": "send_email",
    "description": "Send an email for the user. You don't need the email. Just the name of the recipient. Also, ALWAYS confirm with the user the contents of the email before you send it",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "recipient_name": {"type": "string"}
        },
        "required": ["content", "recipient_name"],
        "additionalProperties": False
    },
    "strict": True
    }]


def call_function(name, args):
    if name == "get_weather":
        print("System: LLM as invoked Weather Function")
        return get_weather(**args)
    if name == "send_email":
        print("System: LLM as invoked Function")
        result = send_email(**args)
        return result
    
def llm_output(input_messages):
    response = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        tools=tools,
    )
    return response
        
        
input_messages = []
input_messages.append({"role": "developer", "content": "You are a helpful email and weather assistant. Always be enthusiastic."})

while True:
    
    print("")
    user_input = input("Send message to the LLM:  ")
    print("")
    input_messages.append({"role": "user", "content": user_input}) ## append user message

    response_1 = llm_output(input_messages)
    input_messages.append({"role": "system", "content": response_1.output_text}) ## append LLM response
    input_messages.append({"role": "user", "content": user_input}) ## append user message
    print(response_1.output_text)                                  ## print for the user in CLI
    
    used_tool = False                                              ## to keep track if the LLM used a tool in the loop
    
    for tool_call in response_1.output:
        if tool_call.type != "function_call":
            continue
        
        used_tool = True
        name = tool_call.name
        args = json.loads(tool_call.arguments)
              
        result = call_function(name, args)
        
        input_messages.append(tool_call)    # append model's function call message
        input_messages.append({             # append result message
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result)
        })
    
    ## Now respond to the user only if the LLM used a tool
    if used_tool == True:
        response_2 = llm_output(input_messages)
        input_messages.append({"role": "system", "content": response_2.output_text}) ## append LLM response
        print(response_2.output_text)
    
