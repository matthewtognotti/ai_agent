from openai import OpenAI
import json
import requests

# Following example from Open AI Documentation: https://platform.openai.com/docs/guides/function-calling?api-mode=responses

client = OpenAI()

def get_weather(latitude: float, longitude: float) -> float:
    """Get current weather for given coordinates.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Current temperature in Celsius
    """
    print("\n============================================================================")
    print(f"LLM invoked Weather Function with args: \n\n{args}")
    print("============================================================================\n")

    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&"
        "current=temperature_2m,wind_speed_10m&"
        "hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data['current']['temperature_2m']

def send_email(recipient_name: str, content: str) -> str:
    """Simulate sending an email.
    
    Args:
        recipient_name: Name of recipient
        content: Email content
        
    Returns:
        Success message
    """
    print("\n============================================================================")
    print(f"LLM invoked Email Function to {recipient_name} with content: \n\n{content}")
    print("============================================================================\n")
    return f"Email Server - Successfully Sent"

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
    }
]

def call_function(name: str, args: dict):
    """Call the appropriate function based on name.
    
    Args:
        name: Function name to call
        args: Arguments to pass to function
        
    Returns:
        Result of called function
    """

    function_dict = {
        "get_weather": get_weather,
        "send_email": send_email
    }

    if name not in function_dict:
        raise ValueError(f"Invalid function name: {name}")

    return function_dict[name](**args)

def llm_output(input_messages: list[dict[str, str]]) -> dict[str, str]:
    """Get LLM response for given messages.
    
    Args:
        input_messages: List of message dictionaries
        
    Returns:
        LLM response object
    """
    response = client.responses.create(
        model="gpt-4o",
        input=input_messages,
        tools=tools,
    )
    return response
        
input_messages = [
    {"role": "system", "content": "You are a helpful email and weather assistant. Always be enthusiastic."}
]

while True:
    print("\n")
    user_input = input("User: ")
    print("\n")
    
    # Append user message
    input_messages.append({"role": "user", "content": user_input})

    response_1 = llm_output(input_messages)

    # Append LLM response
    input_messages.append({"role": "system", "content": response_1.output_text})

    # Print LLM response in CLI for user if response is not a function call
    if response_1.output[0].type != "function_call":
        print("LLM: " + response_1.output_text)
    
    used_tool = False
    
    for tool_call in response_1.output:
        if tool_call.type != "function_call":
            continue
        
        used_tool = True
        name = tool_call.name
        args = json.loads(tool_call.arguments)
              
        result = call_function(name, args)
        
        input_messages.append(tool_call)
        input_messages.append({
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(result)
        })
    
    if used_tool:
        response_2 = llm_output(input_messages)
        input_messages.append({"role": "system", "content": response_2.output_text})
        print("LLM: " + response_2.output_text)
