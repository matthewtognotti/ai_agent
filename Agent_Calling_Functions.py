from openai import OpenAI
import json
import requests
from typing import List, Dict, Any, Union
from dataclasses import dataclass

# Configuration for OpenAI API
OPENAI_MODEL = "gpt-4"

@dataclass
class Message:
    role: str
    content: str

def get_weather(latitude: float, longitude: float) -> float:
    """Get current temperature for provided coordinates in celsius."""
    try:
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m",
                "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data['current']['temperature_2m']
    except (requests.RequestException, KeyError) as e:
        print(f"Error fetching weather data: {e}")
        return None

def send_email(recipient_name: str, content: str) -> str:
    """Simulate sending an email and return confirmation message."""
    return f"Email Server: Successfully sent email to {recipient_name}"

# Define available tools
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
        "description": "Send an email for the user. You don't need the email. Just the name of the recipient. Also, always show the user what email you sent",
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

def call_function(name: str, args: Dict[str, Any]) -> Any:
    """Execute the specified function with given arguments."""
    function_map = {
        "get_weather": get_weather,
        "send_email": send_email
    }
    
    if name not in function_map:
        print(f"Error: Unknown function {name}")
        return None
    
    print(f"System: LLM has invoked {name} function")
    try:
        return function_map[name](**args)
    except Exception as e:
        print(f"Error executing {name}: {e}")
        return None

def llm_output(messages: List[Dict[str, str]]) -> Any:
    """Get response from OpenAI API."""
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=tools
        )
        return response
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        return None

def main():
    """Main interaction loop."""
    messages = [
        {"role": "system", "content": "You are a helpful email and weather assistant. Always be enthusiastic."}
    ]

    while True:
        try:
            print("\n")
            user_input = input("Send message to the LLM: ").strip()
            if not user_input:
                continue
                
            print("\n")
            messages.append({"role": "user", "content": user_input})

            # Get initial response
            response = llm_output(messages)
            if not response:
                continue

            # Process the response
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            print(response.choices[0].message.content)

            # Handle any tool calls
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    result = call_function(name, args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": str(result)
                    })
                
                # Get follow-up response after tool usage
                follow_up = llm_output(messages)
                if follow_up:
                    messages.append({"role": "assistant", "content": follow_up.choices[0].message.content})
                    print(follow_up.choices[0].message.content)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
