"""
Function Calling Example

This example demonstrates how to use function calling (tool use) with the llm-api-router library.
Function calling allows the model to intelligently call functions to retrieve information or 
perform actions.

This example shows:
1. Defining tools/functions for the model to use
2. Making a request with tools
3. Parsing the model's tool call
4. Executing the function
5. Sending the function result back to the model
"""

import json
from llm_api_router import Client, ProviderConfig, Tool, FunctionDefinition


# Define a simple weather function
def get_weather(location: str, unit: str = "celsius") -> dict:
    """
    Get the current weather for a location.
    In a real application, this would call a weather API.
    """
    # Mock weather data
    weather_data = {
        "San Francisco, CA": {"temperature": 20, "condition": "Sunny"},
        "New York, NY": {"temperature": 15, "condition": "Cloudy"},
        "London, UK": {"temperature": 10, "condition": "Rainy"},
        "Tokyo, Japan": {"temperature": 25, "condition": "Clear"},
    }
    
    data = weather_data.get(location, {"temperature": 22, "condition": "Unknown"})
    
    # Convert temperature if needed
    if unit == "fahrenheit":
        data["temperature"] = data["temperature"] * 9/5 + 32
    
    return {
        "location": location,
        "temperature": data["temperature"],
        "unit": unit,
        "condition": data["condition"]
    }


def main():
    """Main function demonstrating function calling."""
    
    # Configure the client - works with OpenAI, Anthropic, or any compatible provider
    config = ProviderConfig(
        provider_type="openai",  # or "anthropic" for Claude
        api_key="your-api-key-here",
        default_model="gpt-4"  # or "claude-3-5-sonnet-20240620" for Anthropic
    )
    
    # Define the tools available to the model
    tools = [
        Tool(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather in a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Default is celsius."
                        }
                    },
                    "required": ["location"]
                }
            )
        )
    ]
    
    # Initial conversation
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ]
    
    with Client(config) as client:
        print("User: What's the weather like in San Francisco?\n")
        
        # First request: Model decides to call the weather function
        response = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice="auto"  # Let model decide when to use tools
        )
        
        # Check if model wants to call a function
        assistant_message = response.choices[0].message
        
        if assistant_message.tool_calls:
            print("Model wants to call functions:")
            
            # Add assistant's message to conversation
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            
            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"  - {function_name}({json.dumps(function_args, indent=4)})")
                
                # Call the actual function
                if function_name == "get_weather":
                    function_result = get_weather(**function_args)
                    print(f"    Result: {json.dumps(function_result, indent=4)}\n")
                    
                    # Add function result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result)
                    })
            
            # Second request: Send function results back to model
            print("Sending function results back to model...\n")
            final_response = client.chat.completions.create(
                messages=messages,
                tools=tools
            )
            
            # Print the final answer
            final_message = final_response.choices[0].message
            print(f"Assistant: {final_message.content}\n")
        
        else:
            # Model responded without calling functions
            print(f"Assistant: {assistant_message.content}\n")


def multi_turn_example():
    """
    Example of multi-turn conversation with function calling.
    Shows how to handle multiple function calls in a conversation.
    """
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-4"
    )
    
    tools = [
        Tool(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather in a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state"
                        }
                    },
                    "required": ["location"]
                }
            )
        )
    ]
    
    messages = [
        {"role": "user", "content": "What's the weather in San Francisco and New York?"}
    ]
    
    with Client(config) as client:
        print("User: What's the weather in San Francisco and New York?\n")
        
        # The model might make multiple function calls
        response = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        if assistant_message.tool_calls:
            print(f"Model making {len(assistant_message.tool_calls)} function call(s)\n")
            
            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            
            # Process all tool calls
            for tool_call in assistant_message.tool_calls:
                function_args = json.loads(tool_call.function.arguments)
                function_result = get_weather(**function_args)
                
                print(f"Getting weather for {function_args['location']}...")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_result)
                })
            
            # Get final response
            print("\nSending results back to model...\n")
            final_response = client.chat.completions.create(
                messages=messages,
                tools=tools
            )
            
            print(f"Assistant: {final_response.choices[0].message.content}\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Basic Function Calling Example")
    print("=" * 80 + "\n")
    
    try:
        main()
    except Exception as e:
        print(f"Note: This example requires a valid API key. Error: {e}\n")
    
    print("\n" + "=" * 80)
    print("Multi-turn Function Calling Example")
    print("=" * 80 + "\n")
    
    try:
        multi_turn_example()
    except Exception as e:
        print(f"Note: This example requires a valid API key. Error: {e}\n")
