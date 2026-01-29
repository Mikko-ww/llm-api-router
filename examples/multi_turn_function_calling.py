"""
Advanced Multi-turn Function Calling Example

This example demonstrates a more complex use case: a travel assistant that can:
1. Search for flights
2. Get weather information
3. Find hotel recommendations
4. Calculate costs

The assistant orchestrates multiple function calls across several turns to help
the user plan their trip.
"""

import json
from typing import List, Dict, Any
from datetime import datetime, timedelta
from llm_api_router import Client, ProviderConfig, Tool, FunctionDefinition


# Mock functions that would typically call real APIs
def search_flights(origin: str, destination: str, date: str) -> Dict[str, Any]:
    """Search for available flights."""
    # Mock flight data
    flights = [
        {
            "airline": "United",
            "flight_number": "UA123",
            "departure": "08:00",
            "arrival": "11:00",
            "price": 299
        },
        {
            "airline": "Delta",
            "flight_number": "DL456",
            "departure": "14:00",
            "arrival": "17:00",
            "price": 349
        }
    ]
    
    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "flights": flights
    }


def get_weather_forecast(location: str, days: int = 7) -> Dict[str, Any]:
    """Get weather forecast for a location."""
    # Mock weather forecast
    forecasts = []
    base_temp = 20
    
    for i in range(min(days, 7)):
        date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
        forecasts.append({
            "date": date,
            "temperature": base_temp + (i % 3) * 2,
            "condition": ["Sunny", "Partly Cloudy", "Cloudy"][i % 3]
        })
    
    return {
        "location": location,
        "forecasts": forecasts
    }


def find_hotels(location: str, checkin: str, checkout: str, budget: str = "medium") -> Dict[str, Any]:
    """Find hotel recommendations."""
    # Mock hotel data
    budget_map = {
        "budget": [
            {"name": "Budget Inn", "rating": 3.5, "price_per_night": 80},
            {"name": "City Hostel", "rating": 4.0, "price_per_night": 60}
        ],
        "medium": [
            {"name": "Comfort Hotel", "rating": 4.2, "price_per_night": 150},
            {"name": "Downtown Suites", "rating": 4.5, "price_per_night": 180}
        ],
        "luxury": [
            {"name": "Grand Hotel", "rating": 4.8, "price_per_night": 350},
            {"name": "Luxury Resort", "rating": 4.9, "price_per_night": 450}
        ]
    }
    
    return {
        "location": location,
        "checkin": checkin,
        "checkout": checkout,
        "hotels": budget_map.get(budget, budget_map["medium"])
    }


def calculate_trip_cost(flights_cost: float, hotel_cost: float, num_nights: int, daily_expenses: float = 100) -> Dict[str, Any]:
    """Calculate total trip cost."""
    total_hotel = hotel_cost * num_nights
    total_daily = daily_expenses * num_nights
    total = flights_cost + total_hotel + total_daily
    
    return {
        "flights": flights_cost,
        "hotel": total_hotel,
        "daily_expenses": total_daily,
        "total": total,
        "currency": "USD"
    }


# Map function names to actual functions
AVAILABLE_FUNCTIONS = {
    "search_flights": search_flights,
    "get_weather_forecast": get_weather_forecast,
    "find_hotels": find_hotels,
    "calculate_trip_cost": calculate_trip_cost
}


def run_conversation(client: Client, tools: List[Tool], initial_message: str, max_turns: int = 10):
    """
    Run a multi-turn conversation with function calling.
    
    Args:
        client: LLM client
        tools: Available tools
        initial_message: User's initial message
        max_turns: Maximum number of turns to prevent infinite loops
    """
    messages = [
        {"role": "user", "content": initial_message}
    ]
    
    print(f"User: {initial_message}\n")
    print("-" * 80 + "\n")
    
    turn = 0
    while turn < max_turns:
        turn += 1
        print(f"Turn {turn}:")
        
        # Make API call
        response = client.chat.completions.create(
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        
        # If no tool calls, we're done
        if not assistant_message.tool_calls:
            print(f"Assistant: {assistant_message.content}\n")
            break
        
        # Model wants to call functions
        print(f"Model is calling {len(assistant_message.tool_calls)} function(s):")
        
        # Add assistant message to conversation
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
        
        # Execute each function call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"  - {function_name}({json.dumps(function_args)})")
            
            # Execute the function
            if function_name in AVAILABLE_FUNCTIONS:
                function_to_call = AVAILABLE_FUNCTIONS[function_name]
                function_result = function_to_call(**function_args)
                
                print(f"    Result: {json.dumps(function_result, indent=2)}")
                
                # Add function result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(function_result)
                })
            else:
                print(f"    Error: Function {function_name} not found")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"error": f"Function {function_name} not implemented"})
                })
        
        print()
    
    if turn >= max_turns:
        print(f"Reached maximum turns ({max_turns})")


def main():
    """Main function demonstrating advanced multi-turn function calling."""
    
    # Configure the client
    config = ProviderConfig(
        provider_type="openai",  # or "anthropic"
        api_key="your-api-key-here",
        default_model="gpt-4"
    )
    
    # Define all available tools
    tools = [
        Tool(
            type="function",
            function=FunctionDefinition(
                name="search_flights",
                description="Search for available flights between two cities",
                parameters={
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Origin city"},
                        "destination": {"type": "string", "description": "Destination city"},
                        "date": {"type": "string", "description": "Departure date (YYYY-MM-DD)"}
                    },
                    "required": ["origin", "destination", "date"]
                }
            )
        ),
        Tool(
            type="function",
            function=FunctionDefinition(
                name="get_weather_forecast",
                description="Get weather forecast for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "days": {"type": "integer", "description": "Number of days (1-7)"}
                    },
                    "required": ["location"]
                }
            )
        ),
        Tool(
            type="function",
            function=FunctionDefinition(
                name="find_hotels",
                description="Find hotel recommendations in a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "checkin": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                        "checkout": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                        "budget": {
                            "type": "string",
                            "enum": ["budget", "medium", "luxury"],
                            "description": "Budget level"
                        }
                    },
                    "required": ["location", "checkin", "checkout"]
                }
            )
        ),
        Tool(
            type="function",
            function=FunctionDefinition(
                name="calculate_trip_cost",
                description="Calculate total cost of a trip",
                parameters={
                    "type": "object",
                    "properties": {
                        "flights_cost": {"type": "number", "description": "Total cost of flights"},
                        "hotel_cost": {"type": "number", "description": "Cost per night for hotel"},
                        "num_nights": {"type": "integer", "description": "Number of nights"},
                        "daily_expenses": {"type": "number", "description": "Daily expense budget"}
                    },
                    "required": ["flights_cost", "hotel_cost", "num_nights"]
                }
            )
        )
    ]
    
    with Client(config) as client:
        # Example conversation
        user_query = (
            "I want to plan a 3-day trip from New York to San Francisco next week. "
            "Can you help me find flights, check the weather, suggest mid-range hotels, "
            "and calculate the total cost? My daily budget for food and activities is $150."
        )
        
        run_conversation(client, tools, user_query)


if __name__ == "__main__":
    print("=" * 80)
    print("Advanced Multi-turn Function Calling Example: Travel Assistant")
    print("=" * 80 + "\n")
    
    try:
        main()
    except Exception as e:
        print(f"\nNote: This example requires a valid API key.")
        print(f"Error: {e}")
        print("\nThe example demonstrates how the library handles:")
        print("- Multiple function calls in a single turn")
        print("- Multi-turn conversations with function results")
        print("- Complex orchestration of multiple APIs")
        print("- Error handling for missing functions")
