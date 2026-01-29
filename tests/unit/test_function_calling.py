"""Unit tests for function calling support."""
import pytest
import json
from unittest.mock import Mock, MagicMock
import httpx

from llm_api_router.providers.openai import OpenAIProvider
from llm_api_router.providers.anthropic import AnthropicProvider
from llm_api_router.types import (
    ProviderConfig, UnifiedRequest, Tool, FunctionDefinition,
    ToolCall, FunctionCall, Message
)
from tests.fixtures.mock_responses import (
    get_openai_function_call_response,
    get_anthropic_function_call_response
)


class TestFunctionCallingTypes:
    """Test function calling data structures."""
    
    def test_function_definition_creation(self):
        """Test FunctionDefinition creation."""
        func_def = FunctionDefinition(
            name="get_weather",
            description="Get the current weather",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        )
        
        assert func_def.name == "get_weather"
        assert func_def.description == "Get the current weather"
        assert "location" in func_def.parameters["properties"]
    
    def test_tool_creation(self):
        """Test Tool creation."""
        func_def = FunctionDefinition(
            name="get_weather",
            description="Get the current weather",
            parameters={"type": "object", "properties": {}}
        )
        
        tool = Tool(type="function", function=func_def)
        
        assert tool.type == "function"
        assert tool.function.name == "get_weather"
    
    def test_function_call_creation(self):
        """Test FunctionCall creation."""
        func_call = FunctionCall(
            name="get_weather",
            arguments='{"location": "San Francisco"}'
        )
        
        assert func_call.name == "get_weather"
        assert json.loads(func_call.arguments)["location"] == "San Francisco"
    
    def test_tool_call_creation(self):
        """Test ToolCall creation."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "San Francisco"}'
            )
        )
        
        assert tool_call.id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
    
    def test_message_with_tool_calls(self):
        """Test Message with tool_calls."""
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(
                name="get_weather",
                arguments='{"location": "San Francisco"}'
            )
        )
        
        message = Message(
            role="assistant",
            content=None,
            tool_calls=[tool_call]
        )
        
        assert message.role == "assistant"
        assert message.content is None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "call_123"


class TestOpenAIFunctionCalling:
    """Test OpenAI provider function calling."""
    
    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return ProviderConfig(
            provider_type="openai",
            api_key="sk-test-key",
            default_model="gpt-4"
        )
    
    @pytest.fixture
    def openai_provider(self, openai_config):
        """OpenAI provider instance."""
        return OpenAIProvider(openai_config)
    
    def test_convert_tools_to_openai(self, openai_provider):
        """Test converting tools to OpenAI format."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get the current weather in a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    },
                    strict=True
                )
            )
        ]
        
        converted = openai_provider._convert_tools_to_openai(tools)
        
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "get_weather"
        assert converted[0]["function"]["strict"] is True
        assert "location" in converted[0]["function"]["parameters"]["properties"]
    
    def test_convert_request_with_tools(self, openai_provider):
        """Test converting request with tools."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {}}
                )
            )
        ]
        
        request = UnifiedRequest(
            messages=[{"role": "user", "content": "What's the weather in SF?"}],
            model="gpt-4",
            tools=tools,
            tool_choice="auto"
        )
        
        converted = openai_provider.convert_request(request)
        
        assert "tools" in converted
        assert len(converted["tools"]) == 1
        assert converted["tool_choice"] == "auto"
    
    def test_parse_tool_calls_from_response(self, openai_provider):
        """Test parsing tool calls from OpenAI response."""
        tool_calls_data = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco, CA"}'
                }
            }
        ]
        
        parsed = openai_provider._parse_tool_calls(tool_calls_data)
        
        assert len(parsed) == 1
        assert parsed[0].id == "call_abc123"
        assert parsed[0].type == "function"
        assert parsed[0].function.name == "get_weather"
        assert "location" in parsed[0].function.arguments
    
    def test_convert_response_with_tool_calls(self, openai_provider):
        """Test converting response with tool calls."""
        provider_response = get_openai_function_call_response()
        
        unified_response = openai_provider.convert_response(provider_response)
        
        assert unified_response.id == "chatcmpl-function-test"
        assert len(unified_response.choices) == 1
        
        message = unified_response.choices[0].message
        assert message.role == "assistant"
        assert message.content is None
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        
        tool_call = message.tool_calls[0]
        assert tool_call.id == "call_abc123"
        assert tool_call.function.name == "get_weather"
        assert "San Francisco" in tool_call.function.arguments
        
        assert unified_response.choices[0].finish_reason == "tool_calls"


class TestAnthropicFunctionCalling:
    """Test Anthropic provider function calling."""
    
    @pytest.fixture
    def anthropic_config(self):
        """Anthropic provider configuration."""
        return ProviderConfig(
            provider_type="anthropic",
            api_key="sk-ant-test-key",
            default_model="claude-3-5-sonnet-20240620"
        )
    
    @pytest.fixture
    def anthropic_provider(self, anthropic_config):
        """Anthropic provider instance."""
        return AnthropicProvider(anthropic_config)
    
    def test_convert_tools_to_anthropic(self, anthropic_provider):
        """Test converting tools to Anthropic format."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get the current weather in a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and state"
                            }
                        },
                        "required": ["location"]
                    }
                )
            )
        ]
        
        converted = anthropic_provider._convert_tools_to_anthropic(tools)
        
        assert len(converted) == 1
        assert converted[0]["name"] == "get_weather"
        assert "input_schema" in converted[0]
        assert "location" in converted[0]["input_schema"]["properties"]
    
    def test_convert_tool_choice_to_anthropic(self, anthropic_provider):
        """Test converting tool_choice to Anthropic format."""
        # Test "auto"
        result = anthropic_provider._convert_tool_choice_to_anthropic("auto")
        assert result["type"] == "auto"
        
        # Test "required"
        result = anthropic_provider._convert_tool_choice_to_anthropic("required")
        assert result["type"] == "any"
        
        # Test "any"
        result = anthropic_provider._convert_tool_choice_to_anthropic("any")
        assert result["type"] == "any"
    
    def test_convert_request_with_tools(self, anthropic_provider):
        """Test converting request with tools."""
        tools = [
            Tool(
                type="function",
                function=FunctionDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {}}
                )
            )
        ]
        
        request = UnifiedRequest(
            messages=[{"role": "user", "content": "What's the weather in SF?"}],
            model="claude-3-5-sonnet-20240620",
            tools=tools,
            tool_choice="auto"
        )
        
        converted = anthropic_provider.convert_request(request)
        
        assert "tools" in converted
        assert len(converted["tools"]) == 1
        assert "tool_choice" in converted
        assert converted["tool_choice"]["type"] == "auto"
    
    def test_convert_response_with_tool_use(self, anthropic_provider):
        """Test converting response with tool use."""
        provider_response = get_anthropic_function_call_response()
        
        unified_response = anthropic_provider.convert_response(provider_response)
        
        assert unified_response.id == "msg_tool_test"
        assert len(unified_response.choices) == 1
        
        message = unified_response.choices[0].message
        assert message.role == "assistant"
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        
        tool_call = message.tool_calls[0]
        assert tool_call.id == "toolu_01A09q90qw90lq917835lq9"
        assert tool_call.function.name == "get_weather"
        
        # Arguments should be JSON string
        args = json.loads(tool_call.function.arguments)
        assert args["location"] == "San Francisco, CA"
        
        assert unified_response.choices[0].finish_reason == "tool_use"
