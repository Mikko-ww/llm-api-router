"""
Example demonstrating error handling and retry mechanisms.

This example shows how to:
1. Configure retry behavior
2. Handle different exception types
3. Use timeout configuration
"""

from llm_api_router import Client, ProviderConfig, RetryConfig
from llm_api_router.exceptions import (
    AuthenticationError,
    RateLimitError,
    RetryExhaustedError,
    TimeoutError,
    LLMRouterError
)


def example_with_custom_retry():
    """Example with custom retry configuration."""
    print("=== Example 1: Custom Retry Configuration ===\n")
    
    # Configure retry with aggressive settings
    retry_config = RetryConfig(
        max_retries=5,
        initial_delay=0.5,
        max_delay=30.0,
        exponential_base=2.0
    )
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo",
        retry_config=retry_config,
        timeout=30.0  # 30 second timeout
    )
    
    print(f"Retry configuration:")
    print(f"  Max retries: {retry_config.max_retries}")
    print(f"  Initial delay: {retry_config.initial_delay}s")
    print(f"  Max delay: {retry_config.max_delay}s")
    print(f"  Exponential base: {retry_config.exponential_base}")
    print(f"  Retry on status codes: {retry_config.retry_on_status_codes}")
    print(f"  Request timeout: {config.timeout}s\n")


def example_with_no_retry():
    """Example with retries disabled."""
    print("=== Example 2: Retries Disabled ===\n")
    
    # Disable retries entirely
    retry_config = RetryConfig(max_retries=0)
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        retry_config=retry_config
    )
    
    print(f"Retries disabled (max_retries=0)")
    print("Errors will fail immediately without retry\n")


def example_error_handling():
    """Example demonstrating comprehensive error handling."""
    print("=== Example 3: Comprehensive Error Handling ===\n")
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo"
    )
    
    try:
        with Client(config) as client:
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            )
            print(f"Response: {response.choices[0].message.content}\n")
            
    except AuthenticationError as e:
        # Handle authentication failures (won't be retried)
        print(f"❌ Authentication failed: {e.message}")
        print(f"   Provider: {e.provider}")
        print(f"   Status code: {e.status_code}")
        print("   Action: Check your API key\n")
        
    except RateLimitError as e:
        # This typically won't be raised because rate limits are retried
        print(f"⚠️  Rate limit error: {e.message}")
        print(f"   Status code: {e.status_code}")
        print("   Note: This should have been retried automatically\n")
        
    except RetryExhaustedError as e:
        # Raised when all retry attempts fail
        print(f"❌ All retry attempts failed: {e.message}")
        print(f"   Max retries: {e.details.get('max_retries')}")
        print(f"   Original error: {e.details.get('original_error')}")
        print(f"   Error type: {e.details.get('error_type')}\n")
        
    except TimeoutError as e:
        # Request timeout
        print(f"⏱️  Request timeout: {e.message}")
        print(f"   Provider: {e.provider}")
        print("   Action: Increase timeout or check network\n")
        
    except LLMRouterError as e:
        # Catch-all for any other errors
        print(f"❌ Error: {e.message}")
        print(f"   Provider: {e.provider}")
        print(f"   Status code: {e.status_code}")
        print(f"   Details: {e.details}\n")


def example_streaming_with_error_handling():
    """Example demonstrating streaming with error handling."""
    print("=== Example 4: Streaming with Error Handling ===\n")
    
    from llm_api_router.exceptions import StreamError
    
    config = ProviderConfig(
        provider_type="openai",
        api_key="your-api-key-here",
        default_model="gpt-3.5-turbo"
    )
    
    try:
        with Client(config) as client:
            stream = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "Write a short poem"}
                ],
                stream=True
            )
            
            print("Streaming response:")
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
            print("\n")
            
    except StreamError as e:
        print(f"❌ Stream parsing error: {e.message}")
        print(f"   Provider: {e.provider}")
        print(f"   Details: {e.details}\n")
        
    except LLMRouterError as e:
        print(f"❌ Error during streaming: {e.message}\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("LLM API Router - Error Handling Examples")
    print("="*60 + "\n")
    
    # Example 1: Custom retry configuration
    example_with_custom_retry()
    
    # Example 2: Disable retries
    example_with_no_retry()
    
    # Example 3: Comprehensive error handling
    print("To run example 3, uncomment the following line:")
    print("# example_error_handling()")
    print()
    
    # Example 4: Streaming with error handling
    print("To run example 4, uncomment the following line:")
    print("# example_streaming_with_error_handling()")
    print()
    
    print("="*60)
    print("Note: Replace 'your-api-key-here' with actual API keys")
    print("to run the error handling examples.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
