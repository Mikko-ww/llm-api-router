# Testing Framework Documentation

## Overview

This testing framework provides comprehensive test coverage for the llm-api-router library, including unit tests, integration tests, and continuous integration.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                         # Shared fixtures and configuration
├── fixtures/
│   ├── __init__.py
│   └── mock_responses.py              # Mock API responses for various providers
├── unit/
│   ├── __init__.py
│   ├── test_client.py                 # Tests for Client and AsyncClient
│   ├── test_exceptions.py             # Tests for exception classes
│   ├── test_factory.py                # Tests for ProviderFactory
│   ├── test_provider_openai.py        # Tests for OpenAI provider
│   └── test_types.py                  # Tests for data types
└── integration/
    ├── __init__.py
    └── test_client_integration.py     # End-to-end integration tests
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test modules
```bash
pytest tests/unit/test_client.py
pytest tests/integration/
```

### Run with coverage report
```bash
pytest tests/ --cov=llm_api_router --cov-report=html --cov-report=term
```

### Run specific test by name
```bash
pytest tests/unit/test_client.py::TestClient::test_client_initialization -v
```

## Test Coverage

Current test coverage:

- **Overall Coverage**: 21%
- **Core Modules**: 100%
  - `client.py`: 100%
  - `types.py`: 100%
  - `exceptions.py`: 100%
  - `factory.py`: 100%
  - `base.py`: 100%
- **Providers**:
  - `openai.py`: 60%
  - `deepseek.py`: 37%
  - Other providers: Not yet tested

### Coverage Goals

- **Target**: 80%+ overall coverage
- **Priority**: Increase coverage for remaining providers

## Test Categories

### Unit Tests (67 tests)

1. **Types Tests** (12 tests)
   - ProviderConfig creation and defaults
   - Message structure
   - UnifiedRequest/Response
   - Usage, Choice, and Chunk types

2. **Exceptions Tests** (16 tests)
   - LLMRouterError base class
   - AuthenticationError
   - RateLimitError
   - ProviderError
   - StreamError
   - Error inheritance and catching

3. **Client Tests** (16 tests)
   - Client initialization
   - Context manager usage
   - Completions creation (streaming and non-streaming)
   - AsyncClient operations
   - Parameter passing

4. **Factory Tests** (10 tests)
   - Provider mapping
   - Provider registration
   - Dynamic loading
   - Error handling

5. **OpenAI Provider Tests** (13 tests)
   - Provider initialization
   - Request conversion
   - Response conversion
   - Error handling (401, 429, 500)
   - Custom configuration

### Integration Tests (11 tests)

1. **Client Integration**
   - OpenAI chat completion
   - DeepSeek integration
   - Model switching
   - Error handling

2. **AsyncClient Integration**
   - Async operations
   - Parameter passing
   - Error handling

3. **Multi-Provider**
   - Parameterized tests for different providers
   - Configuration variations

4. **Configuration**
   - Custom base URLs
   - Extra headers

## Continuous Integration

### GitHub Actions Workflow

The CI workflow (`.github/workflows/tests.yml`) runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

#### Test Matrix
- Python versions: 3.10, 3.11, 3.12
- Operating system: Ubuntu Latest

#### Jobs

1. **Test Job**
   - Install dependencies
   - Run tests with coverage
   - Upload coverage to Codecov

2. **Lint Job**
   - Type checking with mypy

## Fixtures and Mocks

### Common Fixtures (in `conftest.py`)

- `sample_provider_config`: Basic provider configuration
- `sample_messages`: Sample message list
- `sample_unified_response`: Complete response object
- `mock_http_response_success`: Mock HTTP response data
- `mock_stream_chunks`: Mock SSE stream data

### Mock Responses (in `fixtures/mock_responses.py`)

Mock API responses for:
- OpenAI
- Anthropic
- Gemini
- DeepSeek

Sample message templates:
- `SAMPLE_MESSAGES`: Basic user message
- `SAMPLE_MESSAGES_WITH_SYSTEM`: With system prompt
- `SAMPLE_MESSAGES_CONVERSATION`: Multi-turn conversation

## Writing New Tests

### Unit Test Example

```python
def test_new_feature(sample_provider_config):
    """Test description."""
    # Arrange
    client = Client(sample_provider_config)
    
    # Act
    result = client.some_method()
    
    # Assert
    assert result is not None
```

### Integration Test Example

```python
@pytest.mark.asyncio
async def test_new_integration():
    """Test async integration."""
    config = ProviderConfig(provider_type="openai", api_key="test-key")
    
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(return_value=get_openai_chat_response())
        )
        
        async with AsyncClient(config) as client:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}]
            )
            assert response is not None
```

## Best Practices

1. **Use Fixtures**: Leverage shared fixtures from `conftest.py`
2. **Mock External Calls**: Always mock HTTP requests in tests
3. **Test Both Sync and Async**: Cover both synchronous and asynchronous code paths
4. **Test Error Cases**: Include tests for error handling
5. **Descriptive Names**: Use clear, descriptive test names
6. **Arrange-Act-Assert**: Follow the AAA pattern
7. **Isolation**: Tests should be independent and not rely on execution order

## Adding Provider Tests

To add tests for a new provider:

1. Create mock response in `tests/fixtures/mock_responses.py`
2. Add unit tests in `tests/unit/test_provider_<name>.py`
3. Add integration tests in `tests/integration/test_client_integration.py`

Example structure:
```python
class TestNewProvider:
    def test_provider_initialization(self):
        # Test provider setup
        pass
    
    def test_convert_request(self):
        # Test request conversion
        pass
    
    def test_convert_response(self):
        # Test response conversion
        pass
    
    def test_error_handling(self):
        # Test error cases
        pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure package is installed with `pip install -e ".[dev]"`
2. **Async Test Failures**: Make sure to use `@pytest.mark.asyncio` decorator
3. **Mock Issues**: Use `Mock()` for sync, but return values directly for async mocks
4. **Coverage Not Updating**: Delete `.coverage` and `htmlcov/` directories

### Debug Commands

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run specific test with full traceback
pytest tests/unit/test_client.py::TestClient::test_name -vv
```

## Future Improvements

1. Add tests for remaining providers (Anthropic, Gemini, Zhipu, etc.)
2. Add streaming tests (both sync and async)
3. Add performance/load tests
4. Add tests for edge cases and boundary conditions
5. Improve coverage to 80%+ target
6. Add integration tests with real API calls (optional, with API keys from env)
