from typing import List, Optional, Dict, Any, Iterator, AsyncIterator, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import httpx

# --- Data Models (mirroring data-model.md) ---

@dataclass
class ProviderConfig:
    provider_type: str
    api_key: str
    base_url: Optional[str] = None
    default_model: Optional[str] = None

@dataclass
class UnifiedRequest:
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False

@dataclass
class UnifiedResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[Any]  # Typed as specific Choice class in implementation
    usage: Any          # Typed as Usage class in implementation

@dataclass
class UnifiedChunk:
    id: str
    object: str
    created: int
    model: str
    choices: List[Any]

# --- Abstract Base Classes ---

class BaseProvider(ABC):
    """
    Abstract base class that all provider adapters must implement.
    """
    
    @abstractmethod
    def convert_request(self, request: UnifiedRequest) -> Dict[str, Any]:
        """Convert unified request to provider-specific request payload."""
        pass

    @abstractmethod
    def convert_response(self, provider_response: Dict[str, Any]) -> UnifiedResponse:
        """Convert provider-specific response to unified response."""
        pass
    
    @abstractmethod
    def send_request(self, client: httpx.Client, request: UnifiedRequest) -> UnifiedResponse:
        """Execute synchronous request."""
        pass

    @abstractmethod
    async def send_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> UnifiedResponse:
        """Execute asynchronous request."""
        pass

    @abstractmethod
    def stream_request(self, client: httpx.Client, request: UnifiedRequest) -> Iterator[UnifiedChunk]:
        """Execute synchronous streaming request."""
        pass

    @abstractmethod
    async def stream_request_async(self, client: httpx.AsyncClient, request: UnifiedRequest) -> AsyncIterator[UnifiedChunk]:
        """Execute asynchronous streaming request."""
        pass

# --- Public Client Interfaces ---

class Client:
    def __init__(self, provider_config: ProviderConfig):
        ...

    @property
    def chat(self):
        ...

class AsyncClient:
    def __init__(self, provider_config: ProviderConfig):
        ...

    @property
    def chat(self):
        ...
