"""
Modern AI Integration Module
Advanced AI provider integrations with Python 3.13+ features
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Final, Literal, TypeAlias

import aiohttp
import structlog
from pydantic import BaseModel, Field, ConfigDict
from rich.console import Console

# Type aliases for better code readability
APIKey: TypeAlias = str
ModelName: TypeAlias = str
TokenCount: TypeAlias = int
ResponseTime: TypeAlias = float
JSONData: TypeAlias = dict[str, Any]

# Configure structured logging
logger = structlog.get_logger(__name__)


class AIProviderType(Enum):
    """Enumeration of supported AI providers"""
    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    COHERE = auto()
    OLLAMA = auto()  # Local AI models


class ModelCapability(Enum):
    """AI model capabilities"""
    TEXT_GENERATION = auto()
    CODE_GENERATION = auto()
    CODE_ANALYSIS = auto()
    IMAGE_ANALYSIS = auto()
    MULTIMODAL = auto()
    FUNCTION_CALLING = auto()


@dataclass(frozen=True, slots=True)
class AIModel:
    """Modern AI model configuration with validation"""
    name: ModelName
    provider: AIProviderType
    max_tokens: TokenCount
    context_window: TokenCount
    capabilities: set[ModelCapability] = field(default_factory=set)
    cost_per_token_input: float = 0.0
    cost_per_token_output: float = 0.0

    def __post_init__(self) -> None:
        """Validate model configuration"""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.max_tokens > self.context_window:
            raise ValueError("max_tokens cannot exceed context_window")


class AIRequest(BaseModel):
    """Modern AI request model using Pydantic v2"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    message: str = Field(..., min_length=1, max_length=100000)
    model: ModelName = Field(..., min_length=1)
    provider: AIProviderType
    max_tokens: TokenCount = Field(default=4000, ge=1, le=200000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    system_prompt: str | None = Field(default=None, max_length=10000)
    context: list[dict[str, Any]] = Field(default_factory=list)


class AIResponse(BaseModel):
    """Modern AI response model"""
    model_config = ConfigDict(validate_assignment=True)

    content: str
    model: ModelName
    provider: AIProviderType
    tokens_used: TokenCount = Field(ge=0)
    response_time: ResponseTime = Field(ge=0.0)
    cost_estimate: float = Field(default=0.0, ge=0.0)
    metadata: JSONData = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class AIError(Exception):
    """Base AI error class with structured information"""

    def __init__(
        self,
        message: str,
        provider: AIProviderType | None = None,
        error_code: str | None = None,
        retry_after: int | None = None
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.retry_after = retry_after


class RateLimitError(AIError):
    """Rate limit exceeded error"""
    pass


class AuthenticationError(AIError):
    """Authentication failed error"""
    pass


class ModelNotFoundError(AIError):
    """Model not found error"""
    pass


class ModernAIIntegration:
    """Modern AI integration with advanced features"""

    # Modern models with latest capabilities
    MODELS: Final[dict[AIProviderType, dict[str, AIModel]]] = {
        AIProviderType.OPENAI: {
            "gpt-4o": AIModel(
                name="gpt-4o",
                provider=AIProviderType.OPENAI,
                max_tokens=128000,
                context_window=128000,
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_ANALYSIS,
                    ModelCapability.MULTIMODAL,
                    ModelCapability.FUNCTION_CALLING,
                },
                cost_per_token_input=0.0025,
                cost_per_token_output=0.01,
            ),
            "gpt-4-turbo": AIModel(
                name="gpt-4-turbo",
                provider=AIProviderType.OPENAI,
                max_tokens=128000,
                context_window=128000,
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_ANALYSIS,
                    ModelCapability.FUNCTION_CALLING,
                },
                cost_per_token_input=0.01,
                cost_per_token_output=0.03,
            ),
        },
        AIProviderType.ANTHROPIC: {
            "claude-3-5-sonnet-20241022": AIModel(
                name="claude-3-5-sonnet-20241022",
                provider=AIProviderType.ANTHROPIC,
                max_tokens=8192,
                context_window=200000,
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_ANALYSIS,
                },
                cost_per_token_input=0.003,
                cost_per_token_output=0.015,
            ),
        },
        AIProviderType.GOOGLE: {
            "gemini-1.5-pro": AIModel(
                name="gemini-1.5-pro",
                provider=AIProviderType.GOOGLE,
                max_tokens=8192,
                context_window=1048576,  # 1M context window
                capabilities={
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_ANALYSIS,
                    ModelCapability.MULTIMODAL,
                },
                cost_per_token_input=0.00125,
                cost_per_token_output=0.005,
            ),
        },
    }

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._session: aiohttp.ClientSession | None = None
        self._rate_limits: dict[AIProviderType, float] = {}

    async def __aenter__(self) -> ModernAIIntegration:
        """Async context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self._close_session()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is available"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def _close_session(self) -> None:
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def get_model(self, provider: AIProviderType, model_name: str) -> AIModel:
        """Get model configuration with validation"""
        try:
            return self.MODELS[provider][model_name]
        except KeyError:
            available_models = list(self.MODELS.get(provider, {}).keys())
            raise ModelNotFoundError(
                f"Model '{model_name}' not found for provider {provider.name}. "
                f"Available models: {available_models}",
                provider=provider
            )

    async def generate_response(
        self,
        request: AIRequest,
        api_key: APIKey
    ) -> AIResponse:
        """Generate AI response with modern error handling"""
        start_time = time.time()

        try:
            await self._check_rate_limits(request.provider)
            model = self.get_model(request.provider, request.model)

            # Handle regular response
            response = await self._make_request(request, api_key, model)
            response.response_time = time.time() - start_time
            return response

        except Exception as e:
            logger.error(
                "AI request failed",
                provider=request.provider.name,
                model=request.model,
                error=str(e),
                response_time=time.time() - start_time
            )
            raise

    async def generate_streaming_response(
        self,
        request: AIRequest,
        api_key: APIKey
    ) -> AsyncGenerator[str, None]:
        """Generate streaming AI response"""
        start_time = time.time()

        try:
            await self._check_rate_limits(request.provider)
            model = self.get_model(request.provider, request.model)

            # Handle streaming response
            async for chunk in self._stream_response(request, api_key, model):
                yield chunk

        except Exception as e:
            logger.error(
                "AI request failed",
                provider=request.provider.name,
                model=request.model,
                error=str(e),
                response_time=time.time() - start_time
            )
            raise

    async def _check_rate_limits(self, provider: AIProviderType) -> None:
        """Check and enforce rate limits"""
        current_time = time.time()
        last_request = self._rate_limits.get(provider, 0)

        # Enforce minimum delay between requests
        min_delay = {
            AIProviderType.OPENAI: 1.0,
            AIProviderType.ANTHROPIC: 1.2,
            AIProviderType.GOOGLE: 1.0,
            AIProviderType.COHERE: 0.6,
            AIProviderType.OLLAMA: 0.1,  # Local models have no rate limits
        }

        delay_needed = min_delay.get(provider, 1.0)
        time_since_last = current_time - last_request

        if time_since_last < delay_needed:
            await asyncio.sleep(delay_needed - time_since_last)

        self._rate_limits[provider] = time.time()

    async def _make_request(
        self,
        request: AIRequest,
        api_key: APIKey,
        model: AIModel
    ) -> AIResponse:
        """Make HTTP request to AI provider"""
        await self._ensure_session()

        headers, payload = self._prepare_request(request, api_key)
        url = self._get_api_url(request.provider)

        try:
            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get("retry-after", 60))
                    raise RateLimitError(
                        f"Rate limit exceeded for {request.provider.name}",
                        provider=request.provider,
                        retry_after=retry_after
                    )

                if response.status == 401:
                    raise AuthenticationError(
                        f"Invalid API key for {request.provider.name}",
                        provider=request.provider
                    )

                if response.status != 200:
                    error_data = await response.json()
                    raise AIError(
                        f"API request failed: {error_data}",
                        provider=request.provider,
                        error_code=str(response.status)
                    )

                data = await response.json()
                return self._parse_response(data, request, model)

        except aiohttp.ClientError as e:
            raise AIError(
                f"Network error: {e}",
                provider=request.provider
            )

    async def _stream_response(
        self,
        request: AIRequest,
        api_key: APIKey,
        model: AIModel
    ) -> AsyncGenerator[str, None]:
        """Handle streaming AI responses"""
        await self._ensure_session()

        headers, payload = self._prepare_request(request, api_key, stream=True)
        url = self._get_api_url(request.provider)

        async with self._session.post(url, json=payload, headers=headers) as response:
            async for line in response.content:
                if line:
                    # Parse SSE data and yield content chunks
                    content_chunk = self._parse_stream_chunk(line, request.provider)
                    if content_chunk:
                        yield content_chunk

    def _prepare_request(
        self,
        request: AIRequest,
        api_key: APIKey,
        stream: bool = False
    ) -> tuple[dict[str, str], JSONData]:
        """Prepare HTTP request headers and payload"""
        match request.provider:
            case AIProviderType.OPENAI:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": request.model,
                    "messages": [{"role": "user", "content": request.message}],
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "stream": stream
                }
                if request.system_prompt:
                    payload["messages"].insert(0, {
                        "role": "system",
                        "content": request.system_prompt
                    })

            case AIProviderType.ANTHROPIC:
                headers = {
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                payload = {
                    "model": request.model,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "messages": [{"role": "user", "content": request.message}],
                    "stream": stream
                }
                if request.system_prompt:
                    payload["system"] = request.system_prompt

            case _:
                raise ValueError(f"Unsupported provider: {request.provider}")

        return headers, payload

    def _get_api_url(self, provider: AIProviderType) -> str:
        """Get API URL for provider"""
        urls = {
            AIProviderType.OPENAI: "https://api.openai.com/v1/chat/completions",
            AIProviderType.ANTHROPIC: "https://api.anthropic.com/v1/messages",
            AIProviderType.GOOGLE: "https://generativelanguage.googleapis.com/v1beta/models",
            AIProviderType.COHERE: "https://api.cohere.ai/v1/generate",
        }
        return urls[provider]

    def _parse_response(
        self,
        data: JSONData,
        request: AIRequest,
        model: AIModel
    ) -> AIResponse:
        """Parse API response into AIResponse object"""
        match request.provider:
            case AIProviderType.OPENAI:
                choice = data["choices"][0]
                content = choice["message"]["content"]
                tokens_used = data["usage"]["total_tokens"]

            case AIProviderType.ANTHROPIC:
                content = data["content"][0]["text"]
                tokens_used = data["usage"]["output_tokens"]

            case _:
                raise ValueError(f"Unsupported provider: {request.provider}")

        # Calculate cost estimate
        cost_estimate = (
            tokens_used * model.cost_per_token_input +
            tokens_used * model.cost_per_token_output
        )

        return AIResponse(
            content=content,
            model=request.model,
            provider=request.provider,
            tokens_used=tokens_used,
            response_time=0.0,  # Will be set by caller
            cost_estimate=cost_estimate,
            metadata=data
        )

    def _parse_stream_chunk(self, chunk: bytes, provider: AIProviderType) -> str | None:
        """Parse streaming response chunk"""
        # Implementation would depend on provider's streaming format
        # This is a simplified version
        try:
            line = chunk.decode('utf-8').strip()
            if line.startswith("data: "):
                import json
                data = json.loads(line[6:])  # Remove "data: " prefix

                match provider:
                    case AIProviderType.OPENAI:
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            return delta.get("content", "")
                    case _:
                        return None
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        return None

    @contextlib.asynccontextmanager
    async def batch_requests(self, max_concurrent: int = 5) -> AsyncGenerator[list[Any], None]:
        """Context manager for batch processing AI requests"""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def process_request(request_func):
            async with semaphore:
                try:
                    result = await request_func()
                    results.append(result)
                except Exception as e:
                    logger.error("Batch request failed", error=str(e))
                    results.append(e)

        yield results

    def get_available_models(self) -> dict[str, list[str]]:
        """Get list of available models by provider"""
        return {
            provider.name.lower(): list(models.keys())
            for provider, models in self.MODELS.items()
        }


# Utility functions for backwards compatibility
async def create_ai_integration() -> ModernAIIntegration:
    """Create and initialize AI integration"""
    integration = ModernAIIntegration()
    await integration._ensure_session()
    return integration


def get_supported_providers() -> list[str]:
    """Get list of supported AI providers"""
    return [provider.name.lower() for provider in AIProviderType]