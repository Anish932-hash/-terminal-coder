"""
AI Integration Module
Handles all AI provider integrations with automatic model detection and error handling
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom API error class"""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded error"""
    pass


class AuthenticationError(APIError):
    """Authentication failed error"""
    pass


class ModelNotFoundError(APIError):
    """Model not found error"""
    pass


@dataclass
class AIResponse:
    """Standard AI response format"""
    content: str
    model: str
    provider: str
    tokens_used: int
    response_time: float
    metadata: Dict[str, Any]


@dataclass
class ModelInfo:
    """Model information structure"""
    name: str
    max_tokens: int
    input_cost: float  # per 1k tokens
    output_cost: float  # per 1k tokens
    supports_streaming: bool
    supports_images: bool
    supports_tools: bool


class AIProvider:
    """Base class for AI providers"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.rate_limiter = RateLimiter()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_models(self) -> List[ModelInfo]:
        """Get available models from provider"""
        raise NotImplementedError

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from AI model"""
        raise NotImplementedError

    async def stream_completion(self, messages: List[Dict], model: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat completion from AI model"""
        raise NotImplementedError

    def _handle_api_error(self, status_code: int, response_text: str):
        """Handle API errors consistently"""
        if status_code == 401:
            raise AuthenticationError(f"Authentication failed: {response_text}")
        elif status_code == 429:
            raise RateLimitError(f"Rate limit exceeded: {response_text}")
        elif status_code == 404:
            raise ModelNotFoundError(f"Model not found: {response_text}")
        else:
            raise APIError(f"API error ({status_code}): {response_text}")


class OpenAIProvider(AIProvider):
    """OpenAI API integration"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def get_models(self) -> List[ModelInfo]:
        """Get available OpenAI models"""
        models = [
            ModelInfo("gpt-4", 128000, 0.03, 0.06, True, True, True),
            ModelInfo("gpt-4-turbo", 128000, 0.01, 0.03, True, True, True),
            ModelInfo("gpt-3.5-turbo", 16385, 0.001, 0.002, True, False, True),
            ModelInfo("gpt-4o", 128000, 0.005, 0.015, True, True, True),
            ModelInfo("gpt-4o-mini", 128000, 0.00015, 0.0006, True, True, True),
        ]
        return models

    async def auto_detect_model(self) -> str:
        """Automatically detect the best available model"""
        try:
            async with self.session.get(f"{self.base_url}/models", headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [model["id"] for model in data["data"]]

                    # Prefer latest models
                    preference_order = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
                    for model in preference_order:
                        if model in available_models:
                            return model

                    # Return first available model if none match preference
                    return available_models[0] if available_models else "gpt-3.5-turbo"
                else:
                    logger.warning(f"Failed to fetch models, defaulting to gpt-3.5-turbo")
                    return "gpt-3.5-turbo"
        except Exception as e:
            logger.error(f"Error detecting model: {e}")
            return "gpt-3.5-turbo"

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from OpenAI"""
        start_time = time.time()
        await self.rate_limiter.wait()

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }

        try:
            async with self.session.post(f"{self.base_url}/chat/completions",
                                       headers=self.headers, json=payload) as response:

                response_text = await response.text()

                if response.status != 200:
                    self._handle_api_error(response.status, response_text)

                data = json.loads(response_text)
                response_time = time.time() - start_time

                return AIResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    provider="openai",
                    tokens_used=data["usage"]["total_tokens"],
                    response_time=response_time,
                    metadata=data["usage"]
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")


class AnthropicProvider(AIProvider):
    """Anthropic Claude API integration"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

    async def get_models(self) -> List[ModelInfo]:
        """Get available Anthropic models"""
        models = [
            ModelInfo("claude-3-opus-20240229", 200000, 0.015, 0.075, True, True, True),
            ModelInfo("claude-3-sonnet-20240229", 200000, 0.003, 0.015, True, True, True),
            ModelInfo("claude-3-haiku-20240307", 200000, 0.00025, 0.00125, True, True, True),
            ModelInfo("claude-3-5-sonnet-20241022", 200000, 0.003, 0.015, True, True, True),
        ]
        return models

    async def auto_detect_model(self) -> str:
        """Auto-detect best Anthropic model"""
        return "claude-3-5-sonnet-20241022"  # Latest as of 2024

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from Anthropic"""
        start_time = time.time()
        await self.rate_limiter.wait()

        # Convert OpenAI format to Anthropic format
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)

        payload = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.7)
        }

        if system_message:
            payload["system"] = system_message

        try:
            async with self.session.post(f"{self.base_url}/messages",
                                       headers=self.headers, json=payload) as response:

                response_text = await response.text()

                if response.status != 200:
                    self._handle_api_error(response.status, response_text)

                data = json.loads(response_text)
                response_time = time.time() - start_time

                return AIResponse(
                    content=data["content"][0]["text"],
                    model=data["model"],
                    provider="anthropic",
                    tokens_used=data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
                    response_time=response_time,
                    metadata=data["usage"]
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")


class GoogleProvider(AIProvider):
    """Google Gemini API integration"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.api_key = api_key

    async def get_models(self) -> List[ModelInfo]:
        """Get available Google models"""
        models = [
            ModelInfo("gemini-pro", 32000, 0.0005, 0.0015, True, False, True),
            ModelInfo("gemini-pro-vision", 16000, 0.0005, 0.0015, True, True, True),
            ModelInfo("gemini-1.5-pro", 1000000, 0.00125, 0.00375, True, True, True),
            ModelInfo("gemini-1.5-flash", 1000000, 0.000075, 0.0003, True, True, True),
        ]
        return models

    async def auto_detect_model(self) -> str:
        """Auto-detect best Google model"""
        return "gemini-1.5-pro"

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from Google"""
        start_time = time.time()
        await self.rate_limiter.wait()

        # Convert to Google format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 4000),
                "temperature": kwargs.get("temperature", 0.7)
            }
        }

        try:
            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
            async with self.session.post(url, json=payload) as response:

                response_text = await response.text()

                if response.status != 200:
                    self._handle_api_error(response.status, response_text)

                data = json.loads(response_text)
                response_time = time.time() - start_time

                content = data["candidates"][0]["content"]["parts"][0]["text"]

                return AIResponse(
                    content=content,
                    model=model,
                    provider="google",
                    tokens_used=data.get("usageMetadata", {}).get("totalTokenCount", 0),
                    response_time=response_time,
                    metadata=data.get("usageMetadata", {})
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")


class CohereProvider(AIProvider):
    """Cohere API integration"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.cohere.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def get_models(self) -> List[ModelInfo]:
        """Get available Cohere models"""
        models = [
            ModelInfo("command", 4096, 0.0015, 0.002, True, False, True),
            ModelInfo("command-r", 128000, 0.0005, 0.0015, True, False, True),
            ModelInfo("command-r-plus", 128000, 0.003, 0.015, True, False, True),
        ]
        return models

    async def auto_detect_model(self) -> str:
        """Auto-detect best Cohere model"""
        return "command-r-plus"

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from Cohere"""
        start_time = time.time()
        await self.rate_limiter.wait()

        # Extract user message (Cohere uses different format)
        user_message = messages[-1]["content"] if messages else ""

        # Build chat history for context
        chat_history = []
        for msg in messages[:-1]:
            role = "USER" if msg["role"] == "user" else "CHATBOT"
            chat_history.append({
                "role": role,
                "message": msg["content"]
            })

        payload = {
            "model": model,
            "message": user_message,
            "chat_history": chat_history,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.7)
        }

        try:
            async with self.session.post(f"{self.base_url}/chat",
                                       headers=self.headers, json=payload) as response:

                response_text = await response.text()

                if response.status != 200:
                    self._handle_api_error(response.status, response_text)

                data = json.loads(response_text)
                response_time = time.time() - start_time

                return AIResponse(
                    content=data["text"],
                    model=model,
                    provider="cohere",
                    tokens_used=data.get("meta", {}).get("tokens", {}).get("total_tokens", 0),
                    response_time=response_time,
                    metadata=data.get("meta", {})
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")


class RateLimiter:
    """Rate limiter for API calls"""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    async def wait(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()

        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]

        if len(self.calls) >= self.calls_per_minute:
            # Wait until we can make another call
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.calls.append(now)


class AIManager:
    """Central AI management system"""

    def __init__(self):
        self.providers: Dict[str, AIProvider] = {}
        self.current_provider = None
        self.current_model = None

    def add_provider(self, name: str, provider: AIProvider):
        """Add an AI provider"""
        self.providers[name] = provider

    async def initialize_provider(self, provider_name: str, api_key: str) -> bool:
        """Initialize a specific provider"""
        try:
            if provider_name == "openai":
                provider = OpenAIProvider(api_key)
            elif provider_name == "anthropic":
                provider = AnthropicProvider(api_key)
            elif provider_name == "google":
                provider = GoogleProvider(api_key)
            elif provider_name == "cohere":
                provider = CohereProvider(api_key)
            else:
                raise ValueError(f"Unknown provider: {provider_name}")

            self.providers[provider_name] = provider
            self.current_provider = provider_name

            # Auto-detect best model
            async with provider:
                self.current_model = await provider.auto_detect_model()

            logger.info(f"Initialized {provider_name} with model {self.current_model}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {provider_name}: {e}")
            return False

    async def chat(self, messages: List[Dict], **kwargs) -> AIResponse:
        """Send chat request to current provider"""
        if not self.current_provider or self.current_provider not in self.providers:
            raise APIError("No provider initialized")

        provider = self.providers[self.current_provider]

        async with provider:
            return await provider.chat_completion(messages, self.current_model, **kwargs)

    async def get_available_models(self, provider_name: str) -> List[ModelInfo]:
        """Get available models for a provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not initialized")

        return await self.providers[provider_name].get_models()

    def switch_provider(self, provider_name: str, model: str = None):
        """Switch to a different provider/model"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not initialized")

        self.current_provider = provider_name
        if model:
            self.current_model = model

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health = {}

        for name, provider in self.providers.items():
            try:
                async with provider:
                    # Try a simple completion
                    response = await provider.chat_completion(
                        [{"role": "user", "content": "Hello"}],
                        await provider.auto_detect_model(),
                        max_tokens=10
                    )
                    health[name] = True
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health[name] = False

        return health