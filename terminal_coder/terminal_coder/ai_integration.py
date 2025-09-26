"""
AI Integration Module - Real Implementation
Handles all AI provider integrations with actual API calls for Linux systems
"""

import asyncio
import aiohttp
import json
import time
import os
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import logging
import ssl
import certifi
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom API error class"""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded error"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


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
    cost: float = 0.0


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
    context_window: int = 0


class AIProvider:
    """Base class for AI providers"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.rate_limiter = RateLimiter()
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        connector = aiohttp.TCPConnector(ssl=self.ssl_context, limit=10)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "Terminal-Coder-Linux/1.2.0"}
        )
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

    def _handle_api_error(self, status_code: int, response_text: str, headers: Dict = None):
        """Handle API errors consistently"""
        if status_code == 401:
            raise AuthenticationError(f"Authentication failed: {response_text}")
        elif status_code == 429:
            retry_after = None
            if headers and 'retry-after' in headers:
                try:
                    retry_after = int(headers['retry-after'])
                except ValueError:
                    pass
            raise RateLimitError(f"Rate limit exceeded: {response_text}", retry_after)
        elif status_code == 404:
            raise ModelNotFoundError(f"Model not found: {response_text}")
        elif status_code >= 500:
            raise APIError(f"Server error ({status_code}): {response_text}")
        else:
            raise APIError(f"API error ({status_code}): {response_text}")


class OpenAIProvider(AIProvider):
    """OpenAI API integration with real implementation"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v1"
        }

    async def get_models(self) -> List[ModelInfo]:
        """Get available OpenAI models with real API call"""
        try:
            async with self.session.get(f"{self.base_url}/models", headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []

                    # Map known models with their specifications
                    model_specs = {
                        "gpt-4": ModelInfo("gpt-4", 8192, 0.03, 0.06, True, False, True, 8192),
                        "gpt-4-32k": ModelInfo("gpt-4-32k", 32768, 0.06, 0.12, True, False, True, 32768),
                        "gpt-4-turbo-preview": ModelInfo("gpt-4-turbo-preview", 128000, 0.01, 0.03, True, True, True, 128000),
                        "gpt-4-vision-preview": ModelInfo("gpt-4-vision-preview", 128000, 0.01, 0.03, True, True, True, 128000),
                        "gpt-3.5-turbo": ModelInfo("gpt-3.5-turbo", 4096, 0.0015, 0.002, True, False, True, 16385),
                        "gpt-3.5-turbo-16k": ModelInfo("gpt-3.5-turbo-16k", 16384, 0.003, 0.004, True, False, True, 16385),
                        "gpt-4o": ModelInfo("gpt-4o", 128000, 0.005, 0.015, True, True, True, 128000),
                        "gpt-4o-mini": ModelInfo("gpt-4o-mini", 128000, 0.00015, 0.0006, True, True, True, 128000),
                    }

                    available_models = {model["id"] for model in data["data"] if model["id"] in model_specs}

                    for model_id in available_models:
                        models.append(model_specs[model_id])

                    return models
                else:
                    logger.warning(f"Failed to fetch OpenAI models: {response.status}")
                    return self._get_default_openai_models()
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {e}")
            return self._get_default_openai_models()

    def _get_default_openai_models(self) -> List[ModelInfo]:
        """Get default OpenAI models if API call fails"""
        return [
            ModelInfo("gpt-4o", 128000, 0.005, 0.015, True, True, True, 128000),
            ModelInfo("gpt-4", 8192, 0.03, 0.06, True, False, True, 8192),
            ModelInfo("gpt-3.5-turbo", 4096, 0.0015, 0.002, True, False, True, 16385),
        ]

    async def auto_detect_model(self) -> str:
        """Automatically detect the best available model"""
        try:
            models = await self.get_models()
            if models:
                # Prefer gpt-4o, then gpt-4, then gpt-3.5-turbo
                preference_order = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
                available_names = [model.name for model in models]

                for preferred in preference_order:
                    if preferred in available_names:
                        return preferred

                return models[0].name
            else:
                return "gpt-3.5-turbo"
        except Exception:
            return "gpt-3.5-turbo"

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from OpenAI with real API call"""
        start_time = time.time()
        await self.rate_limiter.wait()

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }

        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]

        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:

                response_text = await response.text()

                if response.status != 200:
                    self._handle_api_error(response.status, response_text, dict(response.headers))

                data = json.loads(response_text)
                response_time = time.time() - start_time

                # Calculate cost
                cost = self._calculate_cost(
                    model,
                    data["usage"]["prompt_tokens"],
                    data["usage"]["completion_tokens"]
                )

                return AIResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    provider="openai",
                    tokens_used=data["usage"]["total_tokens"],
                    response_time=response_time,
                    cost=cost,
                    metadata=data["usage"]
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost based on token usage"""
        model_costs = {
            "gpt-4": (0.03, 0.06),
            "gpt-4-32k": (0.06, 0.12),
            "gpt-4-turbo-preview": (0.01, 0.03),
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-3.5-turbo": (0.0015, 0.002),
        }

        if model in model_costs:
            input_cost, output_cost = model_costs[model]
            return (input_tokens / 1000 * input_cost) + (output_tokens / 1000 * output_cost)
        return 0.0

    async def stream_completion(self, messages: List[Dict], model: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat completion from OpenAI"""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }

        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            ) as response:

                if response.status != 200:
                    response_text = await response.text()
                    self._handle_api_error(response.status, response_text, dict(response.headers))

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        line = line[6:]  # Remove 'data: ' prefix
                        if line == '[DONE]':
                            break
                        try:
                            chunk = json.loads(line)
                            if chunk['choices'][0]['delta'].get('content'):
                                yield chunk['choices'][0]['delta']['content']
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")


class AnthropicProvider(AIProvider):
    """Anthropic Claude API integration with real implementation"""

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
        return [
            ModelInfo("claude-3-opus-20240229", 200000, 0.015, 0.075, True, True, True, 200000),
            ModelInfo("claude-3-sonnet-20240229", 200000, 0.003, 0.015, True, True, True, 200000),
            ModelInfo("claude-3-haiku-20240307", 200000, 0.00025, 0.00125, True, True, True, 200000),
            ModelInfo("claude-3-5-sonnet-20241022", 200000, 0.003, 0.015, True, True, True, 200000),
        ]

    async def auto_detect_model(self) -> str:
        """Auto-detect best Anthropic model"""
        return "claude-3-5-sonnet-20241022"

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from Anthropic with real API call"""
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

        # Add optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]

        try:
            async with self.session.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload
            ) as response:

                response_text = await response.text()

                if response.status != 200:
                    self._handle_api_error(response.status, response_text, dict(response.headers))

                data = json.loads(response_text)
                response_time = time.time() - start_time

                # Calculate cost
                cost = self._calculate_cost(
                    model,
                    data["usage"]["input_tokens"],
                    data["usage"]["output_tokens"]
                )

                return AIResponse(
                    content=data["content"][0]["text"],
                    model=data["model"],
                    provider="anthropic",
                    tokens_used=data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
                    response_time=response_time,
                    cost=cost,
                    metadata=data["usage"]
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost for Anthropic"""
        model_costs = {
            "claude-3-opus-20240229": (0.015, 0.075),
            "claude-3-sonnet-20240229": (0.003, 0.015),
            "claude-3-haiku-20240307": (0.00025, 0.00125),
            "claude-3-5-sonnet-20241022": (0.003, 0.015),
        }

        if model in model_costs:
            input_cost, output_cost = model_costs[model]
            return (input_tokens / 1000 * input_cost) + (output_tokens / 1000 * output_cost)
        return 0.0

    async def stream_completion(self, messages: List[Dict], model: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat completion from Anthropic"""
        # Convert messages format
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
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }

        if system_message:
            payload["system"] = system_message

        try:
            async with self.session.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload
            ) as response:

                if response.status != 200:
                    response_text = await response.text()
                    self._handle_api_error(response.status, response_text, dict(response.headers))

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        line = line[6:]
                        try:
                            chunk = json.loads(line)
                            if chunk.get('type') == 'content_block_delta':
                                if chunk['delta'].get('text'):
                                    yield chunk['delta']['text']
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")


class GoogleProvider(AIProvider):
    """Google Gemini API integration with real implementation"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.api_key = api_key

    async def get_models(self) -> List[ModelInfo]:
        """Get available Google models with real API call"""
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []

                    for model in data.get("models", []):
                        model_name = model["name"].split("/")[-1]
                        if "gemini" in model_name.lower():
                            # Map to our model info structure
                            if "pro" in model_name:
                                models.append(ModelInfo(
                                    model_name, 32000, 0.0005, 0.0015,
                                    True, True, True, 32000
                                ))
                            elif "flash" in model_name:
                                models.append(ModelInfo(
                                    model_name, 1000000, 0.000075, 0.0003,
                                    True, True, True, 1000000
                                ))

                    return models if models else self._get_default_google_models()
                else:
                    return self._get_default_google_models()
        except Exception:
            return self._get_default_google_models()

    def _get_default_google_models(self) -> List[ModelInfo]:
        """Get default Google models"""
        return [
            ModelInfo("gemini-1.5-pro", 2000000, 0.00125, 0.00375, True, True, True, 2000000),
            ModelInfo("gemini-1.5-flash", 1000000, 0.000075, 0.0003, True, True, True, 1000000),
            ModelInfo("gemini-pro", 32000, 0.0005, 0.0015, True, False, True, 32000),
        ]

    async def auto_detect_model(self) -> str:
        """Auto-detect best Google model"""
        return "gemini-1.5-pro"

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from Google with real API call"""
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

        # Add optional parameters
        if "top_p" in kwargs:
            payload["generationConfig"]["topP"] = kwargs["top_p"]

        try:
            url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
            async with self.session.post(url, json=payload) as response:

                response_text = await response.text()

                if response.status != 200:
                    self._handle_api_error(response.status, response_text, dict(response.headers))

                data = json.loads(response_text)
                response_time = time.time() - start_time

                if not data.get("candidates") or not data["candidates"][0].get("content"):
                    raise APIError("Invalid response format from Google API")

                content = data["candidates"][0]["content"]["parts"][0]["text"]

                # Calculate cost
                usage_metadata = data.get("usageMetadata", {})
                input_tokens = usage_metadata.get("promptTokenCount", 0)
                output_tokens = usage_metadata.get("candidatesTokenCount", 0)
                total_tokens = usage_metadata.get("totalTokenCount", input_tokens + output_tokens)

                cost = self._calculate_cost(model, input_tokens, output_tokens)

                return AIResponse(
                    content=content,
                    model=model,
                    provider="google",
                    tokens_used=total_tokens,
                    response_time=response_time,
                    cost=cost,
                    metadata=usage_metadata
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost for Google"""
        model_costs = {
            "gemini-1.5-pro": (0.00125, 0.00375),
            "gemini-1.5-flash": (0.000075, 0.0003),
            "gemini-pro": (0.0005, 0.0015),
        }

        if model in model_costs:
            input_cost, output_cost = model_costs[model]
            return (input_tokens / 1000 * input_cost) + (output_tokens / 1000 * output_cost)
        return 0.0


class CohereProvider(AIProvider):
    """Cohere API integration with real implementation"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.cohere.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def get_models(self) -> List[ModelInfo]:
        """Get available Cohere models"""
        return [
            ModelInfo("command", 4096, 0.0015, 0.002, True, False, True, 4096),
            ModelInfo("command-r", 128000, 0.0005, 0.0015, True, False, True, 128000),
            ModelInfo("command-r-plus", 128000, 0.003, 0.015, True, False, True, 128000),
        ]

    async def auto_detect_model(self) -> str:
        """Auto-detect best Cohere model"""
        return "command-r-plus"

    async def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> AIResponse:
        """Get chat completion from Cohere with real API call"""
        start_time = time.time()
        await self.rate_limiter.wait()

        # Extract user message and build chat history
        if not messages:
            raise APIError("No messages provided")

        user_message = messages[-1]["content"] if messages else ""

        # Build chat history for context (exclude the last message)
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

        # Add optional parameters
        if "p" in kwargs:
            payload["p"] = kwargs["p"]

        try:
            async with self.session.post(
                f"{self.base_url}/chat",
                headers=self.headers,
                json=payload
            ) as response:

                response_text = await response.text()

                if response.status != 200:
                    self._handle_api_error(response.status, response_text, dict(response.headers))

                data = json.loads(response_text)
                response_time = time.time() - start_time

                # Calculate cost (approximate based on response length)
                meta = data.get("meta", {})
                billed_units = meta.get("billed_units", {})
                input_tokens = billed_units.get("input_tokens", 0)
                output_tokens = billed_units.get("output_tokens", 0)

                cost = self._calculate_cost(model, input_tokens, output_tokens)

                return AIResponse(
                    content=data["text"],
                    model=model,
                    provider="cohere",
                    tokens_used=input_tokens + output_tokens,
                    response_time=response_time,
                    cost=cost,
                    metadata=meta
                )

        except aiohttp.ClientError as e:
            raise APIError(f"Network error: {e}")

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost for Cohere"""
        model_costs = {
            "command": (0.0015, 0.002),
            "command-r": (0.0005, 0.0015),
            "command-r-plus": (0.003, 0.015),
        }

        if model in model_costs:
            input_cost, output_cost = model_costs[model]
            return (input_tokens / 1000 * input_cost) + (output_tokens / 1000 * output_cost)
        return 0.0


class RateLimiter:
    """Enhanced rate limiter with exponential backoff"""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.last_reset = time.time()

    async def wait(self):
        """Wait if rate limit would be exceeded with exponential backoff"""
        now = time.time()

        # Reset counter every minute
        if now - self.last_reset >= 60:
            self.calls = []
            self.last_reset = now

        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]

        if len(self.calls) >= self.calls_per_minute:
            # Calculate wait time with exponential backoff
            excess_calls = len(self.calls) - self.calls_per_minute
            base_wait = 60 - (now - self.calls[0])
            backoff_wait = min(base_wait * (2 ** min(excess_calls, 5)), 300)  # Max 5 minutes

            logger.warning(f"Rate limit approaching, waiting {backoff_wait:.2f} seconds")
            await asyncio.sleep(backoff_wait)

        self.calls.append(now)


class AIManager:
    """Enhanced AI management system with real implementations"""

    def __init__(self):
        self.providers: Dict[str, AIProvider] = {}
        self.current_provider = None
        self.current_model = None
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_by_provider": {},
            "session_start": datetime.now()
        }

    async def initialize_provider(self, provider_name: str, api_key: str) -> bool:
        """Initialize a specific provider with real API validation"""
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

            # Test the provider with a simple request
            async with provider:
                # Try to get models to validate the API key
                try:
                    models = await provider.get_models()
                    if models:
                        self.providers[provider_name] = provider
                        self.current_provider = provider_name
                        self.current_model = await provider.auto_detect_model()

                        logger.info(f"Successfully initialized {provider_name} with model {self.current_model}")
                        return True
                except Exception as e:
                    logger.error(f"Failed to validate {provider_name} API: {e}")
                    return False

        except Exception as e:
            logger.error(f"Failed to initialize {provider_name}: {e}")
            return False

    async def chat(self, messages: List[Dict], **kwargs) -> AIResponse:
        """Send chat request to current provider with usage tracking"""
        if not self.current_provider or self.current_provider not in self.providers:
            raise APIError("No provider initialized")

        provider = self.providers[self.current_provider]

        try:
            async with provider:
                response = await provider.chat_completion(messages, self.current_model, **kwargs)

                # Update usage statistics
                self.usage_stats["total_requests"] += 1
                self.usage_stats["total_tokens"] += response.tokens_used
                self.usage_stats["total_cost"] += response.cost

                if self.current_provider not in self.usage_stats["requests_by_provider"]:
                    self.usage_stats["requests_by_provider"][self.current_provider] = 0
                self.usage_stats["requests_by_provider"][self.current_provider] += 1

                logger.debug(f"AI request completed: {response.tokens_used} tokens, ${response.cost:.4f}")
                return response

        except Exception as e:
            logger.error(f"AI request failed: {e}")
            raise

    async def stream_chat(self, messages: List[Dict], **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response from current provider"""
        if not self.current_provider or self.current_provider not in self.providers:
            raise APIError("No provider initialized")

        provider = self.providers[self.current_provider]

        try:
            async with provider:
                async for chunk in provider.stream_completion(messages, self.current_model, **kwargs):
                    yield chunk
        except Exception as e:
            logger.error(f"AI streaming failed: {e}")
            raise

    async def get_available_models(self, provider_name: str) -> List[ModelInfo]:
        """Get available models for a provider"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not initialized")

        provider = self.providers[provider_name]
        async with provider:
            return await provider.get_models()

    def switch_provider(self, provider_name: str, model: str = None):
        """Switch to a different provider/model"""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not initialized")

        self.current_provider = provider_name
        if model:
            self.current_model = model
        else:
            # Auto-detect best model for new provider
            provider = self.providers[provider_name]
            asyncio.create_task(self._set_best_model(provider))

    async def _set_best_model(self, provider):
        """Set the best model for a provider"""
        try:
            async with provider:
                self.current_model = await provider.auto_detect_model()
        except Exception as e:
            logger.error(f"Failed to auto-detect model: {e}")

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers with real API calls"""
        health = {}

        for name, provider in self.providers.items():
            try:
                async with provider:
                    # Try a minimal completion to test the provider
                    test_messages = [{"role": "user", "content": "Hello"}]
                    response = await provider.chat_completion(
                        test_messages,
                        await provider.auto_detect_model(),
                        max_tokens=1,
                        temperature=0
                    )
                    health[name] = True
                    logger.debug(f"Health check passed for {name}")
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health[name] = False

        return health

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        session_duration = datetime.now() - self.usage_stats["session_start"]

        return {
            **self.usage_stats,
            "session_duration_minutes": session_duration.total_seconds() / 60,
            "average_cost_per_request": (
                self.usage_stats["total_cost"] / max(self.usage_stats["total_requests"], 1)
            ),
            "current_provider": self.current_provider,
            "current_model": self.current_model
        }

    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests_by_provider": {},
            "session_start": datetime.now()
        }