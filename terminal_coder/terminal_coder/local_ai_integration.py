"""
Local AI Integration Module
Support for local AI models using Ollama, Hugging Face Transformers, and other local inference engines
"""

import asyncio
import aiohttp
import json
import os
import logging
import subprocess
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        pipeline, GPTNeoXForCausalLM, GPT2LMHeadModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Local model support limited.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LocalModelInfo:
    """Local model information"""
    name: str
    model_type: str  # ollama, transformers, onnx
    size_gb: float
    parameters: str
    capabilities: List[str]
    memory_required_gb: float
    status: str  # available, downloading, loaded, error
    model_path: Optional[str] = None
    quantization: Optional[str] = None


@dataclass
class LocalAIResponse:
    """Local AI response format"""
    content: str
    model: str
    provider: str
    inference_time: float
    tokens_per_second: float
    memory_used_gb: float
    metadata: Dict[str, Any]


class OllamaIntegration:
    """Ollama local AI integration with full API support"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None
        self.available_models = {}
        self.model_status = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def check_ollama_status(self) -> bool:
        """Check if Ollama server is running"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False

    async def start_ollama_server(self) -> bool:
        """Start Ollama server if not running"""
        try:
            # Check if already running
            if await self.check_ollama_status():
                return True

            # Try to start Ollama
            logger.info("Starting Ollama server...")
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )

            # Wait for server to start
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                if await self.check_ollama_status():
                    logger.info("Ollama server started successfully")
                    return True

            logger.error("Failed to start Ollama server")
            return False

        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama first.")
            return False
        except Exception as e:
            logger.error(f"Error starting Ollama: {e}")
            return False

    async def get_available_models(self) -> List[LocalModelInfo]:
        """Get list of available Ollama models"""
        try:
            if not await self.check_ollama_status():
                return []

            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    return []

                data = await response.json()
                models = []

                for model in data.get("models", []):
                    name = model.get("name", "unknown")
                    size = model.get("size", 0) / (1024**3)  # Convert to GB
                    modified = model.get("modified_at", "")

                    # Extract model info from name
                    capabilities = self._get_model_capabilities(name)
                    parameters = self._extract_parameters(name)

                    models.append(LocalModelInfo(
                        name=name,
                        model_type="ollama",
                        size_gb=round(size, 2),
                        parameters=parameters,
                        capabilities=capabilities,
                        memory_required_gb=size * 1.5,  # Estimate
                        status="available",
                        model_path=None
                    ))

                self.available_models = {model.name: model for model in models}
                return models

        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return []

    async def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """Pull/download a model from Ollama"""
        try:
            if not await self.check_ollama_status():
                if not await self.start_ollama_server():
                    return False

            logger.info(f"Pulling model: {model_name}")

            # Start pull request
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True}
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to start model pull: {response.status}")
                    return False

                # Stream progress
                async for line in response.content:
                    try:
                        if line.strip():
                            data = json.loads(line)
                            status = data.get("status", "")

                            if progress_callback:
                                await progress_callback(data)

                            if "error" in data:
                                logger.error(f"Model pull error: {data['error']}")
                                return False

                            if status == "success":
                                logger.info(f"Model {model_name} pulled successfully")
                                return True

                    except json.JSONDecodeError:
                        continue

            return False

        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    async def generate_completion(self, model: str, prompt: str, **kwargs) -> LocalAIResponse:
        """Generate completion using Ollama model"""
        try:
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / (1024**3)

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_ctx": kwargs.get("max_tokens", 2048),
                    "top_p": kwargs.get("top_p", 0.9),
                    "stop": kwargs.get("stop", [])
                }
            }

            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")

                data = await response.json()

                inference_time = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / (1024**3)
                memory_used = memory_after - memory_before

                # Calculate tokens per second (approximate)
                response_text = data.get("response", "")
                estimated_tokens = len(response_text.split()) * 1.3  # Rough estimate
                tokens_per_second = estimated_tokens / inference_time if inference_time > 0 else 0

                return LocalAIResponse(
                    content=response_text,
                    model=model,
                    provider="ollama",
                    inference_time=inference_time,
                    tokens_per_second=tokens_per_second,
                    memory_used_gb=memory_used,
                    metadata={
                        "total_duration": data.get("total_duration", 0),
                        "load_duration": data.get("load_duration", 0),
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                        "eval_count": data.get("eval_count", 0)
                    }
                )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    async def stream_completion(self, model: str, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream completion from Ollama model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_ctx": kwargs.get("max_tokens", 2048),
                    "top_p": kwargs.get("top_p", 0.9)
                }
            }

            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")

                async for line in response.content:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise

    def _get_model_capabilities(self, model_name: str) -> List[str]:
        """Determine model capabilities from name"""
        name_lower = model_name.lower()
        capabilities = ["text_generation"]

        if "code" in name_lower or "deepseek" in name_lower or "codellama" in name_lower:
            capabilities.append("code_generation")

        if "instruct" in name_lower or "chat" in name_lower:
            capabilities.append("instruction_following")

        if "vision" in name_lower or "llava" in name_lower:
            capabilities.append("vision")

        if any(x in name_lower for x in ["7b", "13b", "30b", "70b"]):
            capabilities.append("large_context")

        return capabilities

    def _extract_parameters(self, model_name: str) -> str:
        """Extract parameter count from model name"""
        name_lower = model_name.lower()

        if "70b" in name_lower:
            return "70B"
        elif "30b" in name_lower:
            return "30B"
        elif "13b" in name_lower:
            return "13B"
        elif "7b" in name_lower:
            return "7B"
        elif "3b" in name_lower:
            return "3B"
        elif "1b" in name_lower:
            return "1B"
        else:
            return "Unknown"


class HuggingFaceLocal:
    """Local Hugging Face Transformers integration"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/terminal_coder/models")
        self.loaded_models = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {self.device}")

    def get_popular_models(self) -> List[LocalModelInfo]:
        """Get list of popular local models"""
        popular_models = [
            {
                "name": "microsoft/DialoGPT-medium",
                "size": 0.8,
                "params": "345M",
                "caps": ["conversation", "chat"],
                "type": "causal_lm"
            },
            {
                "name": "microsoft/CodeBERT-base",
                "size": 0.5,
                "params": "125M",
                "caps": ["code_understanding"],
                "type": "bert"
            },
            {
                "name": "Salesforce/codet5-base",
                "size": 0.9,
                "params": "220M",
                "caps": ["code_generation", "code_summary"],
                "type": "t5"
            },
            {
                "name": "microsoft/unixcoder-base",
                "size": 0.5,
                "params": "125M",
                "caps": ["code_generation"],
                "type": "bert"
            },
            {
                "name": "EleutherAI/gpt-neo-1.3B",
                "size": 5.2,
                "params": "1.3B",
                "caps": ["text_generation", "code_generation"],
                "type": "causal_lm"
            },
            {
                "name": "facebook/opt-1.3b",
                "size": 2.6,
                "params": "1.3B",
                "caps": ["text_generation"],
                "type": "causal_lm"
            }
        ]

        models = []
        for model_info in popular_models:
            models.append(LocalModelInfo(
                name=model_info["name"],
                model_type="transformers",
                size_gb=model_info["size"],
                parameters=model_info["params"],
                capabilities=model_info["caps"],
                memory_required_gb=model_info["size"] * 2,
                status="available",
                model_path=None
            ))

        return models

    async def load_model(self, model_name: str, progress_callback=None) -> bool:
        """Load a Hugging Face model"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.error("Transformers library not available")
                return False

            if model_name in self.loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return True

            logger.info(f"Loading model: {model_name}")

            # Load in executor to prevent blocking
            loop = asyncio.get_event_loop()

            def _load():
                try:
                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True
                    )

                    # Try different model types
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                            device_map="auto" if self.device == "cuda" else None
                        )
                    except:
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_name,
                            cache_dir=self.cache_dir,
                            trust_remote_code=True,
                            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                        )

                    if self.device != "cuda":
                        model = model.to(self.device)

                    return tokenizer, model

                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    return None, None

            tokenizer, model = await loop.run_in_executor(self.executor, _load)

            if model is None:
                return False

            self.loaded_models[model_name] = {
                "tokenizer": tokenizer,
                "model": model,
                "load_time": time.time()
            }

            logger.info(f"Model {model_name} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    async def generate_completion(self, model_name: str, prompt: str, **kwargs) -> LocalAIResponse:
        """Generate completion using loaded model"""
        try:
            if model_name not in self.loaded_models:
                if not await self.load_model(model_name):
                    raise Exception(f"Failed to load model {model_name}")

            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / (1024**3)

            model_data = self.loaded_models[model_name]
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]

            # Generate in executor
            loop = asyncio.get_event_loop()

            def _generate():
                try:
                    # Tokenize input
                    inputs = tokenizer.encode(prompt, return_tensors="pt")
                    if self.device == "cuda":
                        inputs = inputs.to(self.device)

                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs,
                            max_length=inputs.shape[1] + kwargs.get("max_tokens", 200),
                            temperature=kwargs.get("temperature", 0.7),
                            do_sample=True,
                            top_p=kwargs.get("top_p", 0.9),
                            pad_token_id=tokenizer.eos_token_id
                        )

                    # Decode response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Remove input prompt from response
                    if response.startswith(prompt):
                        response = response[len(prompt):].strip()

                    return response

                except Exception as e:
                    logger.error(f"Generation failed: {e}")
                    return ""

            response_text = await loop.run_in_executor(self.executor, _generate)

            inference_time = time.time() - start_time
            memory_after = psutil.Process().memory_info().rss / (1024**3)
            memory_used = memory_after - memory_before

            # Estimate tokens per second
            estimated_tokens = len(response_text.split()) * 1.3
            tokens_per_second = estimated_tokens / inference_time if inference_time > 0 else 0

            return LocalAIResponse(
                content=response_text,
                model=model_name,
                provider="transformers",
                inference_time=inference_time,
                tokens_per_second=tokens_per_second,
                memory_used_gb=memory_used,
                metadata={
                    "device": self.device,
                    "estimated_tokens": estimated_tokens
                }
            )

        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            raise

    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        try:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Model {model_name} unloaded")
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")


class LocalAIManager:
    """Comprehensive local AI management system"""

    def __init__(self):
        self.ollama = OllamaIntegration()
        self.huggingface = HuggingFaceLocal() if TRANSFORMERS_AVAILABLE else None
        self.active_models = {}
        self.model_metrics = {}

    async def initialize(self) -> bool:
        """Initialize all local AI systems"""
        try:
            success = True

            # Initialize Ollama
            async with self.ollama:
                if not await self.ollama.check_ollama_status():
                    logger.info("Attempting to start Ollama server...")
                    if not await self.ollama.start_ollama_server():
                        logger.warning("Ollama not available")
                        success = False
                else:
                    logger.info("Ollama server is running")

            # Check GPU availability
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU available: {gpu_count} device(s), {gpu_memory:.1f}GB memory")
            else:
                logger.info("Using CPU for local inference")

            return success

        except Exception as e:
            logger.error(f"Failed to initialize local AI: {e}")
            return False

    async def list_available_models(self) -> Dict[str, List[LocalModelInfo]]:
        """List all available local models by provider"""
        models = {
            "ollama": [],
            "transformers": []
        }

        try:
            # Get Ollama models
            async with self.ollama:
                models["ollama"] = await self.ollama.get_available_models()

            # Get HuggingFace models
            if self.huggingface:
                models["transformers"] = self.huggingface.get_popular_models()

        except Exception as e:
            logger.error(f"Error listing models: {e}")

        return models

    async def install_model(self, provider: str, model_name: str, progress_callback=None) -> bool:
        """Install/download a local model"""
        try:
            if provider == "ollama":
                async with self.ollama:
                    return await self.ollama.pull_model(model_name, progress_callback)

            elif provider == "transformers" and self.huggingface:
                return await self.huggingface.load_model(model_name, progress_callback)

            else:
                logger.error(f"Unknown provider: {provider}")
                return False

        except Exception as e:
            logger.error(f"Error installing model {model_name}: {e}")
            return False

    async def generate_with_local_model(self, provider: str, model_name: str,
                                      prompt: str, **kwargs) -> LocalAIResponse:
        """Generate response using local model"""
        try:
            if provider == "ollama":
                async with self.ollama:
                    return await self.ollama.generate_completion(model_name, prompt, **kwargs)

            elif provider == "transformers" and self.huggingface:
                return await self.huggingface.generate_completion(model_name, prompt, **kwargs)

            else:
                raise Exception(f"Provider {provider} not available")

        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            raise

    async def stream_with_local_model(self, provider: str, model_name: str,
                                    prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response from local model"""
        try:
            if provider == "ollama":
                async with self.ollama:
                    async for chunk in self.ollama.stream_completion(model_name, prompt, **kwargs):
                        yield chunk
            else:
                # For non-streaming providers, yield complete response
                response = await self.generate_with_local_model(provider, model_name, prompt, **kwargs)
                yield response.content

        except Exception as e:
            logger.error(f"Local streaming failed: {e}")
            raise

    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all local AI systems"""
        status = {
            "ollama": {"available": False, "models": []},
            "transformers": {"available": TRANSFORMERS_AVAILABLE, "loaded_models": []},
            "system": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }

        try:
            # Check Ollama status
            async with self.ollama:
                status["ollama"]["available"] = await self.ollama.check_ollama_status()
                if status["ollama"]["available"]:
                    models = await self.ollama.get_available_models()
                    status["ollama"]["models"] = [model.name for model in models]

            # Check HuggingFace status
            if self.huggingface:
                status["transformers"]["loaded_models"] = list(self.huggingface.loaded_models.keys())

            # Add GPU info if available
            if torch.cuda.is_available():
                status["system"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                status["system"]["gpu_free_memory_gb"] = (
                    torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated()
                ) / (1024**3)

        except Exception as e:
            logger.error(f"Error getting model status: {e}")

        return status

    def recommend_model(self, use_case: str = "general") -> Dict[str, str]:
        """Recommend best local model for use case"""
        recommendations = {
            "general": {
                "ollama": "llama2:7b-chat",
                "transformers": "microsoft/DialoGPT-medium"
            },
            "code": {
                "ollama": "codellama:7b-code",
                "transformers": "Salesforce/codet5-base"
            },
            "chat": {
                "ollama": "llama2:7b-chat",
                "transformers": "microsoft/DialoGPT-medium"
            },
            "lightweight": {
                "ollama": "tinyllama:1.1b-chat",
                "transformers": "microsoft/CodeBERT-base"
            }
        }

        return recommendations.get(use_case, recommendations["general"])


# Global local AI manager
local_ai_manager = None


async def initialize_local_ai():
    """Initialize local AI systems"""
    global local_ai_manager
    try:
        local_ai_manager = LocalAIManager()
        success = await local_ai_manager.initialize()

        if success:
            logger.info("Local AI systems initialized successfully")
        else:
            logger.warning("Some local AI systems failed to initialize")

        return local_ai_manager
    except Exception as e:
        logger.error(f"Failed to initialize local AI: {e}")
        return None


def get_local_ai_manager():
    """Get global local AI manager"""
    return local_ai_manager