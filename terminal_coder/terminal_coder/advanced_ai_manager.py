"""
Advanced AI Manager - Ultra-Advanced AI Integration
Enterprise-grade AI provider management with automatic model detection, load balancing, and intelligent routing
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncGenerator, Final
import aiohttp
import httpx
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from cryptography.fernet import Fernet
from datetime import datetime, timedelta

try:
    import openai
    import anthropic
    import google.generativeai as genai
    import cohere
    PROVIDERS_AVAILABLE = True
except ImportError:
    PROVIDERS_AVAILABLE = False


class AIProvider(Enum):
    """Advanced AI provider enumeration"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    TOGETHER = "together"
    PERPLEXITY = "perplexity"
    GROQ = "groq"
    MISTRAL = "mistral"


class ModelCapability(Enum):
    """AI model capabilities"""
    CODE_GENERATION = auto()
    CODE_ANALYSIS = auto()
    DEBUGGING = auto()
    DOCUMENTATION = auto()
    SYSTEM_ADMINISTRATION = auto()
    SECURITY_ANALYSIS = auto()
    PERFORMANCE_OPTIMIZATION = auto()
    LINUX_EXPERTISE = auto()
    CONTAINERIZATION = auto()
    DEVOPS = auto()


@dataclass(slots=True, frozen=True)
class AIModel:
    """Advanced AI model configuration"""
    name: str
    provider: AIProvider
    max_tokens: int
    context_window: int
    cost_per_1k_tokens: float
    capabilities: frozenset[ModelCapability]
    linux_optimized: bool = False
    coding_score: int = 0  # 1-10 rating for coding tasks
    speed_rating: int = 0  # 1-10 rating for response speed
    availability_regions: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True)
class APICredentials:
    """Secure API credentials storage"""
    provider: AIProvider
    api_key: str
    organization_id: str | None = None
    project_id: str | None = None
    base_url: str | None = None
    encrypted: bool = False
    last_verified: datetime | None = None
    rate_limit_remaining: int = 0
    rate_limit_reset: datetime | None = None


@dataclass(slots=True)
class ModelPerformanceMetrics:
    """Model performance tracking"""
    model_name: str
    provider: AIProvider
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    total_tokens_used: int = 0
    cost_accumulated: float = 0.0
    linux_task_success_rate: float = 0.0
    last_used: datetime | None = None


class AdvancedAIManager:
    """Ultra-advanced AI provider management system"""

    # Latest model configurations with Linux optimization ratings
    AVAILABLE_MODELS: Final[dict[AIProvider, list[AIModel]]] = {
        AIProvider.OPENAI: [
            AIModel(
                "gpt-4o", AIProvider.OPENAI, 4096, 128000, 0.03,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS,
                          ModelCapability.DEBUGGING, ModelCapability.LINUX_EXPERTISE]),
                linux_optimized=True, coding_score=9, speed_rating=8
            ),
            AIModel(
                "gpt-4o-mini", AIProvider.OPENAI, 16384, 128000, 0.0015,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.DEBUGGING]),
                linux_optimized=True, coding_score=8, speed_rating=9
            ),
            AIModel(
                "gpt-4-turbo", AIProvider.OPENAI, 4096, 128000, 0.01,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS,
                          ModelCapability.SYSTEM_ADMINISTRATION]),
                linux_optimized=True, coding_score=9, speed_rating=7
            ),
        ],
        AIProvider.ANTHROPIC: [
            AIModel(
                "claude-3-5-sonnet-20241022", AIProvider.ANTHROPIC, 8192, 200000, 0.015,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS,
                          ModelCapability.DEBUGGING, ModelCapability.LINUX_EXPERTISE,
                          ModelCapability.SYSTEM_ADMINISTRATION]),
                linux_optimized=True, coding_score=10, speed_rating=8
            ),
            AIModel(
                "claude-3-5-haiku-20241022", AIProvider.ANTHROPIC, 8192, 200000, 0.0025,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.DEBUGGING]),
                linux_optimized=True, coding_score=8, speed_rating=10
            ),
            AIModel(
                "claude-3-opus-20240229", AIProvider.ANTHROPIC, 4096, 200000, 0.075,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS,
                          ModelCapability.SECURITY_ANALYSIS, ModelCapability.LINUX_EXPERTISE]),
                linux_optimized=True, coding_score=10, speed_rating=6
            ),
        ],
        AIProvider.GOOGLE: [
            AIModel(
                "gemini-1.5-pro", AIProvider.GOOGLE, 8192, 2000000, 0.00125,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS,
                          ModelCapability.LINUX_EXPERTISE]),
                linux_optimized=True, coding_score=8, speed_rating=7
            ),
            AIModel(
                "gemini-1.5-flash", AIProvider.GOOGLE, 8192, 1000000, 0.000075,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.DEBUGGING]),
                linux_optimized=True, coding_score=7, speed_rating=9
            ),
        ],
        AIProvider.COHERE: [
            AIModel(
                "command-r-plus", AIProvider.COHERE, 4096, 128000, 0.015,
                frozenset([ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS]),
                linux_optimized=False, coding_score=7, speed_rating=8
            ),
        ],
    }

    def __init__(self, config_dir: Path | None = None) -> None:
        self.console = Console()
        self.config_dir = config_dir or Path.home() / ".terminal_coder"
        self.config_dir.mkdir(exist_ok=True)

        self.credentials_file = self.config_dir / "encrypted_credentials.json"
        self.metrics_file = self.config_dir / "model_metrics.json"
        self.preferences_file = self.config_dir / "ai_preferences.json"

        # Initialize encryption key
        self.key_file = self.config_dir / ".ai_key"
        self._init_encryption()

        # Load existing data
        self.credentials: dict[AIProvider, APICredentials] = {}
        self.metrics: dict[str, ModelPerformanceMetrics] = {}
        self.preferences: dict[str, Any] = {}

        self._load_credentials()
        self._load_metrics()
        self._load_preferences()

        # Performance tracking
        self.session_start = datetime.now()
        self.active_sessions: dict[str, datetime] = {}

    def _init_encryption(self) -> None:
        """Initialize or load encryption key"""
        if self.key_file.exists():
            self.cipher = Fernet(self.key_file.read_bytes())
        else:
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            self.key_file.chmod(0o600)  # Secure permissions
            self.cipher = Fernet(key)

    def _load_credentials(self) -> None:
        """Load encrypted API credentials"""
        if not self.credentials_file.exists():
            return

        try:
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode())

            for provider_name, cred_data in data.items():
                provider = AIProvider(provider_name)
                self.credentials[provider] = APICredentials(
                    provider=provider,
                    api_key=cred_data["api_key"],
                    organization_id=cred_data.get("organization_id"),
                    project_id=cred_data.get("project_id"),
                    base_url=cred_data.get("base_url"),
                    encrypted=True,
                    last_verified=datetime.fromisoformat(cred_data["last_verified"]) if cred_data.get("last_verified") else None
                )
        except Exception as e:
            self.console.print(f"[red]Failed to load credentials: {e}[/red]")

    def _save_credentials(self) -> None:
        """Save encrypted API credentials"""
        data = {}
        for provider, cred in self.credentials.items():
            data[provider.value] = {
                "api_key": cred.api_key,
                "organization_id": cred.organization_id,
                "project_id": cred.project_id,
                "base_url": cred.base_url,
                "last_verified": cred.last_verified.isoformat() if cred.last_verified else None
            }

        encrypted_data = self.cipher.encrypt(json.dumps(data).encode())
        with open(self.credentials_file, 'wb') as f:
            f.write(encrypted_data)
        self.credentials_file.chmod(0o600)

    def _load_metrics(self) -> None:
        """Load model performance metrics"""
        if not self.metrics_file.exists():
            return

        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)

            for model_key, metrics_data in data.items():
                self.metrics[model_key] = ModelPerformanceMetrics(
                    model_name=metrics_data["model_name"],
                    provider=AIProvider(metrics_data["provider"]),
                    total_requests=metrics_data.get("total_requests", 0),
                    successful_requests=metrics_data.get("successful_requests", 0),
                    failed_requests=metrics_data.get("failed_requests", 0),
                    average_response_time=metrics_data.get("average_response_time", 0.0),
                    total_tokens_used=metrics_data.get("total_tokens_used", 0),
                    cost_accumulated=metrics_data.get("cost_accumulated", 0.0),
                    linux_task_success_rate=metrics_data.get("linux_task_success_rate", 0.0),
                    last_used=datetime.fromisoformat(metrics_data["last_used"]) if metrics_data.get("last_used") else None
                )
        except Exception as e:
            self.console.print(f"[red]Failed to load metrics: {e}[/red]")

    def _save_metrics(self) -> None:
        """Save model performance metrics"""
        data = {}
        for model_key, metrics in self.metrics.items():
            data[model_key] = {
                "model_name": metrics.model_name,
                "provider": metrics.provider.value,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "average_response_time": metrics.average_response_time,
                "total_tokens_used": metrics.total_tokens_used,
                "cost_accumulated": metrics.cost_accumulated,
                "linux_task_success_rate": metrics.linux_task_success_rate,
                "last_used": metrics.last_used.isoformat() if metrics.last_used else None
            }

        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_preferences(self) -> None:
        """Load user AI preferences"""
        if not self.preferences_file.exists():
            self.preferences = {
                "preferred_provider": None,
                "preferred_model": None,
                "auto_select_best": True,
                "prioritize_speed": False,
                "prioritize_cost": False,
                "linux_tasks_only": True,
                "fallback_enabled": True,
                "max_cost_per_request": 0.50
            }
            self._save_preferences()
        else:
            with open(self.preferences_file, 'r') as f:
                self.preferences = json.load(f)

    def _save_preferences(self) -> None:
        """Save user AI preferences"""
        with open(self.preferences_file, 'w') as f:
            json.dump(self.preferences, f, indent=2)

    async def setup_provider_interactive(self) -> None:
        """Interactive AI provider setup with enhanced UX"""
        self.console.print(Panel.fit(
            "[bold cyan]🤖 Advanced AI Provider Setup[/bold cyan]\n"
            "[dim]Configure your AI providers for optimal Linux development[/dim]",
            border_style="cyan"
        ))

        # Show available providers
        table = Table(title="Available AI Providers")
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Models Available", style="green")
        table.add_column("Linux Optimized", style="blue")
        table.add_column("Best For", style="yellow")

        provider_info = {
            AIProvider.OPENAI: ("GPT-4o, GPT-4 Turbo", "✅", "General coding, debugging"),
            AIProvider.ANTHROPIC: ("Claude 3.5 Sonnet", "✅", "Complex analysis, Linux expertise"),
            AIProvider.GOOGLE: ("Gemini 1.5 Pro", "✅", "Large context, documentation"),
            AIProvider.COHERE: ("Command-R+", "⚠️", "Basic coding tasks"),
        }

        for provider, (models, linux_opt, best_for) in provider_info.items():
            table.add_row(provider.value.upper(), models, linux_opt, best_for)

        self.console.print(table)

        # Provider selection
        while True:
            provider_choice = Prompt.ask(
                "\n[cyan]Select AI provider to configure[/cyan]",
                choices=[p.value for p in AIProvider],
                default="anthropic"
            )

            provider = AIProvider(provider_choice)

            # Configure the selected provider
            await self._configure_provider(provider)

            # Ask if they want to configure more
            if not Confirm.ask("\n[yellow]Configure another provider?[/yellow]"):
                break

        # Verify all configurations
        await self._verify_all_providers()

        # Set preferences
        self._configure_preferences()

    async def _configure_provider(self, provider: AIProvider) -> None:
        """Configure a specific AI provider"""
        self.console.print(f"\n[bold blue]Configuring {provider.value.upper()}[/bold blue]")

        # Provider-specific setup instructions
        setup_info = {
            AIProvider.OPENAI: {
                "url": "https://platform.openai.com/api-keys",
                "format": "sk-...",
                "fields": ["API Key", "Organization ID (optional)"]
            },
            AIProvider.ANTHROPIC: {
                "url": "https://console.anthropic.com/settings/keys",
                "format": "sk-ant-...",
                "fields": ["API Key"]
            },
            AIProvider.GOOGLE: {
                "url": "https://ai.google.dev/",
                "format": "AI...",
                "fields": ["API Key"]
            },
            AIProvider.COHERE: {
                "url": "https://dashboard.cohere.ai/api-keys",
                "format": "...",
                "fields": ["API Key"]
            }
        }

        info = setup_info.get(provider, {"url": "", "format": "", "fields": ["API Key"]})

        self.console.print(Panel(
            f"[bold]Get your API key from:[/bold] {info['url']}\n"
            f"[bold]Expected format:[/bold] {info['format']}\n"
            f"[dim]Keep this key secure and never share it![/dim]",
            title=f"{provider.value.upper()} Setup",
            border_style="blue"
        ))

        # Get API key with validation
        while True:
            api_key = Prompt.ask(
                f"[green]Enter your {provider.value.upper()} API key[/green]",
                password=True
            )

            if not api_key:
                self.console.print("[red]API key cannot be empty![/red]")
                continue

            # Basic format validation
            if provider == AIProvider.OPENAI and not api_key.startswith("sk-"):
                self.console.print("[yellow]⚠️  OpenAI API keys usually start with 'sk-'[/yellow]")
                if not Confirm.ask("Continue anyway?"):
                    continue
            elif provider == AIProvider.ANTHROPIC and not api_key.startswith("sk-ant-"):
                self.console.print("[yellow]⚠️  Anthropic API keys usually start with 'sk-ant-'[/yellow]")
                if not Confirm.ask("Continue anyway?"):
                    continue

            break

        # Get optional fields
        org_id = None
        project_id = None
        base_url = None

        if provider == AIProvider.OPENAI:
            org_id = Prompt.ask(
                "[green]Organization ID (optional, press Enter to skip)[/green]",
                default=""
            ) or None

        # Custom base URL for enterprise/local deployments
        if Confirm.ask(f"[yellow]Use custom base URL for {provider.value.upper()}?[/yellow]"):
            base_url = Prompt.ask("[green]Enter base URL[/green]")

        # Store credentials
        self.credentials[provider] = APICredentials(
            provider=provider,
            api_key=api_key,
            organization_id=org_id,
            project_id=project_id,
            base_url=base_url
        )

        self._save_credentials()
        self.console.print(f"[green]✅ {provider.value.upper()} configured successfully![/green]")

    async def _verify_all_providers(self) -> None:
        """Verify all configured providers and detect available models"""
        self.console.print("\n[cyan]🔍 Verifying providers and detecting models...[/cyan]")

        verification_results = {}

        for provider, cred in self.credentials.items():
            with self.console.status(f"Verifying {provider.value.upper()}..."):
                try:
                    models = await self._detect_available_models(provider, cred)
                    verification_results[provider] = {
                        "status": "✅ Active",
                        "models": len(models),
                        "verified_at": datetime.now()
                    }
                    cred.last_verified = datetime.now()
                except Exception as e:
                    verification_results[provider] = {
                        "status": f"❌ Failed: {str(e)[:50]}",
                        "models": 0,
                        "verified_at": None
                    }

        # Display verification results
        table = Table(title="🔍 Provider Verification Results")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Models Available", style="blue")
        table.add_column("Verified", style="yellow")

        for provider, result in verification_results.items():
            table.add_row(
                provider.value.upper(),
                result["status"],
                str(result["models"]),
                result["verified_at"].strftime("%H:%M:%S") if result["verified_at"] else "Never"
            )

        self.console.print(table)
        self._save_credentials()

    async def _detect_available_models(self, provider: AIProvider, cred: APICredentials) -> list[str]:
        """Detect available models for a provider"""
        try:
            if provider == AIProvider.OPENAI:
                client = openai.OpenAI(
                    api_key=cred.api_key,
                    organization=cred.organization_id,
                    base_url=cred.base_url
                )
                models = client.models.list()
                return [model.id for model in models.data if 'gpt' in model.id.lower()]

            elif provider == AIProvider.ANTHROPIC:
                # Anthropic doesn't have a models endpoint, return known models
                return ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]

            elif provider == AIProvider.GOOGLE:
                genai.configure(api_key=cred.api_key)
                models = genai.list_models()
                return [model.name.replace("models/", "") for model in models if "gemini" in model.name.lower()]

            elif provider == AIProvider.COHERE:
                client = cohere.Client(api_key=cred.api_key)
                # Cohere doesn't have a models list endpoint, return known models
                return ["command-r-plus", "command-r", "command"]

        except Exception as e:
            raise Exception(f"Failed to verify {provider.value}: {str(e)}")

        return []

    def _configure_preferences(self) -> None:
        """Configure AI usage preferences"""
        self.console.print("\n[bold cyan]⚙️ Configure AI Preferences[/bold cyan]")

        # Auto-select best model
        self.preferences["auto_select_best"] = Confirm.ask(
            "[green]Auto-select best model for each task?[/green]",
            default=self.preferences.get("auto_select_best", True)
        )

        if not self.preferences["auto_select_best"]:
            # Manual provider/model selection
            available_providers = list(self.credentials.keys())
            if available_providers:
                provider_choice = Prompt.ask(
                    "[green]Preferred provider[/green]",
                    choices=[p.value for p in available_providers],
                    default=available_providers[0].value
                )
                self.preferences["preferred_provider"] = provider_choice

        # Performance vs cost preferences
        priority = Prompt.ask(
            "[green]Prioritize[/green]",
            choices=["speed", "cost", "quality", "balanced"],
            default="balanced"
        )

        self.preferences["prioritize_speed"] = priority == "speed"
        self.preferences["prioritize_cost"] = priority == "cost"
        self.preferences["prioritize_quality"] = priority == "quality"

        # Linux-specific optimizations
        self.preferences["linux_tasks_only"] = Confirm.ask(
            "[green]Use only Linux-optimized models?[/green]",
            default=True
        )

        # Cost controls
        if Confirm.ask("[green]Set maximum cost per request?[/green]"):
            max_cost = float(Prompt.ask(
                "[green]Maximum cost per request (USD)[/green]",
                default=str(self.preferences.get("max_cost_per_request", 0.50))
            ))
            self.preferences["max_cost_per_request"] = max_cost

        self._save_preferences()
        self.console.print("[green]✅ Preferences saved![/green]")

    def get_optimal_model(self, task_type: ModelCapability | None = None,
                         context_size: int = 0, max_cost: float | None = None) -> AIModel | None:
        """Intelligently select the optimal model for a task"""
        available_models = []

        # Filter models by available credentials
        for provider, models in self.AVAILABLE_MODELS.items():
            if provider in self.credentials:
                available_models.extend(models)

        if not available_models:
            return None

        # Apply filters
        candidates = available_models.copy()

        # Filter by capability
        if task_type:
            candidates = [m for m in candidates if task_type in m.capabilities]

        # Filter by Linux optimization preference
        if self.preferences.get("linux_tasks_only", True):
            linux_candidates = [m for m in candidates if m.linux_optimized]
            if linux_candidates:
                candidates = linux_candidates

        # Filter by context size
        if context_size > 0:
            candidates = [m for m in candidates if m.context_window >= context_size]

        # Filter by cost
        if max_cost or self.preferences.get("max_cost_per_request"):
            cost_limit = max_cost or self.preferences["max_cost_per_request"]
            candidates = [m for m in candidates if m.cost_per_1k_tokens <= cost_limit]

        if not candidates:
            return None

        # Score and rank models
        scored_models = []
        for model in candidates:
            score = self._calculate_model_score(model, task_type)
            scored_models.append((model, score))

        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)

        return scored_models[0][0] if scored_models else None

    def _calculate_model_score(self, model: AIModel, task_type: ModelCapability | None) -> float:
        """Calculate a score for model selection"""
        score = 0.0

        # Base coding score
        score += model.coding_score * 10

        # Linux optimization bonus
        if model.linux_optimized:
            score += 20

        # Speed consideration
        if self.preferences.get("prioritize_speed"):
            score += model.speed_rating * 5

        # Cost consideration (lower cost = higher score)
        if self.preferences.get("prioritize_cost"):
            score += max(0, 10 - model.cost_per_1k_tokens * 100)

        # Historical performance bonus
        model_key = f"{model.provider.value}:{model.name}"
        if model_key in self.metrics:
            metrics = self.metrics[model_key]
            if metrics.total_requests > 10:  # Enough data
                success_rate = metrics.successful_requests / metrics.total_requests
                score += success_rate * 15

                # Linux task success bonus
                score += metrics.linux_task_success_rate * 10

        # Task-specific capability bonus
        if task_type and task_type in model.capabilities:
            score += 25

        return score

    def display_ai_status(self) -> None:
        """Display comprehensive AI system status"""
        self.console.print(Panel.fit(
            "[bold cyan]🤖 AI System Status[/bold cyan]",
            border_style="cyan"
        ))

        # Provider status
        provider_table = Table(title="🔗 Provider Status")
        provider_table.add_column("Provider", style="cyan")
        provider_table.add_column("Status", style="green")
        provider_table.add_column("Last Verified", style="blue")
        provider_table.add_column("Models", style="yellow")

        for provider in AIProvider:
            if provider in self.credentials:
                cred = self.credentials[provider]
                status = "✅ Active" if cred.last_verified else "⚠️ Unverified"
                verified = cred.last_verified.strftime("%Y-%m-%d %H:%M") if cred.last_verified else "Never"
                model_count = len(self.AVAILABLE_MODELS.get(provider, []))
            else:
                status = "❌ Not configured"
                verified = "N/A"
                model_count = 0

            provider_table.add_row(provider.value.upper(), status, verified, str(model_count))

        self.console.print(provider_table)

        # Performance metrics
        if self.metrics:
            metrics_table = Table(title="📊 Performance Metrics")
            metrics_table.add_column("Model", style="cyan")
            metrics_table.add_column("Requests", style="green")
            metrics_table.add_column("Success Rate", style="blue")
            metrics_table.add_column("Avg Response", style="yellow")
            metrics_table.add_column("Cost", style="red")

            for model_key, metrics in sorted(self.metrics.items(),
                                           key=lambda x: x[1].total_requests, reverse=True)[:10]:
                success_rate = f"{(metrics.successful_requests / max(metrics.total_requests, 1)) * 100:.1f}%"
                avg_response = f"{metrics.average_response_time:.2f}s"
                cost = f"${metrics.cost_accumulated:.4f}"

                metrics_table.add_row(
                    metrics.model_name,
                    str(metrics.total_requests),
                    success_rate,
                    avg_response,
                    cost
                )

            self.console.print(metrics_table)

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self._save_metrics()
        self._save_preferences()