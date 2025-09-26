#!/usr/bin/env python3
"""
Windows AI Integration
Handles AI provider integrations with Windows-specific optimizations
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
import subprocess

# Windows-specific imports
import winreg
from cryptography.fernet import Fernet
import keyring

# AI provider imports
import openai
import anthropic
import google.generativeai as genai
import cohere

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm


@dataclass
class AIProviderConfig:
    """AI provider configuration"""
    name: str
    base_url: str
    models: List[str]
    api_key_env_var: str
    supports_streaming: bool = True
    max_tokens: int = 8000
    rate_limit_per_minute: int = 60
    windows_optimized: bool = True


@dataclass
class AIResponse:
    """AI response data"""
    content: str
    provider: str
    model: str
    tokens_used: int
    response_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class WindowsAIIntegration:
    """Advanced AI integration optimized for Windows"""

    def __init__(self):
        self.console = Console()
        self.logger = logging.getLogger(__name__)

        # Windows-specific paths
        self.config_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'TerminalCoder'
        self.api_keys_file = self.config_dir / 'api_keys.encrypted'
        self.ai_config_file = self.config_dir / 'ai_config.json'

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize encryption for API keys
        self._init_encryption()

        # Load AI providers configuration
        self.providers = self._load_ai_providers()

        # Load API keys
        self.api_keys = self._load_api_keys()

        # Initialize AI clients
        self.clients = {}
        self._initialize_clients()

    def _init_encryption(self):
        """Initialize encryption for API keys storage"""
        key_file = self.config_dir / 'key.key'

        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            # Generate new encryption key
            self.encryption_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)

            # Make key file read-only for current user (Windows)
            try:
                import stat
                os.chmod(key_file, stat.S_IREAD | stat.S_IWRITE)
            except Exception:
                pass

        self.cipher_suite = Fernet(self.encryption_key)

    def _load_ai_providers(self) -> Dict[str, AIProviderConfig]:
        """Load AI provider configurations"""
        providers = {
            'openai': AIProviderConfig(
                name='OpenAI',
                base_url='https://api.openai.com/v1',
                models=[
                    'gpt-4o',
                    'gpt-4o-mini',
                    'gpt-4-turbo',
                    'gpt-4',
                    'gpt-3.5-turbo'
                ],
                api_key_env_var='OPENAI_API_KEY',
                supports_streaming=True,
                max_tokens=128000,
                rate_limit_per_minute=60,
                windows_optimized=True
            ),
            'anthropic': AIProviderConfig(
                name='Anthropic',
                base_url='https://api.anthropic.com/v1',
                models=[
                    'claude-3-5-sonnet-20241022',
                    'claude-3-5-haiku-20241022',
                    'claude-3-opus-20240229',
                    'claude-3-sonnet-20240229',
                    'claude-3-haiku-20240307'
                ],
                api_key_env_var='ANTHROPIC_API_KEY',
                supports_streaming=True,
                max_tokens=200000,
                rate_limit_per_minute=50,
                windows_optimized=True
            ),
            'google': AIProviderConfig(
                name='Google',
                base_url='https://generativelanguage.googleapis.com/v1',
                models=[
                    'gemini-1.5-pro',
                    'gemini-1.5-flash',
                    'gemini-1.0-pro'
                ],
                api_key_env_var='GOOGLE_API_KEY',
                supports_streaming=True,
                max_tokens=32000,
                rate_limit_per_minute=60,
                windows_optimized=True
            ),
            'cohere': AIProviderConfig(
                name='Cohere',
                base_url='https://api.cohere.ai/v1',
                models=[
                    'command-r-plus',
                    'command-r',
                    'command'
                ],
                api_key_env_var='COHERE_API_KEY',
                supports_streaming=True,
                max_tokens=128000,
                rate_limit_per_minute=100,
                windows_optimized=True
            )
        }

        return providers

    def _load_api_keys(self) -> Dict[str, str]:
        """Load encrypted API keys"""
        api_keys = {}

        # Try to load from encrypted file first
        if self.api_keys_file.exists():
            try:
                with open(self.api_keys_file, 'rb') as f:
                    encrypted_data = f.read()
                    decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                    api_keys = json.loads(decrypted_data.decode())
            except Exception as e:
                self.logger.warning(f"Could not decrypt API keys file: {e}")

        # Fallback to Windows credential store
        for provider_id, provider_config in self.providers.items():
            if provider_id not in api_keys:
                try:
                    key = keyring.get_password("TerminalCoder", f"{provider_id}_api_key")
                    if key:
                        api_keys[provider_id] = key
                except Exception:
                    pass

        # Fallback to environment variables
        for provider_id, provider_config in self.providers.items():
            if provider_id not in api_keys:
                env_key = os.environ.get(provider_config.api_key_env_var)
                if env_key:
                    api_keys[provider_id] = env_key

        return api_keys

    def _save_api_keys(self):
        """Save API keys with encryption"""
        try:
            # Save to encrypted file
            json_data = json.dumps(self.api_keys).encode()
            encrypted_data = self.cipher_suite.encrypt(json_data)

            with open(self.api_keys_file, 'wb') as f:
                f.write(encrypted_data)

            # Also save to Windows credential store as backup
            for provider_id, api_key in self.api_keys.items():
                try:
                    keyring.set_password("TerminalCoder", f"{provider_id}_api_key", api_key)
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(f"Error saving API keys: {e}")

    def _initialize_clients(self):
        """Initialize AI provider clients"""
        self.clients = {}

        # OpenAI
        if 'openai' in self.api_keys:
            try:
                self.clients['openai'] = openai.OpenAI(
                    api_key=self.api_keys['openai']
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI client: {e}")

        # Anthropic
        if 'anthropic' in self.api_keys:
            try:
                self.clients['anthropic'] = anthropic.Anthropic(
                    api_key=self.api_keys['anthropic']
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize Anthropic client: {e}")

        # Google
        if 'google' in self.api_keys:
            try:
                genai.configure(api_key=self.api_keys['google'])
                self.clients['google'] = genai
            except Exception as e:
                self.logger.warning(f"Could not initialize Google client: {e}")

        # Cohere
        if 'cohere' in self.api_keys:
            try:
                self.clients['cohere'] = cohere.Client(
                    api_key=self.api_keys['cohere']
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize Cohere client: {e}")

    async def setup_ai_providers(self):
        """Interactive setup for AI providers"""
        self.console.print(Panel("🤖 AI Providers Setup", style="blue"))

        for provider_id, provider_config in self.providers.items():
            current_key = self.api_keys.get(provider_id, "Not configured")
            masked_key = current_key[:8] + "..." if current_key != "Not configured" else "Not configured"

            self.console.print(f"\n[cyan]{provider_config.name}[/cyan]")
            self.console.print(f"Current status: [{'green' if current_key != 'Not configured' else 'red'}]{masked_key}[/]")
            self.console.print(f"Models available: {', '.join(provider_config.models[:3])}")

            if Confirm.ask(f"Configure {provider_config.name}?"):
                api_key = Prompt.ask(f"Enter {provider_config.name} API key", password=True)
                if api_key.strip():
                    self.api_keys[provider_id] = api_key.strip()
                    self.console.print(f"[green]✅ {provider_config.name} API key saved securely[/green]")

        # Save API keys
        self._save_api_keys()

        # Reinitialize clients
        self._initialize_clients()

        self.console.print(Panel("✅ AI Providers setup complete!", style="green"))

    async def test_ai_connections(self):
        """Test connections to all configured AI providers"""
        self.console.print(Panel("🔍 Testing AI Provider Connections", style="yellow"))

        results = {}

        for provider_id, client in self.clients.items():
            provider_config = self.providers[provider_id]

            with self.console.status(f"[bold blue]Testing {provider_config.name}..."):
                try:
                    success, response_time = await self._test_provider_connection(provider_id)
                    results[provider_id] = {
                        'status': 'success' if success else 'failed',
                        'response_time': response_time,
                        'provider_name': provider_config.name
                    }
                except Exception as e:
                    results[provider_id] = {
                        'status': 'error',
                        'error': str(e),
                        'provider_name': provider_config.name
                    }

        # Display results
        table = Table(title="🤖 AI Provider Connection Test Results", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Status", style="white")
        table.add_column("Response Time", style="yellow")
        table.add_column("Notes", style="green")

        for provider_id, result in results.items():
            status_color = "green" if result['status'] == 'success' else "red"
            status_text = f"[{status_color}]{result['status'].upper()}[/{status_color}]"

            response_time = f"{result.get('response_time', 0):.2f}s" if result.get('response_time') else "N/A"
            notes = result.get('error', 'Connection successful') if result['status'] != 'success' else 'OK'

            table.add_row(
                result['provider_name'],
                status_text,
                response_time,
                notes[:50] + "..." if len(notes) > 50 else notes
            )

        self.console.print(table)

    async def _test_provider_connection(self, provider_id: str) -> tuple[bool, float]:
        """Test connection to a specific AI provider"""
        start_time = asyncio.get_event_loop().time()

        try:
            if provider_id == 'openai':
                response = await asyncio.to_thread(
                    self.clients['openai'].chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello, this is a connection test."}],
                    max_tokens=10
                )
                success = True

            elif provider_id == 'anthropic':
                response = await asyncio.to_thread(
                    self.clients['anthropic'].messages.create,
                    model="claude-3-haiku-20240307",
                    messages=[{"role": "user", "content": "Hello, this is a connection test."}],
                    max_tokens=10
                )
                success = True

            elif provider_id == 'google':
                model = genai.GenerativeModel('gemini-1.0-pro')
                response = await asyncio.to_thread(
                    model.generate_content,
                    "Hello, this is a connection test."
                )
                success = True

            elif provider_id == 'cohere':
                response = await asyncio.to_thread(
                    self.clients['cohere'].generate,
                    model='command',
                    prompt="Hello, this is a connection test.",
                    max_tokens=10
                )
                success = True

            else:
                success = False

        except Exception as e:
            success = False
            self.logger.warning(f"Connection test failed for {provider_id}: {e}")

        end_time = asyncio.get_event_loop().time()
        response_time = end_time - start_time

        return success, response_time

    async def chat_with_ai(self, provider_id: str, model: str, message: str, context: Optional[List[Dict]] = None) -> Optional[AIResponse]:
        """Chat with AI provider"""
        if provider_id not in self.clients:
            self.console.print(f"[red]Error: {provider_id} client not available[/red]")
            return None

        provider_config = self.providers[provider_id]
        start_time = asyncio.get_event_loop().time()

        try:
            if provider_id == 'openai':
                messages = context or []
                messages.append({"role": "user", "content": message})

                response = await asyncio.to_thread(
                    self.clients['openai'].chat.completions.create,
                    model=model,
                    messages=messages,
                    max_tokens=provider_config.max_tokens
                )

                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens

            elif provider_id == 'anthropic':
                messages = context or []
                messages.append({"role": "user", "content": message})

                response = await asyncio.to_thread(
                    self.clients['anthropic'].messages.create,
                    model=model,
                    messages=messages,
                    max_tokens=provider_config.max_tokens
                )

                content = response.content[0].text
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

            elif provider_id == 'google':
                model_instance = genai.GenerativeModel(model)
                response = await asyncio.to_thread(
                    model_instance.generate_content,
                    message
                )

                content = response.text
                tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0

            elif provider_id == 'cohere':
                response = await asyncio.to_thread(
                    self.clients['cohere'].generate,
                    model=model,
                    prompt=message,
                    max_tokens=provider_config.max_tokens
                )

                content = response.generations[0].text
                tokens_used = response.meta.billed_units.output_tokens if hasattr(response.meta, 'billed_units') else 0

            else:
                return None

            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time

            return AIResponse(
                content=content,
                provider=provider_id,
                model=model,
                tokens_used=tokens_used,
                response_time=response_time
            )

        except Exception as e:
            self.logger.error(f"Error calling {provider_id} API: {e}")
            self.console.print(f"[red]Error: {e}[/red]")
            return None

    async def start_assistant(self):
        """Start interactive AI assistant"""
        if not self.clients:
            self.console.print("[red]No AI providers configured. Please run setup first.[/red]")
            return

        self.console.print(Panel("🤖 AI Assistant - Windows Edition", style="blue"))

        # Let user choose provider and model
        available_providers = list(self.clients.keys())

        if len(available_providers) == 1:
            provider_id = available_providers[0]
        else:
            provider_id = Prompt.ask(
                "Choose AI provider",
                choices=available_providers,
                default=available_providers[0]
            )

        provider_config = self.providers[provider_id]
        available_models = provider_config.models

        model = Prompt.ask(
            f"Choose {provider_config.name} model",
            choices=available_models,
            default=available_models[0]
        )

        self.console.print(f"[cyan]Using {provider_config.name} - {model}[/cyan]")
        self.console.print("[yellow]Type 'exit' to quit, 'help' for commands[/yellow]\n")

        conversation_context = []

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")

                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'help':
                    self._show_assistant_help()
                    continue
                elif user_input.lower() == 'clear':
                    conversation_context = []
                    self.console.print("[yellow]Conversation history cleared[/yellow]")
                    continue
                elif user_input.lower() == 'save':
                    await self._save_conversation(conversation_context)
                    continue
                elif user_input.lower() == 'models':
                    self._show_available_models()
                    continue

                # Get AI response
                with self.console.status(f"[bold blue]{provider_config.name} is thinking..."):
                    ai_response = await self.chat_with_ai(
                        provider_id, model, user_input, conversation_context.copy()
                    )

                if ai_response:
                    self.console.print(Panel(
                        ai_response.content,
                        title=f"🤖 {provider_config.name} ({model})",
                        border_style="blue"
                    ))

                    # Add to conversation context
                    conversation_context.append({"role": "user", "content": user_input})
                    conversation_context.append({"role": "assistant", "content": ai_response.content})

                    # Keep context manageable (last 10 exchanges)
                    if len(conversation_context) > 20:
                        conversation_context = conversation_context[-20:]

                    # Show usage stats
                    self.console.print(f"[dim]Tokens: {ai_response.tokens_used}, Time: {ai_response.response_time:.2f}s[/dim]")

            except KeyboardInterrupt:
                if Confirm.ask("\nExit AI Assistant?"):
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    def _show_assistant_help(self):
        """Show assistant help commands"""
        help_text = """
[cyan]AI Assistant Commands:[/cyan]

• [yellow]exit/quit[/yellow] - Exit the assistant
• [yellow]help[/yellow] - Show this help message
• [yellow]clear[/yellow] - Clear conversation history
• [yellow]save[/yellow] - Save conversation to file
• [yellow]models[/yellow] - Show available models

[cyan]Tips:[/cyan]
• Be specific in your questions for better responses
• Use conversation context for follow-up questions
• The assistant remembers the last 10 exchanges
        """
        self.console.print(Panel(help_text, style="cyan"))

    def _show_available_models(self):
        """Show available models for all providers"""
        table = Table(title="🤖 Available AI Models", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Models", style="white")
        table.add_column("Status", style="green")

        for provider_id, provider_config in self.providers.items():
            status = "✅ Available" if provider_id in self.clients else "❌ Not configured"
            models_text = ", ".join(provider_config.models[:3]) + f" (+{len(provider_config.models)-3} more)" if len(provider_config.models) > 3 else ", ".join(provider_config.models)

            table.add_row(
                provider_config.name,
                models_text,
                status
            )

        self.console.print(table)

    async def _save_conversation(self, conversation: List[Dict]):
        """Save conversation to file"""
        if not conversation:
            self.console.print("[yellow]No conversation to save[/yellow]")
            return

        # Create conversations directory
        conversations_dir = self.config_dir / 'conversations'
        conversations_dir.mkdir(exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = conversations_dir / f"conversation_{timestamp}.json"

        try:
            conversation_data = {
                'timestamp': datetime.now().isoformat(),
                'messages': conversation,
                'platform': 'Windows'
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            self.console.print(f"[green]✅ Conversation saved to {filename}[/green]")

        except Exception as e:
            self.console.print(f"[red]Error saving conversation: {e}[/red]")

    def get_provider_status(self) -> Dict[str, Dict]:
        """Get status of all AI providers"""
        status = {}

        for provider_id, provider_config in self.providers.items():
            status[provider_id] = {
                'name': provider_config.name,
                'configured': provider_id in self.api_keys,
                'client_available': provider_id in self.clients,
                'models': provider_config.models,
                'max_tokens': provider_config.max_tokens,
                'supports_streaming': provider_config.supports_streaming,
                'windows_optimized': provider_config.windows_optimized
            }

        return status

    def display_provider_status(self):
        """Display AI provider status"""
        status = self.get_provider_status()

        table = Table(title="🤖 AI Provider Status", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Status", style="white")
        table.add_column("Models", style="yellow")
        table.add_column("Max Tokens", style="green")
        table.add_column("Windows Optimized", style="blue")

        for provider_id, info in status.items():
            status_text = "✅ Ready" if info['client_available'] else ("🔑 Key Missing" if not info['configured'] else "❌ Error")
            models_count = len(info['models'])
            models_text = f"{models_count} available"
            max_tokens = f"{info['max_tokens']:,}"
            windows_opt = "✅" if info['windows_optimized'] else "❌"

            table.add_row(
                info['name'],
                status_text,
                models_text,
                max_tokens,
                windows_opt
            )

        self.console.print(table)

    async def analyze_code(self, code: str, language: str, analysis_type: str = "general") -> Optional[str]:
        """Analyze code using AI"""
        # Choose best available provider for code analysis
        preferred_providers = ['openai', 'anthropic', 'google']

        provider_id = None
        for pref in preferred_providers:
            if pref in self.clients:
                provider_id = pref
                break

        if not provider_id:
            self.console.print("[red]No suitable AI provider available for code analysis[/red]")
            return None

        provider_config = self.providers[provider_id]
        model = provider_config.models[0]  # Use first/best model

        # Create analysis prompt
        prompts = {
            "general": f"Analyze this {language} code and provide insights on quality, structure, and potential improvements:\n\n```{language}\n{code}\n```",
            "security": f"Analyze this {language} code for security vulnerabilities and potential threats:\n\n```{language}\n{code}\n```",
            "performance": f"Analyze this {language} code for performance issues and optimization opportunities:\n\n```{language}\n{code}\n```",
            "style": f"Analyze this {language} code for style issues and coding standards compliance:\n\n```{language}\n{code}\n```"
        }

        prompt = prompts.get(analysis_type, prompts["general"])

        response = await self.chat_with_ai(provider_id, model, prompt)

        if response:
            return response.content

        return None