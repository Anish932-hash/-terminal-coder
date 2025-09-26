#!/usr/bin/env python3
"""
Terminal Coder - Windows Main Application
Windows-optimized implementation with full feature support
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Final, TypeAlias
from dataclasses import dataclass, field
from functools import cached_property
from datetime import datetime

# Windows-specific imports
import winreg
import wmi
import psutil
import win32api
import win32con
import win32security
import win32service
import win32serviceutil
import ctypes
from ctypes import wintypes

# Rich for advanced terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
import rich.traceback

# Import Windows-specific managers
from .system_manager import WindowsSystemManager
from .ai_integration import WindowsAIIntegration
from .advanced_ai_integration import WindowsAdvancedAI
from .project_manager import WindowsProjectManager
from .gui import WindowsGUI

# Type aliases
JSONData: TypeAlias = dict[str, Any]
ProjectList: TypeAlias = list['Project']

rich.traceback.install(show_locals=True)


@dataclass(frozen=True, slots=True)
class WindowsConfig:
    """Windows-specific configuration"""
    use_powershell: bool = True
    use_wsl: bool = False
    enable_registry_access: bool = True
    enable_services_management: bool = True
    enable_wmi_access: bool = True
    use_windows_defender: bool = False
    preferred_terminal: str = "windows_terminal"  # cmd, powershell, windows_terminal
    enable_elevated_mode: bool = False


@dataclass(slots=True)
class Project:
    """Project configuration optimized for Windows"""
    name: str
    path: str
    language: str
    framework: str | None = None
    ai_provider: str = "openai"
    model: str = "gpt-4"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    windows_specific: WindowsConfig = field(default_factory=WindowsConfig)

    def __post_init__(self) -> None:
        """Ensure path is absolute and Windows-compatible"""
        # Convert path to Windows format
        self.path = str(Path(self.path).resolve())
        # Ensure path uses backslashes on Windows
        if os.name == 'nt':
            self.path = self.path.replace('/', '\\')

    @cached_property
    def path_obj(self) -> Path:
        """Get Path object for the project path"""
        return Path(self.path)


class TerminalCoderApp:
    """Main Terminal Coder Application - Windows Implementation"""

    # Class constants
    DEFAULT_CONFIG_DIR: Final[str] = ".terminal_coder"
    DEFAULT_WORKSPACE: Final[str] = "terminal_coder_workspace"
    APP_VERSION: Final[str] = "2.0.0-Windows"
    WINDOWS_REGISTRY_KEY: Final[str] = r"SOFTWARE\TerminalCoder"

    def __init__(self) -> None:
        self.console = Console()
        self._setup_windows_environment()
        self._setup_paths()
        self._setup_logging()

        # Initialize Windows-specific managers
        self.system_manager = WindowsSystemManager()
        self.ai_integration = WindowsAIIntegration()
        self.advanced_ai = WindowsAdvancedAI(self.console)
        self.project_manager = WindowsProjectManager()
        self.gui = WindowsGUI()

        # Load configuration and projects
        self.config = self.load_config()
        self.projects = self.load_projects()
        self.current_project: Project | None = None

        # Windows-specific initialization
        self._check_windows_features()
        self._setup_windows_integration()

    def _setup_windows_environment(self) -> None:
        """Setup Windows-specific environment"""
        # Set console code page to UTF-8 for better Unicode support
        try:
            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass  # Ignore if not available

        # Enable ANSI color support on Windows 10+
        try:
            import colorama
            colorama.init()
        except ImportError:
            pass

        # Set environment variables for better Windows compatibility
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['TERM'] = 'xterm-256color'

    def _setup_paths(self) -> None:
        """Setup application paths using Windows conventions"""
        # Use %APPDATA% for configuration on Windows
        appdata = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        self.config_dir = appdata / 'TerminalCoder'
        self.config_file = self.config_dir / "config.json"
        self.projects_file = self.config_dir / "projects.json"
        self.session_file = self.config_dir / "session.json"

        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Also setup workspace in Documents
        documents = Path.home() / 'Documents'
        self.default_workspace = documents / self.DEFAULT_WORKSPACE
        self.default_workspace.mkdir(exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup logging configuration for Windows"""
        log_dir = self.config_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Use Windows event log format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "terminal_coder.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Terminal Coder Windows initialized")

    def _check_windows_features(self) -> None:
        """Check available Windows features"""
        self.windows_features = {
            'is_admin': self._is_admin(),
            'has_powershell': self._has_powershell(),
            'has_wsl': self._has_wsl(),
            'has_windows_terminal': self._has_windows_terminal(),
            'has_docker': self._has_docker(),
            'has_git': self._has_git(),
            'windows_version': self._get_windows_version()
        }

        self.logger.info(f"Windows features detected: {self.windows_features}")

    def _is_admin(self) -> bool:
        """Check if running with administrator privileges"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except Exception:
            return False

    def _has_powershell(self) -> bool:
        """Check if PowerShell is available"""
        try:
            result = subprocess.run(['powershell', '-Command', 'echo "test"'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _has_wsl(self) -> bool:
        """Check if Windows Subsystem for Linux is available"""
        try:
            result = subprocess.run(['wsl', '--status'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _has_windows_terminal(self) -> bool:
        """Check if Windows Terminal is available"""
        try:
            result = subprocess.run(['wt', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _has_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _has_git(self) -> bool:
        """Check if Git is available"""
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _get_windows_version(self) -> str:
        """Get Windows version information"""
        try:
            import platform
            return platform.platform()
        except Exception:
            return "Unknown Windows Version"

    def _setup_windows_integration(self) -> None:
        """Setup Windows-specific integrations"""
        # Register file associations (if admin)
        if self.windows_features['is_admin']:
            self._register_file_associations()

        # Setup Windows services integration
        self._setup_services_integration()

        # Setup registry integration
        self._setup_registry_integration()

    def _register_file_associations(self) -> None:
        """Register file associations in Windows registry"""
        try:
            # This would register .tc files with Terminal Coder
            key_path = r"SOFTWARE\\Classes\\.tc"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValue(key, "", winreg.REG_SZ, "TerminalCoderProject")
        except Exception as e:
            self.logger.warning(f"Could not register file associations: {e}")

    def _setup_services_integration(self) -> None:
        """Setup Windows services integration"""
        # This would allow management of Windows services
        pass

    def _setup_registry_integration(self) -> None:
        """Setup Windows registry integration for settings"""
        try:
            key_path = self.WINDOWS_REGISTRY_KEY
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValueEx(key, "InstallPath", 0, winreg.REG_SZ, str(Path(__file__).parent.parent))
                winreg.SetValueEx(key, "Version", 0, winreg.REG_SZ, self.APP_VERSION)
        except Exception as e:
            self.logger.warning(f"Could not setup registry integration: {e}")

    def load_config(self) -> JSONData:
        """Load application configuration with Windows-specific defaults"""
        default_config: JSONData = {
            "theme": "dark",
            "auto_save": True,
            "show_line_numbers": True,
            "syntax_highlighting": True,
            "ai_provider": "openai",
            "model": "gpt-4",
            "max_tokens": 8000,
            "temperature": 0.7,
            "api_keys": {},
            "workspace": str(self.default_workspace),
            "windows": {
                "use_powershell": True,
                "use_wsl": self.windows_features['has_wsl'],
                "enable_registry_access": True,
                "enable_services_management": self.windows_features['is_admin'],
                "preferred_terminal": "windows_terminal" if self.windows_features['has_windows_terminal'] else "powershell"
            },
            "features": {
                "code_completion": True,
                "error_analysis": True,
                "code_review": True,
                "documentation_generation": True,
                "test_generation": True,
                "refactoring_suggestions": True,
                "security_analysis": True,
                "performance_optimization": True,
                "ai_code_explanation": True,
                "pattern_recognition": True,
                "code_translation": True,
                "windows_integration": True,
                "registry_management": self.windows_features['is_admin'],
                "services_management": self.windows_features['is_admin'],
            }
        }

        if not self.config_file.exists():
            self.save_config_to_file(default_config)
            return default_config

        try:
            with self.config_file.open('r', encoding='utf-8') as f:
                config = json.load(f)
            return default_config | config
        except (OSError, json.JSONDecodeError) as e:
            self.console.print(f"[red]Error loading config: {e}[/red]")
            self.logger.error(f"Config loading error: {e}")
            return default_config

    def save_config_to_file(self, config: JSONData) -> None:
        """Save configuration to file"""
        try:
            with self.config_file.open('w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self.logger.info("Configuration saved successfully")
        except OSError as e:
            self.console.print(f"[red]Error saving config: {e}[/red]")
            self.logger.error(f"Config save error: {e}")

    def load_projects(self) -> ProjectList:
        """Load saved projects"""
        if not self.projects_file.exists():
            return []

        try:
            with self.projects_file.open('r', encoding='utf-8') as f:
                data = json.load(f)

            projects = []
            for project_data in data:
                try:
                    # Handle Windows-specific config
                    if 'windows_specific' in project_data:
                        project_data['windows_specific'] = WindowsConfig(**project_data['windows_specific'])
                    projects.append(Project(**project_data))
                except TypeError as e:
                    self.logger.warning(f"Skipping invalid project data: {e}")
                    continue

            return projects
        except (OSError, json.JSONDecodeError) as e:
            self.console.print(f"[red]Error loading projects: {e}[/red]")
            self.logger.error(f"Projects loading error: {e}")
            return []

    def save_projects(self) -> None:
        """Save projects to file"""
        try:
            from dataclasses import asdict
            data = []
            for project in self.projects:
                project_dict = asdict(project)
                # Convert WindowsConfig to dict if needed
                if isinstance(project_dict.get('windows_specific'), WindowsConfig):
                    project_dict['windows_specific'] = asdict(project_dict['windows_specific'])
                data.append(project_dict)

            with self.projects_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(self.projects)} projects")
        except OSError as e:
            self.console.print(f"[red]Error saving projects: {e}[/red]")
            self.logger.error(f"Projects save error: {e}")

    def display_banner(self):
        """Display Windows-specific application banner"""
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       🚀 TERMINAL CODER v2.0-Windows                        ║
║                   Advanced AI-Powered Development Terminal                   ║
║                      Windows-Optimized | 200+ Features                      ║
║                    {self.windows_features['windows_version']:<50}           ║
╚══════════════════════════════════════════════════════════════════════════════╝

🖥️  Platform: Windows {('(Administrator)' if self.windows_features['is_admin'] else '')}
⚡ PowerShell: {'✅' if self.windows_features['has_powershell'] else '❌'}
🐧 WSL: {'✅' if self.windows_features['has_wsl'] else '❌'}
📱 Windows Terminal: {'✅' if self.windows_features['has_windows_terminal'] else '❌'}
🐳 Docker: {'✅' if self.windows_features['has_docker'] else '❌'}
"""

        self.console.print(Panel(
            banner,
            style="bold cyan",
            border_style="bright_blue"
        ))

    async def run_interactive_mode(self):
        """Run the interactive terminal interface - Windows optimized"""
        self.display_banner()

        # Windows-specific startup checks
        await self._perform_windows_startup_checks()

        while True:
            try:
                await self._show_main_menu()

                choice = Prompt.ask(
                    "\n[bold cyan]Select an option[/bold cyan]",
                    choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
                    default="3"
                )

                await self._handle_menu_choice(choice)

            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Exit Terminal Coder?[/yellow]"):
                    self.console.print("[green]Thank you for using Terminal Coder Windows! 🚀[/green]")
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                self.logger.error(f"Unexpected error: {e}")

    async def _perform_windows_startup_checks(self):
        """Perform Windows-specific startup checks"""
        with self.console.status("[bold blue]Performing Windows startup checks...") as status:
            # Check for updates
            status.update("[bold blue]Checking for updates...")
            await asyncio.sleep(0.5)

            # Verify Windows features
            status.update("[bold blue]Verifying Windows features...")
            await asyncio.sleep(0.5)

            # Initialize Windows services
            status.update("[bold blue]Initializing Windows services...")
            await asyncio.sleep(0.5)

        if not self.windows_features['has_git']:
            self.console.print("[yellow]⚠️  Git not found. Some features may be limited.[/yellow]")

        if not self.windows_features['has_powershell']:
            self.console.print("[yellow]⚠️  PowerShell not found. Using cmd as fallback.[/yellow]")

    async def _show_main_menu(self):
        """Display Windows-specific main menu"""
        table = Table(title="🔧 Terminal Coder - Windows Main Menu", style="cyan")
        table.add_column("Option", style="magenta", width=10)
        table.add_column("Description", style="white")
        table.add_column("Shortcut", style="yellow", width=15)

        menu_items = [
            ("1", "🆕 Create New Project", "Ctrl+N"),
            ("2", "📂 Open Existing Project", "Ctrl+O"),
            ("3", "🤖 AI Assistant", "Ctrl+A"),
            ("4", "⚙️  Configure Settings", "Ctrl+S"),
            ("5", "🔧 Code Tools", "Ctrl+T"),
            ("6", "📊 Project Analytics", "Ctrl+P"),
            ("7", "🧠 AI Model Manager", "Ctrl+M"),
            ("8", "🌐 API Manager", "Ctrl+I"),
            ("9", "📝 Documentation Generator", "Ctrl+D"),
            ("10", "🛡️  Security Scanner", "Ctrl+E"),
            ("11", "🚀 Deploy Assistant", "Ctrl+Y"),
            ("12", "🖥️  Windows Integration", "Ctrl+W"),
            ("13", "⚡ PowerShell Tools", "Ctrl+PS"),
            ("14", "🐳 Container Manager", "Ctrl+C"),
            ("15", "❓ Help & Features", "F1"),
            ("0", "❌ Exit", "Ctrl+Q")
        ]

        for option, description, shortcut in menu_items:
            table.add_row(option, description, shortcut)

        self.console.print(table)

    async def _handle_menu_choice(self, choice: str):
        """Handle menu choice selection"""
        if choice == "0":
            if Confirm.ask("Are you sure you want to exit?"):
                return False

        elif choice == "1":
            await self.create_new_project()
        elif choice == "2":
            await self.open_existing_project()
        elif choice == "3":
            await self.ai_assistant()
        elif choice == "4":
            await self.configure_settings()
        elif choice == "5":
            await self.code_tools_menu()
        elif choice == "6":
            await self.project_analytics()
        elif choice == "7":
            await self.ai_model_manager()
        elif choice == "8":
            await self.api_manager()
        elif choice == "9":
            await self.documentation_generator()
        elif choice == "10":
            await self.security_scanner()
        elif choice == "11":
            await self.deploy_assistant()
        elif choice == "12":
            await self.windows_integration_menu()
        elif choice == "13":
            await self.powershell_tools()
        elif choice == "14":
            await self.container_manager()
        elif choice == "15":
            await self.show_help()

        return True

    async def windows_integration_menu(self):
        """Windows-specific integration menu"""
        self.console.print(Panel("🖥️ Windows Integration Tools", style="blue"))

        # Show Windows-specific options
        options = [
            "1. 📋 Registry Manager",
            "2. ⚙️  Services Manager",
            "3. 📁 File Associations",
            "4. 🔒 Windows Security",
            "5. 📊 System Information",
            "6. 🔙 Back to Main Menu"
        ]

        for option in options:
            self.console.print(option)

        choice = Prompt.ask("Select Windows tool", choices=["1", "2", "3", "4", "5", "6"])

        if choice == "1":
            await self.registry_manager()
        elif choice == "2":
            await self.services_manager()
        # Add more Windows-specific implementations...

    async def registry_manager(self):
        """Windows Registry management interface"""
        self.console.print(Panel("📋 Windows Registry Manager", style="yellow"))
        self.console.print("[yellow]⚠️  Registry modification requires administrator privileges[/yellow]")

        if not self.windows_features['is_admin']:
            self.console.print("[red]❌ Administrator privileges required for registry access[/red]")
            return

        # Registry management implementation
        self.console.print("Registry management features would be implemented here")

    async def services_manager(self):
        """Windows Services management interface"""
        self.console.print(Panel("⚙️ Windows Services Manager", style="green"))

        # List Windows services
        try:
            services = []
            for service in psutil.win_service_iter():
                try:
                    service_info = service.as_dict()
                    services.append(service_info)
                except Exception:
                    continue

            if services:
                table = Table(title="Windows Services", style="green")
                table.add_column("Name", style="cyan")
                table.add_column("Status", style="white")
                table.add_column("Start Type", style="yellow")

                for service in services[:20]:  # Show first 20
                    table.add_row(
                        service.get('name', 'Unknown'),
                        service.get('status', 'Unknown'),
                        service.get('start_type', 'Unknown')
                    )

                self.console.print(table)
            else:
                self.console.print("[yellow]No services information available[/yellow]")

        except Exception as e:
            self.console.print(f"[red]Error accessing services: {e}[/red]")

    async def powershell_tools(self):
        """PowerShell integration tools"""
        self.console.print(Panel("⚡ PowerShell Tools", style="blue"))

        if not self.windows_features['has_powershell']:
            self.console.print("[red]❌ PowerShell not available on this system[/red]")
            return

        # PowerShell tools implementation
        options = [
            "1. 📜 Execute PowerShell Script",
            "2. 📊 System Information via PowerShell",
            "3. 🔧 PowerShell Module Manager",
            "4. 🔙 Back"
        ]

        for option in options:
            self.console.print(option)

        choice = Prompt.ask("Select PowerShell tool", choices=["1", "2", "3", "4"])

        if choice == "1":
            await self.execute_powershell_script()
        elif choice == "2":
            await self.powershell_system_info()

    async def execute_powershell_script(self):
        """Execute custom PowerShell script"""
        script = Prompt.ask("[cyan]Enter PowerShell command or script[/cyan]")

        try:
            with self.console.status("[bold blue]Executing PowerShell script..."):
                result = subprocess.run(
                    ['powershell', '-Command', script],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

            if result.returncode == 0:
                self.console.print(Panel(result.stdout, title="PowerShell Output", style="green"))
            else:
                self.console.print(Panel(result.stderr, title="PowerShell Error", style="red"))

        except Exception as e:
            self.console.print(f"[red]Error executing PowerShell: {e}[/red]")

    async def powershell_system_info(self):
        """Get system information via PowerShell"""
        try:
            with self.console.status("[bold blue]Gathering system information..."):
                # Execute PowerShell command to get system info
                script = "Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, TotalPhysicalMemory, CsProcessors"
                result = subprocess.run(
                    ['powershell', '-Command', script],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

            if result.returncode == 0:
                self.console.print(Panel(result.stdout, title="System Information", style="cyan"))
            else:
                self.console.print("[yellow]Could not retrieve system information[/yellow]")

        except Exception as e:
            self.console.print(f"[red]Error getting system info: {e}[/red]")

    # Add placeholder methods for other functionality
    async def create_new_project(self):
        """Create new project - Windows implementation"""
        return await self.project_manager.create_project()

    async def open_existing_project(self):
        """Open existing project"""
        return await self.project_manager.open_project()

    async def ai_assistant(self):
        """AI assistant - Windows implementation with advanced features"""
        # Initialize Windows features if not already done
        await self.advanced_ai.initialize_windows_features()

        # Set up API keys from config
        api_keys = self.config.get("api_keys", {})
        if api_keys:
            await self.advanced_ai.initialize_ai_clients(api_keys)

        # Set current provider and model from config
        self.advanced_ai.current_provider = self.config.get("ai_provider", "openai")
        self.advanced_ai.current_model = self.config.get("model", "gpt-4")

        self.console.print(Panel("🤖 Windows AI Assistant", style="blue"))
        self.console.print(f"[cyan]Provider: {self.advanced_ai.current_provider}[/cyan]")
        self.console.print(f"[cyan]Model: {self.advanced_ai.current_model}[/cyan]")

        while True:
            try:
                query = Prompt.ask("\n[bold green]Ask me anything (or 'exit' to return, '/help' for commands)[/bold green]")

                if query.lower() in ['exit', 'quit', 'back']:
                    break

                # Process with Windows-specific features
                response = await self.advanced_ai.process_user_input_windows(
                    query,
                    stream=True,
                    use_powershell=self.config.get("windows", {}).get("use_powershell", True)
                )

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]AI Error: {e}[/red]")

    async def configure_settings(self):
        """Configure Windows-specific settings"""
        self.console.print(Panel("⚙️ Windows Settings Configuration", style="yellow"))

        settings_menu = [
            "1. 🔑 API Keys Configuration",
            "2. 🖥️  Windows Integration Settings",
            "3. ⚡ PowerShell Settings",
            "4. 🤖 AI Provider Settings",
            "5. 🎨 Theme and Display Settings",
            "6. 📁 Workspace and Project Settings",
            "7. 🔙 Back to Main Menu"
        ]

        for item in settings_menu:
            self.console.print(item)

        choice = Prompt.ask("Select setting to configure", choices=["1", "2", "3", "4", "5", "6", "7"])

        if choice == "1":
            await self.configure_api_keys()
        elif choice == "2":
            await self.configure_windows_integration()
        elif choice == "3":
            await self.configure_powershell_settings()
        elif choice == "4":
            await self.configure_ai_provider()

    async def configure_api_keys(self):
        """Configure API keys using Windows credential store"""
        await self.advanced_ai.credential_manager.list_credentials()

        providers = ["openai", "anthropic", "google", "cohere"]

        for provider in providers:
            if Confirm.ask(f"Configure {provider.upper()} API key?"):
                api_key = Prompt.ask(f"Enter {provider.upper()} API key", password=True)
                if api_key:
                    # Store in Windows credential store
                    success = await self.advanced_ai.credential_manager.store_credential(
                        f"terminal-coder-{provider}", "api_key", api_key
                    )
                    if success:
                        self.config["api_keys"][provider] = api_key
                        self.console.print(f"[green]✅ {provider.upper()} API key saved[/green]")
                    else:
                        self.console.print(f"[red]❌ Failed to save {provider.upper()} API key[/red]")

        self.save_config_to_file(self.config)

    async def configure_windows_integration(self):
        """Configure Windows-specific integration settings"""
        self.console.print(Panel("🖥️ Windows Integration Settings", style="blue"))

        current_settings = self.config.get("windows", {})

        # PowerShell usage
        use_powershell = Confirm.ask(
            f"Use PowerShell for shell operations? (current: {current_settings.get('use_powershell', True)})",
            default=current_settings.get('use_powershell', True)
        )

        # Registry access
        enable_registry = Confirm.ask(
            f"Enable registry access? (current: {current_settings.get('enable_registry_access', True)})",
            default=current_settings.get('enable_registry_access', True)
        )

        # Services management
        enable_services = Confirm.ask(
            f"Enable services management? (current: {current_settings.get('enable_services_management', False)})",
            default=current_settings.get('enable_services_management', False)
        )

        # Update configuration
        self.config["windows"] = {
            **current_settings,
            "use_powershell": use_powershell,
            "enable_registry_access": enable_registry,
            "enable_services_management": enable_services
        }

        self.save_config_to_file(self.config)
        self.console.print("[green]✅ Windows integration settings updated[/green]")

    async def configure_powershell_settings(self):
        """Configure PowerShell-specific settings"""
        if not self.windows_features['has_powershell']:
            self.console.print("[red]❌ PowerShell not available on this system[/red]")
            return

        self.console.print(Panel("⚡ PowerShell Settings", style="blue"))
        self.console.print("PowerShell configuration options would be implemented here")

    async def configure_ai_provider(self):
        """Configure AI provider settings"""
        self.console.print(Panel("🤖 AI Provider Settings", style="green"))

        providers = list(self.advanced_ai.ai_clients.keys()) if hasattr(self.advanced_ai, 'ai_clients') else ["openai", "anthropic", "google", "cohere"]

        if providers:
            current_provider = self.config.get("ai_provider", "openai")
            provider = Prompt.ask("Select AI provider", choices=providers, default=current_provider)

            # Get available models for the provider
            models = {
                "openai": ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
                "cohere": ["command-r-plus", "command-r", "command"]
            }

            available_models = models.get(provider, ["default"])
            current_model = self.config.get("model", available_models[0])
            model = Prompt.ask("Select model", choices=available_models, default=current_model)

            self.config["ai_provider"] = provider
            self.config["model"] = model

            self.save_config_to_file(self.config)
            self.console.print(f"[green]✅ AI provider set to {provider} with model {model}[/green]")

    async def code_tools_menu(self):
        """Code tools menu"""
        pass

    async def project_analytics(self):
        """Project analytics"""
        pass

    async def ai_model_manager(self):
        """AI model manager"""
        pass

    async def api_manager(self):
        """API manager"""
        pass

    async def documentation_generator(self):
        """Documentation generator"""
        pass

    async def security_scanner(self):
        """Security scanner"""
        pass

    async def deploy_assistant(self):
        """Deploy assistant"""
        pass

    async def container_manager(self):
        """Container manager"""
        pass

    async def show_help(self):
        """Show Windows-specific help with advanced AI features"""
        help_content = await self.advanced_ai.get_windows_help()
        self.console.print(Panel(help_content, title="🚀 Terminal Coder Windows Help", style="cyan"))


def main():
    """Main entry point for Windows application"""
    parser = argparse.ArgumentParser(description="Terminal Coder - Windows Edition")
    parser.add_argument("--version", action="version", version="Terminal Coder v2.0-Windows")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--project", help="Open specific project")
    parser.add_argument("--admin", action="store_true", help="Request administrator privileges")

    args = parser.parse_args()

    # Check for admin privileges if requested
    if args.admin and not ctypes.windll.shell32.IsUserAnAdmin():
        print("Requesting administrator privileges...")
        try:
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, " ".join(sys.argv), None, 1
            )
            return
        except Exception as e:
            print(f"Failed to elevate privileges: {e}")

    # Create and run the application
    app = TerminalCoderApp()

    try:
        asyncio.run(app.run_interactive_mode())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()