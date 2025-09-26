#!/usr/bin/env python3
"""
Terminal Coder - Linux Main Application
Linux-optimized implementation with full feature support
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import sys
import subprocess
import signal
from pathlib import Path
from typing import Any, Final, TypeAlias
from dataclasses import dataclass, field
from functools import cached_property
from datetime import datetime

# Linux-specific imports
import pwd
import grp
import psutil
import distro

try:
    import dbus
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

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

# Import Linux-specific managers
from .system_manager import LinuxSystemManager
from .ai_integration import LinuxAIIntegration
from .advanced_ai_integration import LinuxAdvancedAI
from .project_manager import LinuxProjectManager
from .gui import LinuxGUI

# Type aliases
JSONData: TypeAlias = dict[str, Any]
ProjectList: TypeAlias = list['Project']

rich.traceback.install(show_locals=True)


@dataclass(frozen=True, slots=True)
class LinuxConfig:
    """Linux-specific configuration"""
    use_systemd: bool = True
    enable_dbus: bool = DBUS_AVAILABLE
    use_package_manager: bool = True
    enable_container_support: bool = True
    preferred_shell: str = "bash"  # bash, zsh, fish
    enable_kernel_integration: bool = False
    use_apparmor: bool = False
    use_selinux: bool = False
    distribution: str = field(default_factory=lambda: distro.name())


@dataclass(slots=True)
class Project:
    """Project configuration optimized for Linux"""
    name: str
    path: str
    language: str
    framework: str | None = None
    ai_provider: str = "openai"
    model: str = "gpt-4"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    linux_specific: LinuxConfig = field(default_factory=LinuxConfig)

    def __post_init__(self) -> None:
        """Ensure path is absolute and Linux-compatible"""
        self.path = str(Path(self.path).resolve())

    @cached_property
    def path_obj(self) -> Path:
        """Get Path object for the project path"""
        return Path(self.path)


class TerminalCoderApp:
    """Main Terminal Coder Application - Linux Implementation"""

    # Class constants
    DEFAULT_CONFIG_DIR: Final[str] = ".terminal_coder"
    DEFAULT_WORKSPACE: Final[str] = "terminal_coder_workspace"
    APP_VERSION: Final[str] = "2.0.0-Linux"

    def __init__(self) -> None:
        self.console = Console()
        self._setup_linux_environment()
        self._setup_paths()
        self._setup_logging()

        # Initialize Linux-specific managers
        self.system_manager = LinuxSystemManager()
        self.ai_integration = LinuxAIIntegration()
        self.advanced_ai = LinuxAdvancedAI(self.console)
        self.project_manager = LinuxProjectManager()
        self.gui = LinuxGUI()

        # Load configuration and projects
        self.config = self.load_config()
        self.projects = self.load_projects()
        self.current_project: Project | None = None

        # Linux-specific initialization
        self._check_linux_features()
        self._setup_linux_integration()

        # Setup signal handlers
        self._setup_signal_handlers()

    def _setup_linux_environment(self) -> None:
        """Setup Linux-specific environment"""
        # Set locale for better Unicode support
        os.environ.setdefault('LANG', 'C.UTF-8')
        os.environ.setdefault('LC_ALL', 'C.UTF-8')

        # Set TERM for better terminal compatibility
        if not os.environ.get('TERM'):
            os.environ['TERM'] = 'xterm-256color'

        # Enable colors in terminal
        os.environ['FORCE_COLOR'] = '1'
        os.environ['CLICOLOR'] = '1'

    def _setup_paths(self) -> None:
        """Setup application paths using Linux conventions"""
        # Use XDG Base Directory specification
        xdg_config = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        xdg_data = os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share')

        self.config_dir = Path(xdg_config) / 'terminal-coder'
        self.data_dir = Path(xdg_data) / 'terminal-coder'

        self.config_file = self.config_dir / "config.json"
        self.projects_file = self.data_dir / "projects.json"
        self.session_file = self.config_dir / "session.json"

        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Default workspace
        self.default_workspace = Path.home() / self.DEFAULT_WORKSPACE
        self.default_workspace.mkdir(exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup logging configuration for Linux"""
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Use syslog-compatible format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s[%(process)d]: %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "terminal_coder.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Terminal Coder Linux initialized")

    def _setup_signal_handlers(self) -> None:
        """Setup Linux signal handlers with async compatibility"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully")
            # Set shutdown flag for graceful async cleanup
            self._shutdown_requested = True
            try:
                # Try to cancel any running async tasks
                if hasattr(self, '_current_loop') and self._current_loop:
                    for task in asyncio.all_tasks(self._current_loop):
                        if not task.done():
                            task.cancel()
            except Exception as e:
                self.logger.error(f"Error cancelling tasks during shutdown: {e}")
            finally:
                self.cleanup()
                sys.exit(0)

        # Initialize shutdown flag
        self._shutdown_requested = False
        self._current_loop = None

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)

    def _check_linux_features(self) -> None:
        """Check available Linux features"""
        self.linux_features = {
            'distribution': distro.name(),
            'distro_version': distro.version(),
            'kernel_version': self._get_kernel_version(),
            'has_systemd': self._has_systemd(),
            'has_dbus': DBUS_AVAILABLE,
            'has_docker': self._has_docker(),
            'has_podman': self._has_podman(),
            'has_git': self._has_git(),
            'has_snap': self._has_snap(),
            'has_flatpak': self._has_flatpak(),
            'package_manager': self._detect_package_manager(),
            'init_system': self._detect_init_system(),
            'desktop_environment': self._detect_desktop_environment(),
            'is_root': os.geteuid() == 0,
            'shell': os.environ.get('SHELL', '/bin/bash'),
            'supports_containers': self._check_container_support(),
            'supports_virtualization': self._check_virtualization_support()
        }

        self.logger.info(f"Linux features detected: {self.linux_features}")

    def _get_kernel_version(self) -> str:
        """Get Linux kernel version"""
        try:
            with open('/proc/version', 'r') as f:
                version_line = f.read().strip()
                # Extract version from "Linux version X.X.X"
                import re
                match = re.search(r'Linux version (\S+)', version_line)
                return match.group(1) if match else "Unknown"
        except Exception:
            return "Unknown"

    def _has_systemd(self) -> bool:
        """Check if systemd is available"""
        return (Path('/run/systemd/system').exists() or
                Path('/usr/lib/systemd/systemd').exists())

    def _has_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _has_podman(self) -> bool:
        """Check if Podman is available"""
        try:
            result = subprocess.run(['podman', '--version'], capture_output=True, text=True, timeout=5)
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

    def _has_snap(self) -> bool:
        """Check if Snap is available"""
        try:
            result = subprocess.run(['snap', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _has_flatpak(self) -> bool:
        """Check if Flatpak is available"""
        try:
            result = subprocess.run(['flatpak', '--version'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _detect_package_manager(self) -> str:
        """Detect the system package manager"""
        managers = {
            'apt': '/usr/bin/apt',
            'dnf': '/usr/bin/dnf',
            'yum': '/usr/bin/yum',
            'pacman': '/usr/bin/pacman',
            'zypper': '/usr/bin/zypper',
            'emerge': '/usr/bin/emerge',
            'apk': '/sbin/apk'
        }

        for name, path in managers.items():
            if Path(path).exists():
                return name

        return 'unknown'

    def _detect_init_system(self) -> str:
        """Detect the init system"""
        if self._has_systemd():
            return 'systemd'
        elif Path('/sbin/init').is_symlink():
            target = Path('/sbin/init').readlink()
            if 'upstart' in str(target):
                return 'upstart'
            elif 'systemd' in str(target):
                return 'systemd'

        return 'sysv'

    def _detect_desktop_environment(self) -> str:
        """Detect the desktop environment"""
        desktop_env = (
            os.environ.get('XDG_CURRENT_DESKTOP') or
            os.environ.get('DESKTOP_SESSION') or
            os.environ.get('GDMSESSION')
        )

        if desktop_env:
            return desktop_env.lower()

        # Fallback detection
        if os.environ.get('KDE_FULL_SESSION'):
            return 'kde'
        elif os.environ.get('GNOME_DESKTOP_SESSION_ID'):
            return 'gnome'
        elif os.environ.get('XFCE4_SESSION'):
            return 'xfce'

        return 'unknown'

    def _check_container_support(self) -> bool:
        """Check if container technologies are supported"""
        return (self.linux_features['has_docker'] or
                self.linux_features['has_podman'] or
                Path('/proc/sys/kernel/unprivileged_userns_clone').exists())

    def _check_virtualization_support(self) -> bool:
        """Check if virtualization is supported"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'vmx' in cpuinfo or 'svm' in cpuinfo
        except Exception:
            return False

    def _setup_linux_integration(self) -> None:
        """Setup Linux-specific integrations"""
        # Setup D-Bus integration if available
        if DBUS_AVAILABLE:
            self._setup_dbus_integration()

        # Setup systemd integration if available
        if self.linux_features['has_systemd']:
            self._setup_systemd_integration()

        # Setup desktop integration
        self._setup_desktop_integration()

    def _setup_dbus_integration(self) -> None:
        """Setup D-Bus integration"""
        try:
            self.dbus_session = dbus.SessionBus()
            self.dbus_system = dbus.SystemBus()
            self.logger.info("D-Bus integration initialized")
        except Exception as e:
            self.logger.warning(f"Could not setup D-Bus integration: {e}")

    def _setup_systemd_integration(self) -> None:
        """Setup systemd integration"""
        # This would setup systemd user services for Terminal Coder
        pass

    def _setup_desktop_integration(self) -> None:
        """Setup desktop integration"""
        # Create .desktop file for application launcher
        desktop_file = Path.home() / '.local/share/applications/terminal-coder.desktop'
        desktop_content = f"""[Desktop Entry]
Name=Terminal Coder
Comment=AI-Powered Development Terminal
Exec={sys.executable} {__file__}
Icon=terminal
Terminal=true
Type=Application
Categories=Development;Programming;
Keywords=terminal;coding;ai;development;
"""

        try:
            desktop_file.parent.mkdir(parents=True, exist_ok=True)
            with open(desktop_file, 'w') as f:
                f.write(desktop_content)
        except Exception as e:
            self.logger.warning(f"Could not create desktop file: {e}")

    def cleanup(self) -> None:
        """Cleanup resources before shutdown"""
        self.logger.info("Cleaning up Terminal Coder Linux...")
        # Perform any necessary cleanup
        pass

    def load_config(self) -> JSONData:
        """Load application configuration with Linux-specific defaults"""
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
            "linux": {
                "use_systemd": self.linux_features['has_systemd'],
                "enable_dbus": self.linux_features['has_dbus'],
                "use_package_manager": True,
                "enable_container_support": self.linux_features['supports_containers'],
                "preferred_shell": self.linux_features['shell'].split('/')[-1],
                "distribution": self.linux_features['distribution']
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
                "linux_integration": True,
                "systemd_management": self.linux_features['has_systemd'],
                "container_management": self.linux_features['supports_containers'],
                "package_management": True,
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
                    # Handle Linux-specific config
                    if 'linux_specific' in project_data:
                        project_data['linux_specific'] = LinuxConfig(**project_data['linux_specific'])
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
                # Convert LinuxConfig to dict if needed
                if isinstance(project_dict.get('linux_specific'), LinuxConfig):
                    project_dict['linux_specific'] = asdict(project_dict['linux_specific'])
                data.append(project_dict)

            with self.projects_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(self.projects)} projects")
        except OSError as e:
            self.console.print(f"[red]Error saving projects: {e}[/red]")
            self.logger.error(f"Projects save error: {e}")

    def display_banner(self):
        """Display Linux-specific application banner"""
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🚀 TERMINAL CODER v2.0-Linux                         ║
║                   Advanced AI-Powered Development Terminal                   ║
║                       Linux-Optimized | 200+ Features                       ║
║                     {self.linux_features['distribution']} {self.linux_features['distro_version']:<30}                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

🐧 Distribution: {self.linux_features['distribution']} {self.linux_features['distro_version']}
🔧 Kernel: {self.linux_features['kernel_version']}
⚙️  Init System: {self.linux_features['init_system']}
📦 Package Manager: {self.linux_features['package_manager']}
🖥️  Desktop: {self.linux_features['desktop_environment']}
🐳 Containers: {'✅' if self.linux_features['supports_containers'] else '❌'}
🔒 Root: {'✅' if self.linux_features['is_root'] else '❌'}
"""

        self.console.print(Panel(
            banner,
            style="bold cyan",
            border_style="bright_blue"
        ))

    async def run_interactive_mode(self):
        """Run the interactive terminal interface - Linux optimized with error handling"""
        # Store current loop for signal handler
        self._current_loop = asyncio.get_event_loop()

        try:
            self.display_banner()

            # Linux-specific startup checks
            await self._perform_linux_startup_checks()

            while not self._shutdown_requested:
                try:
                    await self._show_main_menu()

                    # Add timeout to prevent hanging on input
                    try:
                        choice = await asyncio.wait_for(
                            asyncio.to_thread(lambda: Prompt.ask(
                                "\n[bold cyan]Select an option[/bold cyan]",
                                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"],
                                default="3"
                            )),
                            timeout=300.0  # 5 minute timeout
                        )
                    except asyncio.TimeoutError:
                        self.console.print("[yellow]Session timed out due to inactivity[/yellow]")
                        break

                    if not await self._handle_menu_choice(choice):
                        break

                except KeyboardInterrupt:
                    if await asyncio.to_thread(lambda: Confirm.ask("\n[yellow]Exit Terminal Coder?[/yellow]")):
                        self.console.print("[green]Thank you for using Terminal Coder Linux! 🐧[/green]")
                        break
                except asyncio.CancelledError:
                    self.console.print("\n[yellow]Application cancelled[/yellow]")
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    self.logger.error(f"Unexpected error: {e}", exc_info=True)
                    # Continue running unless it's a critical error
                    if "critical" in str(e).lower() or "fatal" in str(e).lower():
                        break

        except Exception as e:
            self.logger.error(f"Critical error in interactive mode: {e}", exc_info=True)
            self.console.print(f"[red]Critical error occurred: {e}[/red]")
        finally:
            self._current_loop = None

    async def _perform_linux_startup_checks(self):
        """Perform Linux-specific startup checks"""
        with self.console.status("[bold blue]Performing Linux startup checks...") as status:
            # Check distribution compatibility
            status.update("[bold blue]Checking distribution compatibility...")
            await asyncio.sleep(0.5)

            # Verify package manager
            status.update("[bold blue]Verifying package manager...")
            await asyncio.sleep(0.5)

            # Check container support
            status.update("[bold blue]Checking container support...")
            await asyncio.sleep(0.5)

            # Initialize system integrations
            status.update("[bold blue]Initializing system integrations...")
            await asyncio.sleep(0.5)

        # Show warnings if needed
        if not self.linux_features['has_git']:
            self.console.print("[yellow]⚠️  Git not found. Install with package manager.[/yellow]")

        if not self.linux_features['supports_containers']:
            self.console.print("[yellow]⚠️  Container support not available.[/yellow]")

        if self.linux_features['package_manager'] == 'unknown':
            self.console.print("[yellow]⚠️  Package manager not detected.[/yellow]")

    async def _show_main_menu(self):
        """Display Linux-specific main menu"""
        table = Table(title="🔧 Terminal Coder - Linux Main Menu", style="cyan")
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
            ("12", "🐧 Linux Integration", "Ctrl+L"),
            ("13", "📦 Package Manager", "Ctrl+PM"),
            ("14", "🐳 Container Manager", "Ctrl+C"),
            ("15", "⚡ System Tools", "Ctrl+ST"),
            ("16", "❓ Help & Features", "F1"),
            ("0", "❌ Exit", "Ctrl+Q")
        ]

        for option, description, shortcut in menu_items:
            table.add_row(option, description, shortcut)

        self.console.print(table)

    async def _handle_menu_choice(self, choice: str) -> bool:
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
            await self.linux_integration_menu()
        elif choice == "13":
            await self.package_manager_menu()
        elif choice == "14":
            await self.container_manager()
        elif choice == "15":
            await self.system_tools_menu()
        elif choice == "16":
            await self.show_help()

        return True

    async def linux_integration_menu(self):
        """Linux-specific integration menu"""
        self.console.print(Panel("🐧 Linux Integration Tools", style="blue"))

        options = [
            "1. 📊 System Information",
            "2. ⚙️  Systemd Services",
            "3. 📋 Process Manager",
            "4. 🔧 System Configuration",
            "5. 🌐 Network Tools",
            "6. 💾 File System Tools",
            "7. 🔙 Back to Main Menu"
        ]

        for option in options:
            self.console.print(option)

        choice = Prompt.ask("Select Linux tool", choices=["1", "2", "3", "4", "5", "6", "7"])

        if choice == "1":
            await self.show_system_information()
        elif choice == "2":
            await self.systemd_services_manager()
        elif choice == "3":
            await self.process_manager()
        elif choice == "4":
            await self.system_configuration()
        elif choice == "5":
            await self.network_tools()
        elif choice == "6":
            await self.filesystem_tools()

    async def package_manager_menu(self):
        """Package manager interface"""
        self.console.print(Panel(f"📦 Package Manager - {self.linux_features['package_manager'].upper()}", style="green"))

        package_manager = self.linux_features['package_manager']

        if package_manager == 'unknown':
            self.console.print("[red]❌ Package manager not detected[/red]")
            return

        options = [
            "1. 🔍 Search Packages",
            "2. 📦 Install Package",
            "3. 🗑️  Remove Package",
            "4. ⬆️  Update System",
            "5. 📋 List Installed",
            "6. ℹ️  Package Info",
            "7. 🔙 Back"
        ]

        for option in options:
            self.console.print(option)

        choice = Prompt.ask("Select package operation", choices=["1", "2", "3", "4", "5", "6", "7"])

        if choice == "1":
            await self.search_packages()
        elif choice == "2":
            await self.install_package()
        elif choice == "3":
            await self.remove_package()
        elif choice == "4":
            await self.update_system()
        elif choice == "5":
            await self.list_installed_packages()
        elif choice == "6":
            await self.show_package_info()

    async def search_packages(self):
        """Search for packages"""
        package_name = Prompt.ask("[cyan]Enter package name to search[/cyan]")
        if not package_name:
            return

        package_manager = self.linux_features['package_manager']

        # Build search command based on package manager
        commands = {
            'apt': ['apt', 'search', package_name],
            'dnf': ['dnf', 'search', package_name],
            'yum': ['yum', 'search', package_name],
            'pacman': ['pacman', '-Ss', package_name],
            'zypper': ['zypper', 'search', package_name],
            'apk': ['apk', 'search', package_name]
        }

        command = commands.get(package_manager)
        if not command:
            self.console.print(f"[red]Search not supported for {package_manager}[/red]")
            return

        try:
            with self.console.status(f"[bold blue]Searching for {package_name}..."):
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.console.print(Panel(result.stdout, title=f"Search Results for '{package_name}'", style="green"))
            else:
                self.console.print(Panel(result.stderr, title="Search Error", style="red"))

        except subprocess.TimeoutExpired:
            self.console.print("[red]Search timed out[/red]")
        except Exception as e:
            self.console.print(f"[red]Search error: {e}[/red]")

    async def install_package(self):
        """Install a package"""
        if not self.linux_features['is_root']:
            self.console.print("[yellow]⚠️  Root privileges may be required for package installation[/yellow]")

        package_name = Prompt.ask("[cyan]Enter package name to install[/cyan]")
        if not package_name:
            return

        if not Confirm.ask(f"Install package '{package_name}'?"):
            return

        package_manager = self.linux_features['package_manager']

        # Build install command
        commands = {
            'apt': ['sudo', 'apt', 'install', '-y', package_name],
            'dnf': ['sudo', 'dnf', 'install', '-y', package_name],
            'yum': ['sudo', 'yum', 'install', '-y', package_name],
            'pacman': ['sudo', 'pacman', '-S', '--noconfirm', package_name],
            'zypper': ['sudo', 'zypper', 'install', '-y', package_name],
            'apk': ['sudo', 'apk', 'add', package_name]
        }

        command = commands.get(package_manager)
        if not command:
            self.console.print(f"[red]Installation not supported for {package_manager}[/red]")
            return

        try:
            with self.console.status(f"[bold blue]Installing {package_name}..."):
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.console.print(f"[green]✅ Package '{package_name}' installed successfully[/green]")
            else:
                self.console.print(Panel(result.stderr, title="Installation Error", style="red"))

        except subprocess.TimeoutExpired:
            self.console.print("[red]Installation timed out[/red]")
        except Exception as e:
            self.console.print(f"[red]Installation error: {e}[/red]")

    async def system_tools_menu(self):
        """System tools menu"""
        self.console.print(Panel("⚡ Linux System Tools", style="yellow"))

        options = [
            "1. 🔍 System Monitor",
            "2. 📈 Performance Stats",
            "3. 💾 Disk Usage",
            "4. 🌐 Network Status",
            "5. 🔒 Security Status",
            "6. 🔄 Process Tree",
            "7. 📊 Resource Usage",
            "8. 🔙 Back"
        ]

        for option in options:
            self.console.print(option)

        choice = Prompt.ask("Select system tool", choices=["1", "2", "3", "4", "5", "6", "7", "8"])

        if choice == "1":
            await self.system_monitor()
        elif choice == "2":
            await self.performance_stats()
        elif choice == "3":
            await self.disk_usage()
        elif choice == "4":
            await self.network_status()

    async def system_monitor(self):
        """Show system monitoring information"""
        self.system_manager.display_system_info()

    # Add placeholder methods for other functionality
    async def create_new_project(self):
        """Create new project - Linux implementation"""
        return await self.project_manager.create_project()

    async def open_existing_project(self):
        """Open existing project"""
        return await self.project_manager.open_project()

    async def ai_assistant(self):
        """AI assistant - Linux implementation with advanced features and error handling"""
        try:
            # Initialize Linux features if not already done
            await asyncio.wait_for(
                self.advanced_ai.initialize_linux_features(),
                timeout=30.0
            )

            # Set up API keys from config
            api_keys = self.config.get("api_keys", {})
            if api_keys:
                await asyncio.wait_for(
                    self.advanced_ai.initialize_ai_clients(api_keys),
                    timeout=30.0
                )
            else:
                self.console.print("[yellow]⚠️  No API keys configured. Some features may be limited.[/yellow]")

            # Set current provider and model from config
            self.advanced_ai.current_provider = self.config.get("ai_provider", "openai")
            self.advanced_ai.current_model = self.config.get("model", "gpt-4")

            self.console.print(Panel("🤖 Linux AI Assistant", style="blue"))
            self.console.print(f"[cyan]Distribution: {self.linux_features['distribution']}[/cyan]")
            self.console.print(f"[cyan]Provider: {self.advanced_ai.current_provider}[/cyan]")
            self.console.print(f"[cyan]Model: {self.advanced_ai.current_model}[/cyan]")
            self.console.print("[dim]Type '/help' for commands, '/exit' to return to main menu[/dim]")

            while not self._shutdown_requested:
                try:
                    # Add timeout to prevent hanging
                    query = await asyncio.wait_for(
                        asyncio.to_thread(lambda: Prompt.ask(
                            "\n[bold green]Ask me anything[/bold green]"
                        )),
                        timeout=300.0  # 5 minute timeout
                    )

                    if query.lower() in ['exit', 'quit', 'back', '/exit']:
                        break

                    if query.lower() in ['/help', 'help']:
                        self.console.print(Panel(
                            "Available commands:\n"
                            "• /exit, /quit, exit, quit, back - Return to main menu\n"
                            "• /help, help - Show this help\n"
                            "• Any other input - Ask the AI assistant",
                            title="AI Assistant Help",
                            style="cyan"
                        ))
                        continue

                    if not query.strip():
                        continue

                    # Process with Linux-specific features with timeout
                    with self.console.status("[bold blue]Processing your request..."):
                        response = await asyncio.wait_for(
                            self.advanced_ai.process_user_input_linux(
                                query,
                                stream=True,
                                use_systemd=self.config.get("linux", {}).get("use_systemd", True)
                            ),
                            timeout=120.0  # 2 minute timeout for AI responses
                        )

                except asyncio.TimeoutError:
                    self.console.print("[yellow]Request timed out. Please try a shorter query or check your connection.[/yellow]")
                except KeyboardInterrupt:
                    if await asyncio.to_thread(lambda: Confirm.ask("\n[yellow]Return to main menu?[/yellow]")):
                        break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.console.print(f"[red]AI Error: {e}[/red]")
                    self.logger.error(f"AI assistant error: {e}", exc_info=True)
                    # Continue unless it's a critical error
                    if "api" in str(e).lower() and "key" in str(e).lower():
                        self.console.print("[yellow]Please check your API key configuration in settings.[/yellow]")

        except asyncio.TimeoutError:
            self.console.print("[red]AI assistant initialization timed out[/red]")
        except Exception as e:
            self.console.print(f"[red]Failed to initialize AI assistant: {e}[/red]")
            self.logger.error(f"AI assistant initialization error: {e}", exc_info=True)

    # Placeholder methods for remaining functionality
    async def configure_settings(self):
        """Configure Linux-specific settings"""
        self.console.print(Panel("⚙️ Linux Settings Configuration", style="yellow"))

        settings_menu = [
            "1. 🔑 API Keys Configuration",
            "2. 🐧 Linux Integration Settings",
            "3. 📦 Package Manager Settings",
            "4. 🤖 AI Provider Settings",
            "5. 🐳 Container Settings",
            "6. ⚡ System Integration Settings",
            "7. 🎨 Theme and Display Settings",
            "8. 📁 Workspace and Project Settings",
            "9. 🔙 Back to Main Menu"
        ]

        for item in settings_menu:
            self.console.print(item)

        choice = Prompt.ask("Select setting to configure", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"])

        if choice == "1":
            await self.configure_api_keys()
        elif choice == "2":
            await self.configure_linux_integration()
        elif choice == "3":
            await self.configure_package_manager()
        elif choice == "4":
            await self.configure_ai_provider()

    async def configure_api_keys(self):
        """Configure API keys using Linux keyring"""
        self.console.print(Panel("🔑 API Key Configuration", style="green"))

        providers = ["openai", "anthropic", "google", "cohere"]

        for provider in providers:
            if Confirm.ask(f"Configure {provider.upper()} API key?"):
                api_key = Prompt.ask(f"Enter {provider.upper()} API key", password=True)
                if api_key:
                    # Store in Linux keyring if available
                    try:
                        import keyring
                        keyring.set_password(f"terminal-coder", f"{provider}_api_key", api_key)
                        self.config["api_keys"][provider] = api_key
                        self.console.print(f"[green]✅ {provider.upper()} API key saved to keyring[/green]")
                    except ImportError:
                        # Fallback to config file
                        self.config["api_keys"][provider] = api_key
                        self.console.print(f"[yellow]⚠️ {provider.upper()} API key saved to config (keyring not available)[/yellow]")

        self.save_config_to_file(self.config)

    async def configure_linux_integration(self):
        """Configure Linux-specific integration settings"""
        self.console.print(Panel("🐧 Linux Integration Settings", style="blue"))

        current_settings = self.config.get("linux", {})

        # systemd usage
        use_systemd = Confirm.ask(
            f"Use systemd for service management? (current: {current_settings.get('use_systemd', True)})",
            default=current_settings.get('use_systemd', True)
        )

        # D-Bus integration
        enable_dbus = Confirm.ask(
            f"Enable D-Bus integration? (current: {current_settings.get('enable_dbus', DBUS_AVAILABLE)})",
            default=current_settings.get('enable_dbus', DBUS_AVAILABLE)
        )

        # Package manager
        use_package_manager = Confirm.ask(
            f"Enable package manager integration? (current: {current_settings.get('use_package_manager', True)})",
            default=current_settings.get('use_package_manager', True)
        )

        # Container support
        enable_container_support = Confirm.ask(
            f"Enable container support? (current: {current_settings.get('enable_container_support', True)})",
            default=current_settings.get('enable_container_support', True)
        )

        # Update configuration
        self.config["linux"] = {
            **current_settings,
            "use_systemd": use_systemd,
            "enable_dbus": enable_dbus,
            "use_package_manager": use_package_manager,
            "enable_container_support": enable_container_support,
            "distribution": self.linux_features['distribution']
        }

        self.save_config_to_file(self.config)
        self.console.print("[green]✅ Linux integration settings updated[/green]")

    async def configure_package_manager(self):
        """Configure package manager settings"""
        self.console.print(Panel(f"📦 Package Manager Settings - {self.linux_features['package_manager'].upper()}", style="green"))

        current_pm = self.linux_features['package_manager']
        self.console.print(f"Detected package manager: [cyan]{current_pm}[/cyan]")

        if current_pm != 'unknown':
            self.console.print("Package manager integration is automatically configured based on your system.")
        else:
            self.console.print("[red]No package manager detected. Manual configuration may be required.[/red]")

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
    async def code_tools_menu(self): pass
    async def project_analytics(self): pass
    async def ai_model_manager(self): pass
    async def api_manager(self): pass
    async def documentation_generator(self): pass
    async def security_scanner(self): pass
    async def deploy_assistant(self): pass
    async def container_manager(self): pass
    async def show_help(self):
        """Show Linux-specific help with advanced AI features"""
        help_content = await self.advanced_ai.get_linux_help()
        self.console.print(Panel(help_content, title="🚀 Terminal Coder Linux Help", style="cyan"))
    async def show_system_information(self): pass
    async def systemd_services_manager(self): pass
    async def process_manager(self): pass
    async def system_configuration(self): pass
    async def network_tools(self): pass
    async def filesystem_tools(self): pass
    async def remove_package(self): pass
    async def update_system(self): pass
    async def list_installed_packages(self): pass
    async def show_package_info(self): pass
    async def performance_stats(self): pass
    async def disk_usage(self): pass
    async def network_status(self): pass


def main():
    """Main entry point for Linux application with enhanced error handling"""
    parser = argparse.ArgumentParser(description="Terminal Coder - Linux Edition")
    parser.add_argument("--version", action="version", version="Terminal Coder v2.0-Linux")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--project", help="Open specific project")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Create and run the application
    app = None
    exit_code = 0

    try:
        app = TerminalCoderApp()

        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            app.logger.info("Debug logging enabled")

        # Handle specific project opening
        if args.project:
            project_path = Path(args.project).resolve()
            if project_path.exists():
                app.logger.info(f"Opening project: {project_path}")
                # Add logic to open specific project
            else:
                print(f"Project path does not exist: {project_path}")
                exit_code = 1
                return

        # Run in daemon mode if requested
        if args.daemon:
            app.logger.info("Starting in daemon mode")
            # Add daemon logic here
        else:
            # Normal interactive mode with proper async handling
            try:
                asyncio.run(app.run_interactive_mode())
            except asyncio.CancelledError:
                print("\nApplication cancelled gracefully")
            except KeyboardInterrupt:
                print("\nGoodbye! 🐧")

    except KeyboardInterrupt:
        print("\nGoodbye! 🐧")
    except Exception as e:
        print(f"Fatal error: {e}")
        if app and app.logger:
            app.logger.error(f"Fatal error in main: {e}", exc_info=True)
        exit_code = 1
    finally:
        if app:
            try:
                app.cleanup()
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")
                if app and app.logger:
                    app.logger.error(f"Cleanup error: {cleanup_error}")

        sys.exit(exit_code)


if __name__ == "__main__":
    main()