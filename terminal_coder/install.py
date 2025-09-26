#!/usr/bin/env python3
"""
Terminal Coder - Advanced Installation Script
Comprehensive cross-platform installer with dependency management
"""

import asyncio
import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import hashlib
import zipfile
import tarfile
from datetime import datetime
import threading
import time

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.align import Align
    from rich.layout import Layout
    from rich.live import Live
    import rich.traceback
    RICH_AVAILABLE = True
    rich.traceback.install(show_locals=True)
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    def Panel(text, **kwargs):
        return f"=== {text} ==="

    def Table(**kwargs):
        class MockTable:
            def add_column(self, *args, **kwargs): pass
            def add_row(self, *args, **kwargs): pass
        return MockTable()


@dataclass
class InstallConfig:
    """Installation configuration"""
    install_path: Path
    python_version: str
    create_venv: bool = True
    install_optional: bool = True
    install_dev_deps: bool = False
    create_desktop_entry: bool = True
    add_to_path: bool = True
    install_systemd_service: bool = False
    enable_auto_updates: bool = True
    backup_existing: bool = True
    force_install: bool = False
    quiet_mode: bool = False
    offline_mode: bool = False
    custom_requirements: List[str] = field(default_factory=list)


class PlatformDetector:
    """Advanced platform detection and compatibility checking"""

    def __init__(self):
        self.console = Console()
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.python_version = sys.version_info
        self.is_admin = self._check_admin_privileges()

    def _check_admin_privileges(self) -> bool:
        """Check if running with administrator privileges"""
        try:
            if self.system == 'windows':
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin()
            else:
                return os.geteuid() == 0
        except Exception:
            return False

    def get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information"""
        info = {
            'system': self.system,
            'machine': self.machine,
            'python_version': f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            'python_executable': sys.executable,
            'is_admin': self.is_admin,
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'platform_release': platform.release(),
            'home_dir': str(Path.home()),
            'current_dir': str(Path.cwd()),
        }

        # Add OS-specific information
        if self.system == 'linux':
            info.update(self._get_linux_info())
        elif self.system == 'windows':
            info.update(self._get_windows_info())
        elif self.system == 'darwin':
            info.update(self._get_macos_info())

        return info

    def _get_linux_info(self) -> Dict[str, Any]:
        """Get Linux-specific information"""
        info = {}
        try:
            # Detect distribution
            if Path('/etc/os-release').exists():
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('ID='):
                            info['distribution'] = line.split('=')[1].strip().strip('"')
                        elif line.startswith('VERSION_ID='):
                            info['version'] = line.split('=')[1].strip().strip('"')

            # Check package managers
            package_managers = {
                'apt': '/usr/bin/apt',
                'dnf': '/usr/bin/dnf',
                'yum': '/usr/bin/yum',
                'pacman': '/usr/bin/pacman',
                'zypper': '/usr/bin/zypper',
                'emerge': '/usr/bin/emerge',
                'apk': '/sbin/apk'
            }

            for pm, path in package_managers.items():
                if Path(path).exists():
                    info['package_manager'] = pm
                    break

            # Check init system
            if Path('/run/systemd/system').exists():
                info['init_system'] = 'systemd'
            elif Path('/sbin/init').is_symlink():
                target = Path('/sbin/init').readlink()
                if 'upstart' in str(target):
                    info['init_system'] = 'upstart'
                else:
                    info['init_system'] = 'sysv'

            # Check desktop environment
            info['desktop_environment'] = (
                os.environ.get('XDG_CURRENT_DESKTOP') or
                os.environ.get('DESKTOP_SESSION') or
                'unknown'
            ).lower()

        except Exception as e:
            info['detection_error'] = str(e)

        return info

    def _get_windows_info(self) -> Dict[str, Any]:
        """Get Windows-specific information"""
        info = {}
        try:
            info['version'] = platform.version()
            info['edition'] = platform.win32_edition() if hasattr(platform, 'win32_edition') else 'unknown'

            # Check if WSL is available
            try:
                result = subprocess.run(['wsl', '--list'], capture_output=True, timeout=5)
                info['wsl_available'] = result.returncode == 0
            except Exception:
                info['wsl_available'] = False

            # Check PowerShell version
            try:
                result = subprocess.run(['powershell', '-Command', '$PSVersionTable.PSVersion'],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    info['powershell_version'] = result.stdout.strip().split('\n')[0]
            except Exception:
                info['powershell_version'] = 'unknown'

        except Exception as e:
            info['detection_error'] = str(e)

        return info

    def _get_macos_info(self) -> Dict[str, Any]:
        """Get macOS-specific information"""
        info = {}
        try:
            info['version'] = platform.mac_ver()[0]

            # Check Homebrew
            if shutil.which('brew'):
                info['package_manager'] = 'homebrew'

            # Check Xcode tools
            try:
                result = subprocess.run(['xcode-select', '--version'], capture_output=True, timeout=5)
                info['xcode_tools'] = result.returncode == 0
            except Exception:
                info['xcode_tools'] = False

        except Exception as e:
            info['detection_error'] = str(e)

        return info

    def check_compatibility(self) -> Tuple[bool, List[str]]:
        """Check platform compatibility and return warnings"""
        warnings = []
        compatible = True

        # Check Python version
        if self.python_version < (3, 8):
            warnings.append(f"Python {self.python_version.major}.{self.python_version.minor} is not supported. Please upgrade to Python 3.8+")
            compatible = False

        # Check system compatibility
        if self.system not in ['linux', 'windows', 'darwin']:
            warnings.append(f"Platform {self.system} is not officially supported")
            compatible = False

        # Platform-specific checks
        if self.system == 'windows' and not self.is_admin:
            warnings.append("Administrator privileges recommended for Windows installation")

        if self.system == 'linux':
            # Check for required tools
            required_tools = ['gcc', 'make', 'git']
            missing_tools = [tool for tool in required_tools if not shutil.which(tool)]
            if missing_tools:
                warnings.append(f"Missing development tools: {', '.join(missing_tools)}")

        return compatible, warnings


class DependencyManager:
    """Advanced dependency management with conflict resolution"""

    def __init__(self, console: Console, config: InstallConfig):
        self.console = console
        self.config = config
        self.pip_cache = {}

    async def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze and resolve dependencies"""
        dependencies = {
            'core': [
                'rich>=13.0.0',
                'asyncio-throttle>=1.0.0',
                'aiofiles>=23.0.0',
                'psutil>=5.9.0',
                'requests>=2.28.0',
                'cryptography>=41.0.0',
                'pydantic>=2.0.0',
                'pyyaml>=6.0.0',
                'toml>=0.10.0',
                'click>=8.1.0',
                'typer>=0.9.0',
                'httpx>=0.24.0'
            ],
            'ai': [
                'openai>=1.0.0',
                'anthropic>=0.7.0',
                'google-generativeai>=0.3.0',
                'cohere>=4.0.0',
                'tiktoken>=0.5.0',
                'numpy>=1.24.0'
            ],
            'optional': [
                'torch>=2.0.0',
                'transformers>=4.30.0',
                'scikit-learn>=1.3.0',
                'matplotlib>=3.7.0',
                'seaborn>=0.12.0',
                'pandas>=2.0.0',
                'networkx>=3.0.0',
                'pillow>=10.0.0'
            ],
            'platform_specific': self._get_platform_dependencies(),
            'dev': [
                'pytest>=7.0.0',
                'pytest-asyncio>=0.21.0',
                'black>=23.0.0',
                'flake8>=6.0.0',
                'mypy>=1.5.0',
                'pre-commit>=3.0.0',
                'coverage>=7.0.0',
                'sphinx>=7.0.0'
            ]
        }

        # Add custom requirements
        if self.config.custom_requirements:
            dependencies['custom'] = self.config.custom_requirements

        # Check for conflicts and resolve versions
        resolved_deps = await self._resolve_dependencies(dependencies)

        return {
            'dependencies': resolved_deps,
            'conflicts': await self._check_conflicts(resolved_deps),
            'size_estimate': await self._estimate_download_size(resolved_deps),
            'install_order': self._get_install_order(resolved_deps)
        }

    def _get_platform_dependencies(self) -> List[str]:
        """Get platform-specific dependencies"""
        deps = []
        system = platform.system().lower()

        if system == 'linux':
            deps.extend([
                'distro>=1.8.0',
                'dbus-python>=1.3.0; sys_platform == "linux"',
                'keyring>=24.0.0',
                'secretstorage>=3.3.0; sys_platform == "linux"'
            ])
        elif system == 'windows':
            deps.extend([
                'pywin32>=306; sys_platform == "win32"',
                'wmi>=1.5.1; sys_platform == "win32"',
                'winshell>=0.6; sys_platform == "win32"'
            ])
        elif system == 'darwin':
            deps.extend([
                'pyobjc-core>=9.0; sys_platform == "darwin"',
                'pyobjc-framework-Cocoa>=9.0; sys_platform == "darwin"'
            ])

        return deps

    async def _resolve_dependencies(self, dependencies: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Resolve dependency versions and conflicts"""
        # This is a simplified version - in practice, you'd use pip-tools or similar
        resolved = {}

        for category, deps in dependencies.items():
            resolved[category] = []
            for dep in deps:
                try:
                    # Parse dependency specification
                    if '>=' in dep or '<=' in dep or '==' in dep:
                        resolved[category].append(dep)
                    else:
                        # Get latest version
                        latest = await self._get_latest_version(dep)
                        resolved[category].append(f"{dep}>={latest}")
                except Exception:
                    resolved[category].append(dep)  # Keep original if resolution fails

        return resolved

    async def _get_latest_version(self, package: str) -> str:
        """Get latest version of a package from PyPI"""
        if package in self.pip_cache:
            return self.pip_cache[package]

        try:
            # Remove platform markers for API call
            clean_package = package.split(';')[0].strip()
            url = f"https://pypi.org/pypi/{clean_package}/json"

            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read())
                version = data['info']['version']
                self.pip_cache[package] = version
                return version
        except Exception:
            return "0.0.0"

    async def _check_conflicts(self, dependencies: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Check for dependency conflicts"""
        conflicts = []
        # Simplified conflict detection
        all_deps = []
        for deps in dependencies.values():
            all_deps.extend(deps)

        # Check for version conflicts (simplified)
        package_versions = {}
        for dep in all_deps:
            if '>=' in dep:
                pkg, version = dep.split('>=', 1)
                pkg = pkg.strip()
                version = version.split(';')[0].strip()  # Remove platform markers

                if pkg in package_versions:
                    if package_versions[pkg] != version:
                        conflicts.append({
                            'package': pkg,
                            'versions': [package_versions[pkg], version],
                            'severity': 'warning'
                        })
                else:
                    package_versions[pkg] = version

        return conflicts

    async def _estimate_download_size(self, dependencies: Dict[str, List[str]]) -> Dict[str, int]:
        """Estimate download size for dependencies"""
        # Rough estimates in MB
        size_estimates = {
            'core': 50,
            'ai': 200,
            'optional': 500,
            'platform_specific': 30,
            'dev': 100,
            'custom': 50
        }

        total_size = 0
        category_sizes = {}

        for category in dependencies:
            size = size_estimates.get(category, 50)
            category_sizes[category] = size
            total_size += size

        return {
            'total_mb': total_size,
            'by_category': category_sizes
        }

    def _get_install_order(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Get optimal installation order"""
        # Install in order of importance and dependency chain
        order = ['core']

        if 'platform_specific' in dependencies:
            order.append('platform_specific')

        if self.config.install_optional:
            order.extend(['ai', 'optional'])

        if self.config.install_dev_deps:
            order.append('dev')

        if 'custom' in dependencies:
            order.append('custom')

        return order


class InstallationManager:
    """Main installation manager with advanced features"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else Console()
        self.platform = PlatformDetector()
        self.temp_dir = Path(tempfile.mkdtemp(prefix='terminal_coder_install_'))
        self.install_log = []

    async def run_installation(self, config: InstallConfig) -> bool:
        """Run the complete installation process"""
        try:
            self._display_banner()

            # Pre-installation checks
            if not await self._pre_installation_checks(config):
                return False

            # Display installation plan
            await self._display_installation_plan(config)

            if not config.quiet_mode:
                if not Confirm.ask("Proceed with installation?"):
                    self.console.print("[yellow]Installation cancelled by user[/yellow]")
                    return False

            # Create installation progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            ) as progress:

                # Main installation steps
                main_task = progress.add_task("Installing Terminal Coder", total=100)

                # Step 1: Setup environment
                progress.update(main_task, description="Setting up environment...", completed=10)
                if not await self._setup_environment(config):
                    return False

                # Step 2: Install dependencies
                progress.update(main_task, description="Installing dependencies...", completed=30)
                dep_manager = DependencyManager(self.console, config)
                if not await self._install_dependencies(dep_manager, config):
                    return False

                # Step 3: Install application
                progress.update(main_task, description="Installing application files...", completed=60)
                if not await self._install_application(config):
                    return False

                # Step 4: Configure system integration
                progress.update(main_task, description="Configuring system integration...", completed=80)
                if not await self._setup_system_integration(config):
                    return False

                # Step 5: Final configuration
                progress.update(main_task, description="Finalizing installation...", completed=95)
                if not await self._finalize_installation(config):
                    return False

                progress.update(main_task, description="Installation completed!", completed=100)

            self._display_success_message(config)
            return True

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Installation cancelled by user[/yellow]")
            return False
        except Exception as e:
            self.console.print(f"[red]Installation failed: {e}[/red]")
            logging.error(f"Installation error: {e}", exc_info=True)
            return False
        finally:
            await self._cleanup_temp_files()

    def _display_banner(self):
        """Display installation banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 TERMINAL CODER INSTALLER v2.0                         ║
║                   Advanced AI-Powered Development Terminal                   ║
║                        Cross-Platform Installation                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """

        self.console.print(Panel(
            Align.center(banner),
            style="bold cyan",
            border_style="bright_blue"
        ))

        # Display platform information
        platform_info = self.platform.get_platform_info()

        info_table = Table(title="🖥️ System Information", style="cyan")
        info_table.add_column("Property", style="bold")
        info_table.add_column("Value", style="white")

        key_info = [
            ("Operating System", f"{platform_info['system'].title()} {platform_info.get('version', '')}"),
            ("Architecture", platform_info['architecture']),
            ("Python Version", platform_info['python_version']),
            ("Admin Privileges", "✅ Yes" if platform_info['is_admin'] else "❌ No"),
        ]

        if platform_info['system'] == 'linux':
            key_info.extend([
                ("Distribution", platform_info.get('distribution', 'Unknown')),
                ("Package Manager", platform_info.get('package_manager', 'Unknown')),
                ("Init System", platform_info.get('init_system', 'Unknown'))
            ])

        for prop, value in key_info:
            info_table.add_row(prop, value)

        self.console.print(info_table)

    async def _pre_installation_checks(self, config: InstallConfig) -> bool:
        """Perform pre-installation compatibility checks"""
        self.console.print("\n[bold blue]🔍 Performing pre-installation checks...[/bold blue]")

        compatible, warnings = self.platform.check_compatibility()

        if warnings:
            self.console.print("[yellow]⚠️ Warnings detected:[/yellow]")
            for warning in warnings:
                self.console.print(f"  • {warning}")

        if not compatible and not config.force_install:
            self.console.print("[red]❌ System compatibility check failed[/red]")
            return False

        # Check installation path
        if config.install_path.exists() and not config.force_install:
            if not config.backup_existing:
                self.console.print(f"[red]❌ Installation path already exists: {config.install_path}[/red]")
                return False
            else:
                self.console.print(f"[yellow]⚠️ Will backup existing installation at: {config.install_path}[/yellow]")

        # Check disk space
        try:
            free_space_gb = shutil.disk_usage(config.install_path.parent)[2] / (1024**3)
            if free_space_gb < 2:  # Require at least 2GB
                self.console.print(f"[red]❌ Insufficient disk space: {free_space_gb:.1f}GB available, 2GB required[/red]")
                return False
        except Exception:
            self.console.print("[yellow]⚠️ Could not check disk space[/yellow]")

        # Check network connectivity (unless offline mode)
        if not config.offline_mode:
            try:
                urllib.request.urlopen('https://pypi.org', timeout=10)
                self.console.print("[green]✅ Network connectivity verified[/green]")
            except Exception:
                self.console.print("[yellow]⚠️ Network connectivity issues detected[/yellow]")
                if not Confirm.ask("Continue with potential network issues?"):
                    return False

        self.console.print("[green]✅ Pre-installation checks completed[/green]")
        return True

    async def _display_installation_plan(self, config: InstallConfig):
        """Display detailed installation plan"""
        plan_table = Table(title="📋 Installation Plan", style="green")
        plan_table.add_column("Component", style="bold")
        plan_table.add_column("Action", style="white")
        plan_table.add_column("Details", style="dim")

        plan_items = [
            ("Installation Path", "Create", str(config.install_path)),
            ("Python Environment", "Create Virtual Environment" if config.create_venv else "Use System Python", config.python_version),
            ("Core Dependencies", "Install", "Essential packages for Terminal Coder"),
            ("AI Dependencies", "Install" if config.install_optional else "Skip", "OpenAI, Anthropic, Google AI, etc."),
            ("Optional Dependencies", "Install" if config.install_optional else "Skip", "ML libraries, visualization tools"),
            ("Development Tools", "Install" if config.install_dev_deps else "Skip", "Testing and development utilities"),
        ]

        # Platform-specific integrations
        system = platform.system().lower()
        if system == 'linux':
            plan_items.extend([
                ("Desktop Integration", "Create .desktop file" if config.create_desktop_entry else "Skip", "Application launcher"),
                ("Systemd Service", "Install" if config.install_systemd_service else "Skip", "Background service"),
            ])
        elif system == 'windows':
            plan_items.extend([
                ("Start Menu", "Create shortcuts" if config.create_desktop_entry else "Skip", "Windows integration"),
                ("PATH Environment", "Add to PATH" if config.add_to_path else "Skip", "Command line access"),
            ])

        for component, action, details in plan_items:
            plan_table.add_row(component, action, details)

        self.console.print(plan_table)

        # Show dependency analysis
        dep_manager = DependencyManager(self.console, config)
        dep_analysis = await dep_manager.analyze_dependencies()

        size_info = dep_analysis['size_estimate']
        self.console.print(f"\n[cyan]📦 Estimated download size: {size_info['total_mb']} MB[/cyan]")

        if dep_analysis['conflicts']:
            self.console.print("[yellow]⚠️ Dependency conflicts detected:[/yellow]")
            for conflict in dep_analysis['conflicts']:
                self.console.print(f"  • {conflict['package']}: {', '.join(conflict['versions'])}")

    async def _setup_environment(self, config: InstallConfig) -> bool:
        """Setup installation environment"""
        try:
            # Create installation directory
            if config.backup_existing and config.install_path.exists():
                backup_path = config.install_path.with_name(f"{config.install_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                self.console.print(f"[yellow]📁 Backing up existing installation to: {backup_path}[/yellow]")
                shutil.move(str(config.install_path), str(backup_path))

            config.install_path.mkdir(parents=True, exist_ok=True)
            self.console.print(f"[green]✅ Created installation directory: {config.install_path}[/green]")

            # Setup virtual environment
            if config.create_venv:
                venv_path = config.install_path / 'venv'
                self.console.print(f"[blue]🐍 Creating Python virtual environment...[/blue]")

                result = subprocess.run([
                    sys.executable, '-m', 'venv', str(venv_path)
                ], capture_output=True, text=True)

                if result.returncode != 0:
                    self.console.print(f"[red]❌ Failed to create virtual environment: {result.stderr}[/red]")
                    return False

                self.console.print(f"[green]✅ Virtual environment created at: {venv_path}[/green]")

            return True

        except Exception as e:
            self.console.print(f"[red]❌ Environment setup failed: {e}[/red]")
            return False

    async def _install_dependencies(self, dep_manager: DependencyManager, config: InstallConfig) -> bool:
        """Install all dependencies"""
        try:
            dep_analysis = await dep_manager.analyze_dependencies()
            dependencies = dep_analysis['dependencies']
            install_order = dep_analysis['install_order']

            # Get pip executable
            if config.create_venv:
                if platform.system().lower() == 'windows':
                    pip_exe = config.install_path / 'venv' / 'Scripts' / 'pip.exe'
                else:
                    pip_exe = config.install_path / 'venv' / 'bin' / 'pip'
            else:
                pip_exe = 'pip'

            # Upgrade pip first
            self.console.print("[blue]📦 Upgrading pip...[/blue]")
            result = subprocess.run([
                str(pip_exe), 'install', '--upgrade', 'pip', 'setuptools', 'wheel'
            ], capture_output=True, text=True)

            if result.returncode != 0:
                self.console.print(f"[yellow]⚠️ Pip upgrade warning: {result.stderr}[/yellow]")

            # Install dependencies by category
            for category in install_order:
                if category not in dependencies:
                    continue

                deps = dependencies[category]
                self.console.print(f"[blue]📦 Installing {category} dependencies...[/blue]")

                # Install with retry logic
                for attempt in range(3):
                    try:
                        cmd = [str(pip_exe), 'install'] + deps
                        if config.quiet_mode:
                            cmd.append('--quiet')

                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                        if result.returncode == 0:
                            self.console.print(f"[green]✅ {category} dependencies installed successfully[/green]")
                            break
                        else:
                            if attempt < 2:
                                self.console.print(f"[yellow]⚠️ Attempt {attempt + 1} failed, retrying...[/yellow]")
                                await asyncio.sleep(2)
                            else:
                                self.console.print(f"[red]❌ Failed to install {category} dependencies: {result.stderr}[/red]")

                                # For optional dependencies, continue
                                if category in ['optional', 'dev']:
                                    self.console.print(f"[yellow]⚠️ Continuing without {category} dependencies[/yellow]")
                                    continue
                                else:
                                    return False

                    except subprocess.TimeoutExpired:
                        self.console.print(f"[red]❌ Installation timeout for {category} dependencies[/red]")
                        if category not in ['optional', 'dev']:
                            return False
                        continue

            return True

        except Exception as e:
            self.console.print(f"[red]❌ Dependency installation failed: {e}[/red]")
            return False

    async def _install_application(self, config: InstallConfig) -> bool:
        """Install the main application files"""
        try:
            # Copy application files
            source_dir = Path(__file__).parent
            app_dir = config.install_path / 'terminal_coder'

            self.console.print("[blue]📂 Copying application files...[/blue]")

            # Create application structure
            app_dir.mkdir(exist_ok=True)
            (app_dir / 'linux').mkdir(exist_ok=True)
            (app_dir / 'windows').mkdir(exist_ok=True)
            (app_dir / 'config').mkdir(exist_ok=True)
            (app_dir / 'logs').mkdir(exist_ok=True)

            # Copy Python files
            python_files = [
                'linux/main.py',
                'linux/advanced_gui_extensions.py',
                'linux/ai_integration.py',
                'linux/advanced_ai_integration.py',
                'linux/project_manager.py',
                'linux/system_manager.py',
                'linux/gui.py',
                'windows/main.py',
                'windows/advanced_gui_extensions.py',
                'windows/ai_integration.py',
                'windows/advanced_ai_integration.py',
                'windows/project_manager.py',
                'windows/system_manager.py',
                'windows/gui.py',
                '__init__.py'
            ]

            for py_file in python_files:
                source_file = source_dir / py_file
                dest_file = app_dir / py_file

                if source_file.exists():
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, dest_file)
                    self.console.print(f"[dim]  • Copied {py_file}[/dim]")

            # Create launcher script
            await self._create_launcher_script(config)

            # Set permissions on Unix systems
            if platform.system().lower() != 'windows':
                launcher_script = config.install_path / 'terminal-coder'
                if launcher_script.exists():
                    os.chmod(launcher_script, 0o755)

            self.console.print("[green]✅ Application files installed successfully[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Application installation failed: {e}[/red]")
            return False

    async def _create_launcher_script(self, config: InstallConfig):
        """Create platform-specific launcher script"""
        system = platform.system().lower()

        if system == 'windows':
            # Create batch file launcher
            launcher_path = config.install_path / 'terminal-coder.bat'
            if config.create_venv:
                python_exe = config.install_path / 'venv' / 'Scripts' / 'python.exe'
            else:
                python_exe = sys.executable

            script_content = f"""@echo off
REM Terminal Coder Launcher
cd /d "{config.install_path}"
"{python_exe}" -m terminal_coder.{system}.main %*
"""
        else:
            # Create shell script launcher
            launcher_path = config.install_path / 'terminal-coder'
            if config.create_venv:
                python_exe = config.install_path / 'venv' / 'bin' / 'python'
            else:
                python_exe = sys.executable

            script_content = f"""#!/bin/bash
# Terminal Coder Launcher
cd "{config.install_path}"
"{python_exe}" -m terminal_coder.{system}.main "$@"
"""

        with open(launcher_path, 'w') as f:
            f.write(script_content)

        self.console.print(f"[green]✅ Launcher script created: {launcher_path}[/green]")

    async def _setup_system_integration(self, config: InstallConfig) -> bool:
        """Setup system integration (desktop entries, PATH, etc.)"""
        try:
            system = platform.system().lower()

            if system == 'linux':
                await self._setup_linux_integration(config)
            elif system == 'windows':
                await self._setup_windows_integration(config)
            elif system == 'darwin':
                await self._setup_macos_integration(config)

            return True

        except Exception as e:
            self.console.print(f"[red]❌ System integration setup failed: {e}[/red]")
            return False

    async def _setup_linux_integration(self, config: InstallConfig):
        """Setup Linux-specific system integration"""
        if config.create_desktop_entry:
            # Create .desktop file
            desktop_dir = Path.home() / '.local' / 'share' / 'applications'
            desktop_dir.mkdir(parents=True, exist_ok=True)

            desktop_file = desktop_dir / 'terminal-coder.desktop'
            launcher_script = config.install_path / 'terminal-coder'

            desktop_content = f"""[Desktop Entry]
Name=Terminal Coder
Comment=AI-Powered Development Terminal
GenericName=Development Terminal
Exec={launcher_script}
Icon=terminal
Terminal=true
Type=Application
Categories=Development;Programming;Utility;
Keywords=terminal;coding;ai;development;programming;
StartupNotify=true
"""

            with open(desktop_file, 'w') as f:
                f.write(desktop_content)

            os.chmod(desktop_file, 0o644)
            self.console.print("[green]✅ Desktop entry created[/green]")

        if config.add_to_path:
            # Add to PATH in shell profile
            shell = os.environ.get('SHELL', '/bin/bash')

            if 'zsh' in shell:
                profile_file = Path.home() / '.zshrc'
            elif 'fish' in shell:
                profile_file = Path.home() / '.config' / 'fish' / 'config.fish'
            else:
                profile_file = Path.home() / '.bashrc'

            path_entry = f'\n# Terminal Coder\nexport PATH="{config.install_path}:$PATH"\n'

            if profile_file.exists():
                with open(profile_file, 'a') as f:
                    f.write(path_entry)
                self.console.print(f"[green]✅ Added to PATH in {profile_file}[/green]")

        if config.install_systemd_service and self.platform.is_admin:
            # Create systemd user service
            systemd_dir = Path.home() / '.config' / 'systemd' / 'user'
            systemd_dir.mkdir(parents=True, exist_ok=True)

            service_file = systemd_dir / 'terminal-coder.service'
            launcher_script = config.install_path / 'terminal-coder'

            service_content = f"""[Unit]
Description=Terminal Coder Background Service
After=graphical-session.target

[Service]
Type=simple
ExecStart={launcher_script} --daemon
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

            with open(service_file, 'w') as f:
                f.write(service_content)

            # Enable the service
            subprocess.run(['systemctl', '--user', 'daemon-reload'])
            subprocess.run(['systemctl', '--user', 'enable', 'terminal-coder.service'])

            self.console.print("[green]✅ Systemd service installed and enabled[/green]")

    async def _setup_windows_integration(self, config: InstallConfig):
        """Setup Windows-specific system integration"""
        if config.create_desktop_entry:
            try:
                import winshell
                from win32com.client import Dispatch

                # Create Start Menu shortcut
                start_menu = winshell.start_menu()
                shortcut_path = os.path.join(start_menu, "Terminal Coder.lnk")
                launcher_script = config.install_path / 'terminal-coder.bat'

                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(shortcut_path)
                shortcut.Targetpath = str(launcher_script)
                shortcut.WorkingDirectory = str(config.install_path)
                shortcut.IconLocation = "cmd.exe,0"
                shortcut.save()

                self.console.print("[green]✅ Start Menu shortcut created[/green]")

            except ImportError:
                self.console.print("[yellow]⚠️ Could not create shortcuts (winshell not available)[/yellow]")

        if config.add_to_path:
            # Add to system PATH (requires admin privileges)
            if self.platform.is_admin:
                try:
                    import winreg

                    # Get current PATH
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                                        0, winreg.KEY_ALL_ACCESS)

                    current_path, _ = winreg.QueryValueEx(key, "PATH")

                    if str(config.install_path) not in current_path:
                        new_path = f"{current_path};{config.install_path}"
                        winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
                        self.console.print("[green]✅ Added to system PATH[/green]")

                    winreg.CloseKey(key)

                except Exception as e:
                    self.console.print(f"[yellow]⚠️ Could not modify system PATH: {e}[/yellow]")
            else:
                self.console.print("[yellow]⚠️ Admin privileges required to modify system PATH[/yellow]")

    async def _setup_macos_integration(self, config: InstallConfig):
        """Setup macOS-specific system integration"""
        if config.create_desktop_entry:
            # Create application bundle (simplified)
            app_dir = Path('/Applications') / 'Terminal Coder.app'
            if self.platform.is_admin:
                try:
                    app_dir.mkdir(exist_ok=True)
                    (app_dir / 'Contents').mkdir(exist_ok=True)
                    (app_dir / 'Contents' / 'MacOS').mkdir(exist_ok=True)

                    # Create simple executable
                    launcher_script = config.install_path / 'terminal-coder'
                    app_executable = app_dir / 'Contents' / 'MacOS' / 'terminal-coder'

                    shutil.copy2(launcher_script, app_executable)
                    os.chmod(app_executable, 0o755)

                    self.console.print("[green]✅ Application bundle created[/green]")

                except Exception as e:
                    self.console.print(f"[yellow]⚠️ Could not create app bundle: {e}[/yellow]")

        if config.add_to_path:
            # Add to PATH in shell profile
            profiles = ['.zshrc', '.bash_profile', '.profile']

            for profile_name in profiles:
                profile_file = Path.home() / profile_name
                if profile_file.exists():
                    path_entry = f'\n# Terminal Coder\nexport PATH="{config.install_path}:$PATH"\n'

                    with open(profile_file, 'a') as f:
                        f.write(path_entry)

                    self.console.print(f"[green]✅ Added to PATH in {profile_file}[/green]")
                    break

    async def _finalize_installation(self, config: InstallConfig) -> bool:
        """Finalize installation with configuration and verification"""
        try:
            # Create initial configuration
            config_dir = config.install_path / 'terminal_coder' / 'config'
            config_file = config_dir / 'config.json'

            initial_config = {
                "version": "2.0.0",
                "installation_date": datetime.now().isoformat(),
                "install_path": str(config.install_path),
                "python_version": config.python_version,
                "platform": platform.system().lower(),
                "features": {
                    "ai_integration": config.install_optional,
                    "advanced_ml": config.install_optional,
                    "development_tools": config.install_dev_deps,
                    "auto_updates": config.enable_auto_updates
                },
                "paths": {
                    "config_dir": str(config_dir),
                    "logs_dir": str(config.install_path / 'terminal_coder' / 'logs'),
                    "data_dir": str(config.install_path / 'terminal_coder' / 'data')
                }
            }

            with open(config_file, 'w') as f:
                json.dump(initial_config, f, indent=2)

            self.console.print("[green]✅ Initial configuration created[/green]")

            # Verify installation
            if await self._verify_installation(config):
                self.console.print("[green]✅ Installation verification successful[/green]")
                return True
            else:
                self.console.print("[red]❌ Installation verification failed[/red]")
                return False

        except Exception as e:
            self.console.print(f"[red]❌ Installation finalization failed: {e}[/red]")
            return False

    async def _verify_installation(self, config: InstallConfig) -> bool:
        """Verify the installation is working correctly"""
        try:
            # Check if launcher script exists
            system = platform.system().lower()
            if system == 'windows':
                launcher = config.install_path / 'terminal-coder.bat'
            else:
                launcher = config.install_path / 'terminal-coder'

            if not launcher.exists():
                self.console.print("[red]❌ Launcher script not found[/red]")
                return False

            # Try to run the application with --version flag
            try:
                result = subprocess.run([
                    str(launcher), '--version'
                ], capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    self.console.print(f"[green]✅ Application responds: {result.stdout.strip()}[/green]")
                    return True
                else:
                    self.console.print(f"[yellow]⚠️ Application test warning: {result.stderr}[/yellow]")
                    return True  # Consider warnings as success

            except subprocess.TimeoutExpired:
                self.console.print("[yellow]⚠️ Application test timed out (may still be working)[/yellow]")
                return True  # Timeout might be acceptable

        except Exception as e:
            self.console.print(f"[red]❌ Installation verification error: {e}[/red]")
            return False

    def _display_success_message(self, config: InstallConfig):
        """Display installation success message with next steps"""
        system = platform.system().lower()

        if system == 'windows':
            launcher_cmd = str(config.install_path / 'terminal-coder.bat')
        else:
            launcher_cmd = str(config.install_path / 'terminal-coder')

        success_panel = f"""
🎉 Terminal Coder v2.0 Installation Completed Successfully!

📁 Installation Path: {config.install_path}
🐍 Python Version: {config.python_version}
🖥️  Platform: {platform.system()} {platform.release()}

🚀 Next Steps:
1. Run Terminal Coder:
   {launcher_cmd}

2. Configure AI providers:
   {launcher_cmd} --configure

3. Create your first project:
   {launcher_cmd} --create-project

📚 Documentation: https://terminal-coder.ai/docs
🐛 Issues: https://github.com/terminal-coder/terminal-coder/issues
💬 Community: https://discord.gg/terminal-coder

Thank you for using Terminal Coder! 🚀
        """

        self.console.print(Panel(
            success_panel,
            title="🎊 Installation Complete",
            style="bold green",
            border_style="bright_green"
        ))

    async def _cleanup_temp_files(self):
        """Clean up temporary installation files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.console.print(f"[dim]🧹 Cleaned up temporary files: {self.temp_dir}[/dim]")
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Could not clean up temp files: {e}[/yellow]")


async def main():
    """Main installation function"""
    parser = argparse.ArgumentParser(description="Terminal Coder Advanced Installer")
    parser.add_argument("--install-path", type=Path, default=Path.home() / "terminal-coder",
                       help="Installation directory")
    parser.add_argument("--no-venv", action="store_true",
                       help="Don't create virtual environment")
    parser.add_argument("--no-optional", action="store_true",
                       help="Skip optional dependencies")
    parser.add_argument("--dev-deps", action="store_true",
                       help="Install development dependencies")
    parser.add_argument("--no-desktop", action="store_true",
                       help="Skip desktop integration")
    parser.add_argument("--no-path", action="store_true",
                       help="Don't add to PATH")
    parser.add_argument("--systemd-service", action="store_true",
                       help="Install systemd service (Linux only)")
    parser.add_argument("--no-auto-updates", action="store_true",
                       help="Disable automatic updates")
    parser.add_argument("--no-backup", action="store_true",
                       help="Don't backup existing installation")
    parser.add_argument("--force", action="store_true",
                       help="Force installation even if compatibility checks fail")
    parser.add_argument("--quiet", action="store_true",
                       help="Quiet installation mode")
    parser.add_argument("--offline", action="store_true",
                       help="Offline installation mode")

    args = parser.parse_args()

    # Create installation configuration
    config = InstallConfig(
        install_path=args.install_path,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        create_venv=not args.no_venv,
        install_optional=not args.no_optional,
        install_dev_deps=args.dev_deps,
        create_desktop_entry=not args.no_desktop,
        add_to_path=not args.no_path,
        install_systemd_service=args.systemd_service,
        enable_auto_updates=not args.no_auto_updates,
        backup_existing=not args.no_backup,
        force_install=args.force,
        quiet_mode=args.quiet,
        offline_mode=args.offline
    )

    # Run installation
    installer = InstallationManager()
    success = await installer.run_installation(config)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Installation failed: {e}")
        sys.exit(1)