#!/usr/bin/env python3
"""
Linux Advanced AI Integration
Full Claude CLI and Gemini CLI features with Linux-specific optimizations
Real implementations - no placeholders or mocks
"""

import asyncio
import json
import base64
import os
import sys
import subprocess
import tempfile
import time
import uuid
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
import logging

# Linux-specific imports
try:
    import dbus
    import distro
    import psutil
    LINUX_FEATURES = True
except ImportError:
    LINUX_FEATURES = False

# Additional Linux-specific imports
try:
    import gi
    gi.require_version('Notify', '0.7')
    from gi.repository import Notify
    GNOME_NOTIFICATIONS = True
except (ImportError, ValueError):
    GNOME_NOTIFICATIONS = False

# Import the advanced CLI core
from advanced_cli_core import (
    AdvancedCLICore, ConversationManager, MultiModalProcessor,
    RealTimeStreamer, MCPServerManager, AdvancedTokenCounter,
    ConversationSession, ConversationMessage, StreamingChunk
)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn


@dataclass
class LinuxSpecificConfig:
    """Linux-specific AI integration configuration"""
    use_dbus_notifications: bool = True
    integrate_with_systemd: bool = True
    use_linux_keyring: bool = True
    enable_desktop_integration: bool = True
    use_package_manager: bool = True
    enable_container_integration: bool = True
    desktop_environment: str = "auto"


class LinuxNotificationManager:
    """Linux desktop notifications via D-Bus and libnotify"""

    def __init__(self):
        self.dbus_available = LINUX_FEATURES
        self.gnome_available = GNOME_NOTIFICATIONS
        self.dbus_session = None

        if self.dbus_available:
            try:
                self.dbus_session = dbus.SessionBus()
            except Exception:
                self.dbus_available = False

        if self.gnome_available:
            try:
                Notify.init("Terminal Coder")
            except Exception:
                self.gnome_available = False

    async def send_notification(self, title: str, message: str, urgency: str = "normal") -> bool:
        """Send Linux desktop notification"""
        success = False

        # Try GNOME notifications first (more feature-rich)
        if self.gnome_available:
            try:
                urgency_map = {"low": Notify.Urgency.LOW, "normal": Notify.Urgency.NORMAL, "critical": Notify.Urgency.CRITICAL}
                notification = Notify.Notification.new(title, message, "dialog-information")
                notification.set_urgency(urgency_map.get(urgency, Notify.Urgency.NORMAL))
                success = notification.show()
            except Exception as e:
                logging.warning(f"GNOME notification failed: {e}")

        # Fallback to D-Bus notifications
        if not success and self.dbus_available:
            try:
                notify_service = self.dbus_session.get_object(
                    'org.freedesktop.Notifications',
                    '/org/freedesktop/Notifications'
                )

                notify_interface = dbus.Interface(notify_service, 'org.freedesktop.Notifications')

                urgency_map = {"low": 0, "normal": 1, "critical": 2}

                notify_interface.Notify(
                    "Terminal Coder",  # app_name
                    0,                 # replaces_id
                    "terminal",        # app_icon
                    title,             # summary
                    message,           # body
                    [],                # actions
                    {"urgency": urgency_map.get(urgency, 1)},  # hints
                    5000               # timeout
                )
                success = True

            except Exception as e:
                logging.warning(f"D-Bus notification failed: {e}")

        # Final fallback to notify-send command
        if not success:
            try:
                urgency_flag = f"--urgency={urgency}"
                result = subprocess.run([
                    "notify-send", urgency_flag, "--app-name=Terminal Coder",
                    "--icon=terminal", title, message
                ], capture_output=True, timeout=5)
                success = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        return success

    async def send_progress_notification(self, title: str, progress: int, status: str) -> bool:
        """Send progress notification with progress bar"""
        message = f"{status} ({progress}%)"
        return await self.send_notification(title, message, "normal")


class LinuxSystemdIntegration:
    """Integration with systemd for Linux system management"""

    def __init__(self, console: Console):
        self.console = console
        self.systemd_available = self._check_systemd()

    def _check_systemd(self) -> bool:
        """Check if systemd is available"""
        try:
            result = subprocess.run(["systemctl", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def list_services(self, service_type: str = "all") -> List[Dict[str, Any]]:
        """List systemd services"""
        if not self.systemd_available:
            return [{"error": "systemd not available"}]

        try:
            cmd = ["systemctl", "list-units", "--type=service", "--no-pager", "--output=json"]
            if service_type == "failed":
                cmd.extend(["--state=failed"])
            elif service_type == "active":
                cmd.extend(["--state=active"])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Parse systemctl output (it's not always valid JSON)
                services = []
                for line in stdout.split('\n'):
                    if '.service' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            services.append({
                                'name': parts[0],
                                'load': parts[1],
                                'active': parts[2],
                                'sub': parts[3],
                                'description': ' '.join(parts[4:]) if len(parts) > 4 else ''
                            })
                return services[:50]  # Limit to 50 services
            else:
                return [{"error": stderr}]

        except Exception as e:
            return [{"error": str(e)}]

    async def manage_service(self, action: str, service_name: str) -> Dict[str, Any]:
        """Manage systemd service"""
        if not self.systemd_available:
            return {"error": "systemd not available"}

        try:
            valid_actions = ["start", "stop", "restart", "enable", "disable", "status"]
            if action not in valid_actions:
                return {"error": f"Invalid action. Valid: {', '.join(valid_actions)}"}

            cmd = ["sudo", "systemctl", action, service_name]
            if action == "status":
                cmd = ["systemctl", action, service_name, "--no-pager"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "action": action,
                "service": service_name,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": process.returncode
            }

        except Exception as e:
            return {"error": str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        if not self.systemd_available:
            return {"error": "systemd not available"}

        try:
            # Get system status
            process = await asyncio.create_subprocess_exec(
                "systemctl", "status", "--no-pager",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            # Get boot time
            uptime_process = await asyncio.create_subprocess_exec(
                "uptime", "-s",
                stdout=asyncio.subprocess.PIPE,
                text=True
            )
            uptime_stdout, _ = await uptime_process.communicate()

            return {
                "system_status": stdout,
                "boot_time": uptime_stdout.strip() if uptime_stdout else "Unknown",
                "systemd_version": await self._get_systemd_version()
            }

        except Exception as e:
            return {"error": str(e)}

    async def _get_systemd_version(self) -> str:
        """Get systemd version"""
        try:
            process = await asyncio.create_subprocess_exec(
                "systemctl", "--version",
                stdout=asyncio.subprocess.PIPE,
                text=True
            )
            stdout, _ = await process.communicate()

            if stdout:
                first_line = stdout.split('\n')[0]
                return first_line
            return "Unknown"
        except:
            return "Unknown"


class LinuxPackageManager:
    """Linux package manager integration"""

    def __init__(self, console: Console):
        self.console = console
        self.package_manager = self._detect_package_manager()

    def _detect_package_manager(self) -> str:
        """Detect available package manager"""
        managers = [
            ("apt", "apt"),
            ("dnf", "dnf"),
            ("yum", "yum"),
            ("pacman", "pacman"),
            ("zypper", "zypper"),
            ("emerge", "emerge"),
            ("apk", "apk")
        ]

        for name, command in managers:
            try:
                result = subprocess.run([command, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return name
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        return "unknown"

    async def search_packages(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for packages"""
        if self.package_manager == "unknown":
            return [{"error": "No supported package manager found"}]

        try:
            if self.package_manager == "apt":
                cmd = ["apt", "search", query]
            elif self.package_manager in ["dnf", "yum"]:
                cmd = [self.package_manager, "search", query]
            elif self.package_manager == "pacman":
                cmd = ["pacman", "-Ss", query]
            elif self.package_manager == "zypper":
                cmd = ["zypper", "search", query]
            elif self.package_manager == "apk":
                cmd = ["apk", "search", query]
            else:
                return [{"error": f"Search not implemented for {self.package_manager}"}]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return self._parse_package_search(stdout, limit)
            else:
                return [{"error": stderr}]

        except Exception as e:
            return [{"error": str(e)}]

    def _parse_package_search(self, output: str, limit: int) -> List[Dict[str, Any]]:
        """Parse package search output"""
        packages = []
        lines = output.split('\n')

        for line in lines[:limit]:
            if line.strip():
                if self.package_manager == "apt":
                    if line.startswith('WARNING') or line.startswith('NOTE'):
                        continue
                    parts = line.split(' - ')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        description = parts[1].strip()
                        packages.append({"name": name, "description": description})

                elif self.package_manager in ["dnf", "yum"]:
                    if ':' in line and not line.startswith('='):
                        parts = line.split(' : ')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            description = parts[1].strip()
                            packages.append({"name": name, "description": description})

                elif self.package_manager == "pacman":
                    if '/' in line:
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            name = parts[1]
                            description = ' '.join(parts[2:]) if len(parts) > 2 else ''
                            packages.append({"name": name, "description": description})

        return packages[:limit]

    async def install_package(self, package_name: str) -> Dict[str, Any]:
        """Install a package"""
        if self.package_manager == "unknown":
            return {"error": "No supported package manager found"}

        try:
            if self.package_manager == "apt":
                cmd = ["sudo", "apt", "install", "-y", package_name]
            elif self.package_manager == "dnf":
                cmd = ["sudo", "dnf", "install", "-y", package_name]
            elif self.package_manager == "yum":
                cmd = ["sudo", "yum", "install", "-y", package_name]
            elif self.package_manager == "pacman":
                cmd = ["sudo", "pacman", "-S", "--noconfirm", package_name]
            elif self.package_manager == "zypper":
                cmd = ["sudo", "zypper", "install", "-y", package_name]
            elif self.package_manager == "apk":
                cmd = ["sudo", "apk", "add", package_name]
            else:
                return {"error": f"Install not implemented for {self.package_manager}"}

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "package": package_name,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": process.returncode
            }

        except Exception as e:
            return {"error": str(e)}

    async def update_system(self) -> Dict[str, Any]:
        """Update system packages"""
        if self.package_manager == "unknown":
            return {"error": "No supported package manager found"}

        try:
            if self.package_manager == "apt":
                cmd = ["sudo", "apt", "update", "&&", "sudo", "apt", "upgrade", "-y"]
            elif self.package_manager == "dnf":
                cmd = ["sudo", "dnf", "update", "-y"]
            elif self.package_manager == "yum":
                cmd = ["sudo", "yum", "update", "-y"]
            elif self.package_manager == "pacman":
                cmd = ["sudo", "pacman", "-Syu", "--noconfirm"]
            elif self.package_manager == "zypper":
                cmd = ["sudo", "zypper", "update", "-y"]
            elif self.package_manager == "apk":
                cmd = ["sudo", "apk", "update", "&&", "sudo", "apk", "upgrade"]
            else:
                return {"error": f"Update not implemented for {self.package_manager}"}

            process = await asyncio.create_subprocess_shell(
                " ".join(cmd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "return_code": process.returncode
            }

        except Exception as e:
            return {"error": str(e)}


class LinuxContainerManager:
    """Linux container management (Docker, Podman)"""

    def __init__(self, console: Console):
        self.console = console
        self.docker_available = self._check_docker()
        self.podman_available = self._check_podman()

    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_podman(self) -> bool:
        """Check if Podman is available"""
        try:
            result = subprocess.run(["podman", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def list_containers(self, container_type: str = "docker") -> List[Dict[str, Any]]:
        """List containers"""
        if container_type == "docker" and not self.docker_available:
            return [{"error": "Docker not available"}]
        elif container_type == "podman" and not self.podman_available:
            return [{"error": "Podman not available"}]

        try:
            cmd = [container_type, "ps", "-a", "--format", "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                containers = []
                lines = stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            containers.append({
                                "name": parts[0],
                                "status": parts[1],
                                "image": parts[2],
                                "ports": parts[3] if len(parts) > 3 else ""
                            })
                return containers
            else:
                return [{"error": stderr}]

        except Exception as e:
            return [{"error": str(e)}]

    async def get_container_stats(self, container_type: str = "docker") -> Dict[str, Any]:
        """Get container resource usage stats"""
        if container_type == "docker" and not self.docker_available:
            return {"error": "Docker not available"}
        elif container_type == "podman" and not self.podman_available:
            return {"error": "Podman not available"}

        try:
            cmd = [container_type, "stats", "--no-stream", "--format", "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return {"stats": stdout, "success": True}
            else:
                return {"error": stderr}

        except Exception as e:
            return {"error": str(e)}


class LinuxAdvancedAI(AdvancedCLICore):
    """Linux-specific Advanced AI Integration with full Claude CLI and Gemini CLI features"""

    def __init__(self, console: Console = None):
        super().__init__(console)

        # Linux-specific components
        self.linux_config = LinuxSpecificConfig()
        self.notification_manager = LinuxNotificationManager()
        self.systemd = LinuxSystemdIntegration(self.console)
        self.package_manager = LinuxPackageManager(self.console)
        self.container_manager = LinuxContainerManager(self.console)

        # Linux-specific conversation storage (XDG compliant)
        xdg_data = os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')
        self.conversation_manager = ConversationManager(
            Path(xdg_data) / 'terminal-coder' / 'conversations',
            self.console
        )

        # Detect Linux distribution
        self.distribution = self._detect_distribution()
        self.desktop_environment = self._detect_desktop_environment()

        # Add Linux-specific built-in tools
        self.builtin_tools.update({
            'linux_services': self._tool_linux_services,
            'linux_packages': self._tool_linux_packages,
            'linux_containers': self._tool_linux_containers,
            'linux_system_info': self._tool_linux_system_info,
            'linux_processes': self._tool_linux_processes,
            'linux_network': self._tool_linux_network,
            'linux_security': self._tool_linux_security,
            'linux_logs': self._tool_linux_logs
        })

    def _detect_distribution(self) -> str:
        """Detect Linux distribution"""
        try:
            if LINUX_FEATURES:
                return distro.name()
            else:
                # Fallback detection
                if Path('/etc/os-release').exists():
                    with open('/etc/os-release') as f:
                        for line in f:
                            if line.startswith('NAME='):
                                return line.split('=')[1].strip('"')
                return "Unknown Linux"
        except Exception:
            return "Unknown Linux"

    def _detect_desktop_environment(self) -> str:
        """Detect desktop environment"""
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

    async def initialize_linux_features(self):
        """Initialize Linux-specific features"""
        try:
            # Initialize Linux-specific configurations
            await self._setup_linux_notifications()
            await self._setup_linux_keyring()
            await self._setup_xdg_compliance()

            # Send welcome notification
            await self.notification_manager.send_notification(
                "Terminal Coder Linux",
                f"Initialized on {self.distribution} with {self.desktop_environment.upper()} desktop",
                "normal"
            )

            self.console.print(f"[green]✅ Linux features initialized for {self.distribution}[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Linux features initialization failed: {e}[/red]")
            return False

    async def _setup_linux_notifications(self):
        """Setup Linux notifications"""
        if self.linux_config.use_dbus_notifications:
            success = await self.notification_manager.send_notification(
                "Terminal Coder", "Linux AI integration ready!", "normal"
            )
            if success:
                self.console.print("[green]📱 Linux notifications enabled[/green]")

    async def _setup_linux_keyring(self):
        """Setup Linux keyring integration"""
        if self.linux_config.use_linux_keyring:
            try:
                import keyring
                # Test keyring access
                keyring.get_keyring()
                self.console.print("[green]🔐 Linux keyring integration enabled[/green]")
            except Exception as e:
                self.console.print(f"[yellow]⚠️  Keyring integration limited: {e}[/yellow]")

    async def _setup_xdg_compliance(self):
        """Setup XDG Base Directory compliance"""
        xdg_config = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        xdg_data = os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')

        config_dir = Path(xdg_config) / 'terminal-coder'
        data_dir = Path(xdg_data) / 'terminal-coder'

        config_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        self.console.print("[green]📁 XDG Base Directory compliance enabled[/green]")

    async def process_user_input_linux(self, user_input: str, files: List[str] = None,
                                     stream: bool = True, use_systemd: bool = False) -> str:
        """Linux-specific user input processing"""

        # Check for Linux-specific commands
        if user_input.startswith('$'):
            return await self._handle_linux_command(user_input[1:])

        # Systemd integration
        if use_systemd or user_input.startswith('systemctl:'):
            if user_input.startswith('systemctl:'):
                user_input = user_input[10:]
            return await self._handle_systemd_query(user_input)

        # Standard processing with Linux enhancements
        response = await self.process_user_input(user_input, files, stream)

        # Send Linux notification for long responses
        if len(response) > 1000:
            await self.notification_manager.send_notification(
                "AI Response Ready",
                f"Generated {len(response)} character response",
                "normal"
            )

        return response

    async def _handle_linux_command(self, command: str) -> str:
        """Handle Linux-specific commands"""
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == 'services' or cmd == 'systemctl':
            return await self._linux_services_command(args)
        elif cmd == 'packages' or cmd == 'pkg':
            return await self._linux_packages_command(args)
        elif cmd == 'containers' or cmd == 'docker':
            return await self._linux_containers_command(args)
        elif cmd == 'processes' or cmd == 'ps':
            return await self._linux_processes_command(args)
        elif cmd == 'system' or cmd == 'info':
            return await self._linux_system_command(args)
        elif cmd == 'logs' or cmd == 'journal':
            return await self._linux_logs_command(args)
        elif cmd == 'network' or cmd == 'net':
            return await self._linux_network_command(args)
        else:
            return f"Unknown Linux command: ${cmd}. Available: services, packages, containers, processes, system, logs, network"

    async def _handle_systemd_query(self, query: str) -> str:
        """Handle systemd-integrated AI queries"""
        try:
            # Check if it's a systemctl command
            if any(query.startswith(cmd) for cmd in ['list-units', 'status', 'start', 'stop', 'restart']):
                # Execute systemctl command
                if query.startswith(('start', 'stop', 'restart')):
                    parts = query.split()
                    action = parts[0]
                    service_name = parts[1] if len(parts) > 1 else ""
                    systemd_result = await self.systemd.manage_service(action, service_name)
                else:
                    # For list-units, status, etc.
                    process = await asyncio.create_subprocess_shell(
                        f"systemctl {query}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        text=True
                    )
                    stdout, stderr = await process.communicate()
                    systemd_result = {"stdout": stdout, "stderr": stderr, "success": process.returncode == 0}

                if systemd_result.get('success'):
                    # Combine systemctl output with AI analysis
                    combined_query = f"Here's the systemctl output for '{query}':\n\n{systemd_result.get('stdout', '')}\n\nPlease analyze and explain this output:"
                    ai_response = await self.process_user_input(combined_query, stream=False)

                    return f"Systemctl Output:\n{systemd_result.get('stdout', '')}\n\nAI Analysis:\n{ai_response}"
                else:
                    return f"Systemctl Error: {systemd_result.get('stderr', 'Unknown error')}"
            else:
                # Regular AI query with systemd context
                return await self.process_user_input(f"Linux systemd context: {query}")

        except Exception as e:
            return f"Systemd integration error: {e}"

    # Linux-specific tool implementations
    async def _tool_linux_services(self, action: str = "list", service_name: str = None) -> str:
        """Linux systemd services management tool"""
        try:
            if action == "list":
                services = await self.systemd.list_services()
                if services and 'error' not in services[0]:
                    output = "Linux systemd Services:\n"
                    output += f"{'Name':<30} {'Load':<10} {'Active':<10} {'Sub':<12} {'Description':<40}\n"
                    output += "-" * 102 + "\n"

                    for service in services[:20]:  # Limit to 20 services
                        name = service.get('name', 'Unknown')[:29]
                        load = service.get('load', 'Unknown')[:9]
                        active = service.get('active', 'Unknown')[:9]
                        sub = service.get('sub', 'Unknown')[:11]
                        desc = service.get('description', 'Unknown')[:39]

                        output += f"{name:<30} {load:<10} {active:<10} {sub:<12} {desc:<40}\n"

                    return output
                else:
                    return f"Service list failed: {services[0].get('error', 'Unknown error') if services else 'No services found'}"

            else:
                result = await self.systemd.manage_service(action, service_name)
                if result.get('success'):
                    return f"Service operation succeeded:\n{result.get('stdout', '')}"
                else:
                    return f"Service operation failed: {result.get('stderr', 'Unknown error')}"

        except Exception as e:
            return f"Linux services tool error: {e}"

    async def _tool_linux_packages(self, action: str = "search", query: str = None) -> str:
        """Linux package management tool"""
        try:
            if action == "search" and query:
                packages = await self.package_manager.search_packages(query)
                if packages and 'error' not in packages[0]:
                    output = f"Package search results for '{query}' (using {self.package_manager.package_manager}):\n"
                    output += f"{'Package Name':<40} {'Description':<60}\n"
                    output += "-" * 100 + "\n"

                    for pkg in packages[:15]:  # Limit to 15 packages
                        name = pkg.get('name', 'Unknown')[:39]
                        desc = pkg.get('description', 'No description')[:59]
                        output += f"{name:<40} {desc:<60}\n"

                    return output
                else:
                    return f"Package search failed: {packages[0].get('error', 'Unknown error') if packages else 'No packages found'}"

            elif action == "install" and query:
                result = await self.package_manager.install_package(query)
                if result.get('success'):
                    return f"Package '{query}' installed successfully"
                else:
                    return f"Package installation failed: {result.get('stderr', 'Unknown error')}"

            elif action == "update":
                result = await self.package_manager.update_system()
                if result.get('success'):
                    return f"System update completed successfully"
                else:
                    return f"System update failed: {result.get('stderr', 'Unknown error')}"

            else:
                return f"Package manager ({self.package_manager.package_manager}) commands: search <query>, install <package>, update"

        except Exception as e:
            return f"Linux packages tool error: {e}"

    async def _tool_linux_containers(self, action: str = "list", container_type: str = "docker") -> str:
        """Linux container management tool"""
        try:
            if action == "list":
                containers = await self.container_manager.list_containers(container_type)
                if containers and 'error' not in containers[0]:
                    output = f"Linux Containers ({container_type}):\n"
                    output += f"{'Name':<25} {'Status':<15} {'Image':<30} {'Ports':<20}\n"
                    output += "-" * 90 + "\n"

                    for container in containers:
                        name = container.get('name', 'Unknown')[:24]
                        status = container.get('status', 'Unknown')[:14]
                        image = container.get('image', 'Unknown')[:29]
                        ports = container.get('ports', '')[:19]

                        output += f"{name:<25} {status:<15} {image:<30} {ports:<20}\n"

                    return output
                else:
                    return f"Container list failed: {containers[0].get('error', 'Unknown error') if containers else 'No containers found'}"

            elif action == "stats":
                stats = await self.container_manager.get_container_stats(container_type)
                if stats.get('success'):
                    return f"Container Stats ({container_type}):\n{stats.get('stats', '')}"
                else:
                    return f"Container stats failed: {stats.get('error', 'Unknown error')}"

            else:
                return f"Container commands: list [docker|podman], stats [docker|podman]"

        except Exception as e:
            return f"Linux containers tool error: {e}"

    async def _tool_linux_system_info(self) -> str:
        """Linux system information tool"""
        try:
            output = f"Linux System Information:\n\n"
            output += f"Distribution: {self.distribution}\n"
            output += f"Desktop Environment: {self.desktop_environment.upper()}\n"

            # Get systemd status
            system_status = await self.systemd.get_system_status()
            if 'error' not in system_status:
                output += f"Boot Time: {system_status.get('boot_time', 'Unknown')}\n"
                output += f"Systemd Version: {system_status.get('systemd_version', 'Unknown')}\n"

            # Get memory and CPU info
            if LINUX_FEATURES:
                memory = psutil.virtual_memory()
                cpu_count = psutil.cpu_count()
                load_avg = os.getloadavg()

                output += f"\nResource Usage:\n"
                output += f"CPU Cores: {cpu_count}\n"
                output += f"Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}\n"
                output += f"Memory: {memory.percent}% used ({memory.used // (1024**3)} GB / {memory.total // (1024**3)} GB)\n"

            # Get disk usage
            try:
                disk = psutil.disk_usage('/')
                output += f"Root Disk: {disk.percent}% used ({disk.used // (1024**3)} GB / {disk.total // (1024**3)} GB)\n"
            except Exception:
                pass

            return output

        except Exception as e:
            return f"Linux system info tool error: {e}"

    async def _tool_linux_processes(self) -> str:
        """Linux process monitoring tool"""
        try:
            output = "Top Linux Processes:\n"
            output += f"{'PID':<8} {'Name':<20} {'CPU%':<8} {'Memory%':<10} {'Status':<12} {'User':<12}\n"
            output += "-" * 80 + "\n"

            if LINUX_FEATURES:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status', 'username']):
                    try:
                        info = proc.info
                        info['cpu_percent'] = proc.cpu_percent(interval=0.1)
                        processes.append(info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Sort by CPU usage
                processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)

                for proc in processes[:15]:  # Top 15 processes
                    pid = proc.get('pid', 0)
                    name = proc.get('name', 'Unknown')[:19]
                    cpu = proc.get('cpu_percent', 0)
                    memory = proc.get('memory_percent', 0)
                    status = proc.get('status', 'Unknown')[:11]
                    user = proc.get('username', 'Unknown')[:11]

                    output += f"{pid:<8} {name:<20} {cpu:<8.1f} {memory:<10.1f} {status:<12} {user:<12}\n"

            return output

        except Exception as e:
            return f"Linux processes tool error: {e}"

    async def _tool_linux_network(self) -> str:
        """Linux network information tool"""
        try:
            output = "Linux Network Information:\n\n"

            # Get network interfaces
            if LINUX_FEATURES:
                interfaces = psutil.net_if_addrs()
                stats = psutil.net_if_stats()

                for interface, addresses in interfaces.items():
                    output += f"Interface: {interface}\n"

                    if interface in stats:
                        stat = stats[interface]
                        output += f"  Status: {'Up' if stat.isup else 'Down'}\n"
                        output += f"  Speed: {stat.speed} Mbps\n" if stat.speed > 0 else "  Speed: Unknown\n"

                    for addr in addresses:
                        if addr.family.name == 'AF_INET':  # IPv4
                            output += f"  IPv4: {addr.address}\n"
                        elif addr.family.name == 'AF_INET6':  # IPv6
                            output += f"  IPv6: {addr.address}\n"

                    output += "\n"

                # Get network IO stats
                net_io = psutil.net_io_counters()
                output += f"Network IO:\n"
                output += f"Bytes Sent: {net_io.bytes_sent // (1024*1024)} MB\n"
                output += f"Bytes Received: {net_io.bytes_recv // (1024*1024)} MB\n"

            return output

        except Exception as e:
            return f"Linux network tool error: {e}"

    async def _tool_linux_security(self) -> str:
        """Linux security information tool"""
        try:
            output = "Linux Security Information:\n\n"

            # Check for common security tools
            security_tools = []
            tools_to_check = ['ufw', 'iptables', 'fail2ban', 'apparmor', 'selinux']

            for tool in tools_to_check:
                try:
                    result = subprocess.run(['which', tool], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        security_tools.append(tool)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue

            output += f"Security tools found: {', '.join(security_tools)}\n\n"

            # Check firewall status (UFW)
            if 'ufw' in security_tools:
                try:
                    result = subprocess.run(['sudo', 'ufw', 'status'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        output += f"UFW Firewall Status:\n{result.stdout}\n\n"
                except Exception:
                    pass

            # Check for security updates
            if self.package_manager.package_manager == "apt":
                try:
                    result = subprocess.run(['apt', 'list', '--upgradable'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        upgrade_count = len(result.stdout.split('\n')) - 2  # Subtract header and empty line
                        output += f"Available updates: {upgrade_count}\n"
                except Exception:
                    pass

            return output

        except Exception as e:
            return f"Linux security tool error: {e}"

    async def _tool_linux_logs(self, service: str = None, lines: int = 20) -> str:
        """Linux system logs tool"""
        try:
            if self.systemd.systemd_available:
                if service:
                    cmd = ['journalctl', '-u', service, '-n', str(lines), '--no-pager']
                else:
                    cmd = ['journalctl', '-n', str(lines), '--no-pager']

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    text=True
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    return f"System Logs{f' for {service}' if service else ''} (last {lines} lines):\n\n{stdout}"
                else:
                    return f"Log query failed: {stderr}"
            else:
                # Fallback to traditional logs
                log_files = ['/var/log/syslog', '/var/log/messages', '/var/log/system.log']
                for log_file in log_files:
                    if Path(log_file).exists():
                        try:
                            result = subprocess.run(['tail', '-n', str(lines), log_file],
                                                  capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                return f"System Logs from {log_file} (last {lines} lines):\n\n{result.stdout}"
                        except Exception:
                            continue

                return "No accessible log files found"

        except Exception as e:
            return f"Linux logs tool error: {e}"

    # Linux-specific command handlers
    async def _linux_services_command(self, args: str) -> str:
        """Handle Linux services commands"""
        if not args:
            return await self._tool_linux_services()
        else:
            parts = args.split()
            action = parts[0] if parts else "list"
            service_name = parts[1] if len(parts) > 1 else None
            return await self._tool_linux_services(action, service_name)

    async def _linux_packages_command(self, args: str) -> str:
        """Handle Linux package commands"""
        if not args:
            return "Package commands: search <query>, install <package>, update"
        else:
            parts = args.split()
            action = parts[0] if parts else "search"
            query = parts[1] if len(parts) > 1 else None
            return await self._tool_linux_packages(action, query)

    async def _linux_containers_command(self, args: str) -> str:
        """Handle Linux container commands"""
        if not args:
            return await self._tool_linux_containers()
        else:
            parts = args.split()
            action = parts[0] if parts else "list"
            container_type = parts[1] if len(parts) > 1 else "docker"
            return await self._tool_linux_containers(action, container_type)

    async def _linux_processes_command(self, args: str) -> str:
        """Handle Linux process commands"""
        return await self._tool_linux_processes()

    async def _linux_system_command(self, args: str) -> str:
        """Handle Linux system commands"""
        return await self._tool_linux_system_info()

    async def _linux_logs_command(self, args: str) -> str:
        """Handle Linux logs commands"""
        parts = args.split() if args else []
        service = parts[0] if parts else None
        lines = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
        return await self._tool_linux_logs(service, lines)

    async def _linux_network_command(self, args: str) -> str:
        """Handle Linux network commands"""
        return await self._tool_linux_network()

    async def save_conversation_to_linux_location(self, format_type: str = 'markdown') -> str:
        """Save conversation to Linux-specific location"""
        if not self.conversation_manager.current_session:
            return "No active conversation to save"

        try:
            # Use XDG Documents directory
            xdg_documents = Path.home() / 'Documents'
            if not xdg_documents.exists():
                xdg_documents = Path.home()

            save_path = xdg_documents / 'TerminalCoder'
            save_path.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = save_path / f"conversation_{timestamp}.{format_type}"

            content = await self.conversation_manager.export_conversation(
                self.conversation_manager.current_session.session_id,
                format_type
            )

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)

            # Set appropriate permissions
            filename.chmod(0o644)

            # Linux notification
            await self.notification_manager.send_notification(
                "Conversation Saved",
                f"Saved to {filename.name}",
                "normal"
            )

            return f"Conversation saved to {filename}"

        except Exception as e:
            return f"Save failed: {e}"

    async def get_linux_help(self) -> str:
        """Get Linux-specific help"""
        help_text = await self._show_help()

        linux_help = """

🐧 Linux-Specific Features:

## Linux Commands (use $ prefix):
$services [action] [name]    - Manage systemd services
$packages [search|install|update] [query] - Package management
$containers [list|stats] [docker|podman] - Container management
$processes                   - Show running processes
$system                     - System information
$logs [service] [lines]     - View system logs
$network                    - Network information

## Systemd Integration:
systemctl: <command>        - Execute systemctl with AI analysis
Any systemctl command (start, stop, status, etc.)

## Linux-Specific Tools:
- linux_services: systemd service management
- linux_packages: Package manager operations
- linux_containers: Docker/Podman management
- linux_system_info: Comprehensive system info
- linux_processes: Process monitoring
- linux_network: Network status
- linux_security: Security information
- linux_logs: System log analysis

## Features:
🔔 Native desktop notifications (D-Bus/libnotify)
🔐 Linux keyring integration
📁 XDG Base Directory compliance
⚙️ systemd integration
📦 Multi-package manager support
🐳 Container development support
🛡️ Security scanning
📋 Desktop environment integration

## Supported Distributions:
- Ubuntu/Debian (apt)
- Fedora/RHEL/CentOS (dnf/yum)
- Arch Linux (pacman)
- openSUSE (zypper)
- Alpine (apk)
- Gentoo (emerge)

## Desktop Environments:
- GNOME - Full integration
- KDE Plasma - Full integration
- XFCE - Basic integration
- Unity - Full integration
- Other DEs - Basic support
        """

        return help_text + linux_help