"""
Linux System Manager - Advanced Linux Integration
Deep Linux system integration with distribution detection, package management, and system optimization
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Final

import distro
import psutil
import sh
from rich.console import Console
from rich.table import Table

# Linux-specific imports
try:
    import dbus
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

try:
    import pyinotify
    INOTIFY_AVAILABLE = True
except ImportError:
    INOTIFY_AVAILABLE = False


class LinuxDistribution(Enum):
    """Supported Linux distributions"""
    UBUNTU = auto()
    DEBIAN = auto()
    FEDORA = auto()
    CENTOS = auto()
    RHEL = auto()
    ARCH = auto()
    MANJARO = auto()
    OPENSUSE = auto()
    GENTOO = auto()
    ALPINE = auto()
    UNKNOWN = auto()


class PackageManager(Enum):
    """Linux package managers"""
    APT = "apt"           # Debian/Ubuntu
    DNF = "dnf"           # Fedora/RHEL 8+
    YUM = "yum"           # CentOS/RHEL 7
    PACMAN = "pacman"     # Arch/Manjaro
    ZYPPER = "zypper"     # OpenSUSE
    EMERGE = "emerge"     # Gentoo
    APK = "apk"           # Alpine
    SNAP = "snap"         # Universal
    FLATPAK = "flatpak"   # Universal


class SystemdService(Enum):
    """Common systemd services"""
    DOCKER = "docker"
    NGINX = "nginx"
    APACHE = "apache2"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    REDIS = "redis"


@dataclass(slots=True)
class LinuxSystemInfo:
    """Linux system information"""
    distribution: LinuxDistribution
    distro_name: str
    distro_version: str
    kernel_version: str
    desktop_environment: str | None
    package_manager: PackageManager
    architecture: str
    cpu_cores: int
    memory_gb: float
    disk_usage: dict[str, Any] = field(default_factory=dict)
    systemd_available: bool = False
    docker_available: bool = False
    x11_available: bool = False
    wayland_available: bool = False


class LinuxSystemManager:
    """Advanced Linux system integration and management"""

    # Distribution detection mappings
    DISTRO_MAP: Final[dict[str, LinuxDistribution]] = {
        "ubuntu": LinuxDistribution.UBUNTU,
        "debian": LinuxDistribution.DEBIAN,
        "fedora": LinuxDistribution.FEDORA,
        "centos": LinuxDistribution.CENTOS,
        "rhel": LinuxDistribution.RHEL,
        "red hat": LinuxDistribution.RHEL,
        "arch": LinuxDistribution.ARCH,
        "manjaro": LinuxDistribution.MANJARO,
        "opensuse": LinuxDistribution.OPENSUSE,
        "suse": LinuxDistribution.OPENSUSE,
        "gentoo": LinuxDistribution.GENTOO,
        "alpine": LinuxDistribution.ALPINE,
    }

    # Package manager mappings
    PKG_MANAGER_MAP: Final[dict[LinuxDistribution, PackageManager]] = {
        LinuxDistribution.UBUNTU: PackageManager.APT,
        LinuxDistribution.DEBIAN: PackageManager.APT,
        LinuxDistribution.FEDORA: PackageManager.DNF,
        LinuxDistribution.CENTOS: PackageManager.YUM,
        LinuxDistribution.RHEL: PackageManager.YUM,
        LinuxDistribution.ARCH: PackageManager.PACMAN,
        LinuxDistribution.MANJARO: PackageManager.PACMAN,
        LinuxDistribution.OPENSUSE: PackageManager.ZYPPER,
        LinuxDistribution.GENTOO: PackageManager.EMERGE,
        LinuxDistribution.ALPINE: PackageManager.APK,
    }

    def __init__(self) -> None:
        self.console = Console()
        self._system_info: LinuxSystemInfo | None = None

        # Verify we're running on Linux
        if platform.system() != "Linux":
            raise RuntimeError("LinuxSystemManager can only run on Linux systems")

    @property
    def system_info(self) -> LinuxSystemInfo:
        """Get cached system information"""
        if self._system_info is None:
            self._system_info = self.detect_system_info()
        return self._system_info

    def detect_system_info(self) -> LinuxSystemInfo:
        """Detect comprehensive Linux system information"""
        # Detect distribution
        distro_name = distro.name().lower()
        distro_version = distro.version()

        distribution = LinuxDistribution.UNKNOWN
        for name, dist_enum in self.DISTRO_MAP.items():
            if name in distro_name:
                distribution = dist_enum
                break

        # Get package manager
        package_manager = self.PKG_MANAGER_MAP.get(distribution, PackageManager.APT)

        # Detect desktop environment
        desktop_env = (
            os.environ.get("XDG_CURRENT_DESKTOP") or
            os.environ.get("DESKTOP_SESSION") or
            os.environ.get("GDMSESSION")
        )

        # System capabilities
        systemd_available = Path("/run/systemd/system").exists()
        docker_available = self._command_exists("docker")
        x11_available = os.environ.get("DISPLAY") is not None
        wayland_available = os.environ.get("WAYLAND_DISPLAY") is not None

        # Hardware info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count()

        # Disk usage
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = {
                    "total": usage.total / (1024**3),  # GB
                    "used": usage.used / (1024**3),
                    "free": usage.free / (1024**3),
                    "percent": (usage.used / usage.total) * 100
                }
            except PermissionError:
                continue

        return LinuxSystemInfo(
            distribution=distribution,
            distro_name=distro.name(),
            distro_version=distro_version,
            kernel_version=platform.release(),
            desktop_environment=desktop_env,
            package_manager=package_manager,
            architecture=platform.machine(),
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_usage=disk_usage,
            systemd_available=systemd_available,
            docker_available=docker_available,
            x11_available=x11_available,
            wayland_available=wayland_available
        )

    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            subprocess.run(
                ["which", command],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def install_system_dependencies(self) -> bool:
        """Install system dependencies based on distribution"""
        info = self.system_info

        self.console.print(f"[cyan]Installing system dependencies for {info.distro_name}...[/cyan]")

        try:
            match info.package_manager:
                case PackageManager.APT:
                    # Ubuntu/Debian
                    commands = [
                        "sudo apt update",
                        "sudo apt install -y python3-dev python3-pip git curl wget",
                        "sudo apt install -y build-essential libffi-dev libssl-dev",
                        "sudo apt install -y libdbus-1-dev libdbus-glib-1-dev",
                        "sudo apt install -y python3-tk python3-venv"
                    ]

                case PackageManager.DNF:
                    # Fedora
                    commands = [
                        "sudo dnf update -y",
                        "sudo dnf install -y python3-devel python3-pip git curl wget",
                        "sudo dnf install -y gcc gcc-c++ make libffi-devel openssl-devel",
                        "sudo dnf install -y dbus-devel dbus-glib-devel",
                        "sudo dnf install -y python3-tkinter"
                    ]

                case PackageManager.PACMAN:
                    # Arch/Manjaro
                    commands = [
                        "sudo pacman -Syu --noconfirm",
                        "sudo pacman -S --noconfirm python python-pip git curl wget",
                        "sudo pacman -S --noconfirm base-devel libffi openssl",
                        "sudo pacman -S --noconfirm dbus dbus-glib",
                        "sudo pacman -S --noconfirm tk"
                    ]

                case PackageManager.ZYPPER:
                    # OpenSUSE
                    commands = [
                        "sudo zypper refresh",
                        "sudo zypper install -y python3-devel python3-pip git curl wget",
                        "sudo zypper install -y gcc gcc-c++ make libffi-devel libopenssl-devel",
                        "sudo zypper install -y dbus-1-devel dbus-1-glib-devel",
                        "sudo zypper install -y python3-tk"
                    ]

                case _:
                    self.console.print(f"[yellow]Package manager {info.package_manager} not fully supported yet[/yellow]")
                    return False

            # Execute installation commands
            for cmd in commands:
                self.console.print(f"[dim]Running: {cmd}[/dim]")
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    self.console.print(f"[red]Command failed: {cmd}[/red]")
                    self.console.print(f"[red]Error: {result.stderr}[/red]")
                    return False

            self.console.print("[green]✅ System dependencies installed successfully![/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Failed to install dependencies: {e}[/red]")
            return False

    def setup_shell_integration(self) -> bool:
        """Setup shell completion and integration"""
        self.console.print("[cyan]Setting up shell integration...[/cyan]")

        shell = os.environ.get("SHELL", "/bin/bash")
        shell_name = Path(shell).name

        try:
            match shell_name:
                case "bash":
                    self._setup_bash_completion()
                case "zsh":
                    self._setup_zsh_completion()
                case "fish":
                    self._setup_fish_completion()
                case _:
                    self.console.print(f"[yellow]Shell {shell_name} not fully supported[/yellow]")
                    return False

            self.console.print("[green]✅ Shell integration setup complete![/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Shell integration setup failed: {e}[/red]")
            return False

    def _setup_bash_completion(self) -> None:
        """Setup bash completion"""
        bashrc_path = Path.home() / ".bashrc"
        completion_line = 'eval "$(_TERMINAL_CODER_COMPLETE=bash_source terminal-coder)"'

        if bashrc_path.exists():
            content = bashrc_path.read_text()
            if completion_line not in content:
                with bashrc_path.open("a") as f:
                    f.write(f"\n# Terminal Coder completion\n{completion_line}\n")

    def _setup_zsh_completion(self) -> None:
        """Setup zsh completion"""
        zshrc_path = Path.home() / ".zshrc"
        completion_line = 'eval "$(_TERMINAL_CODER_COMPLETE=zsh_source terminal-coder)"'

        if zshrc_path.exists():
            content = zshrc_path.read_text()
            if completion_line not in content:
                with zshrc_path.open("a") as f:
                    f.write(f"\n# Terminal Coder completion\n{completion_line}\n")

    def _setup_fish_completion(self) -> None:
        """Setup fish completion"""
        fish_config_dir = Path.home() / ".config" / "fish" / "completions"
        fish_config_dir.mkdir(parents=True, exist_ok=True)

        completion_file = fish_config_dir / "terminal-coder.fish"
        if not completion_file.exists():
            completion_content = '''
complete -c terminal-coder -f
complete -c terminal-coder -s h -l help -d "Show help"
complete -c terminal-coder -l version -d "Show version"
complete -c terminal-coder -l tui -d "Launch TUI mode"
'''
            completion_file.write_text(completion_content)

    def optimize_for_development(self) -> bool:
        """Apply Linux-specific optimizations for development"""
        self.console.print("[cyan]Applying Linux development optimizations...[/cyan]")

        try:
            # Increase file descriptor limits
            self._increase_file_limits()

            # Optimize kernel parameters for development
            self._optimize_kernel_params()

            # Setup development tools
            self._setup_dev_tools()

            self.console.print("[green]✅ Linux optimizations applied![/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Optimization failed: {e}[/red]")
            return False

    def _increase_file_limits(self) -> None:
        """Increase file descriptor limits for development"""
        limits_conf = Path("/etc/security/limits.conf")

        if limits_conf.exists() and os.geteuid() == 0:  # Running as root
            content = limits_conf.read_text()

            limits_lines = [
                "* soft nofile 65536",
                "* hard nofile 65536",
                "* soft nproc 65536",
                "* hard nproc 65536"
            ]

            for line in limits_lines:
                if line not in content:
                    with limits_conf.open("a") as f:
                        f.write(f"{line}\n")

    def _optimize_kernel_params(self) -> None:
        """Optimize kernel parameters for development workloads"""
        sysctl_conf = Path("/etc/sysctl.conf")

        if sysctl_conf.exists() and os.geteuid() == 0:
            optimizations = [
                "fs.inotify.max_user_watches=524288",
                "vm.swappiness=10",
                "kernel.shmmax=268435456",
                "net.core.rmem_default=262144",
                "net.core.wmem_default=262144"
            ]

            content = sysctl_conf.read_text()

            for param in optimizations:
                if param.split("=")[0] not in content:
                    with sysctl_conf.open("a") as f:
                        f.write(f"{param}\n")

    def _setup_dev_tools(self) -> None:
        """Setup additional development tools"""
        info = self.system_info

        # Install common development packages
        dev_packages = {
            PackageManager.APT: [
                "htop", "tree", "jq", "curl", "wget", "unzip",
                "tmux", "screen", "vim", "nano", "git-flow"
            ],
            PackageManager.DNF: [
                "htop", "tree", "jq", "curl", "wget", "unzip",
                "tmux", "screen", "vim", "nano", "git-flow"
            ],
            PackageManager.PACMAN: [
                "htop", "tree", "jq", "curl", "wget", "unzip",
                "tmux", "screen", "vim", "nano", "git-flow"
            ]
        }

        packages = dev_packages.get(info.package_manager, [])
        if packages:
            cmd_map = {
                PackageManager.APT: f"sudo apt install -y {' '.join(packages)}",
                PackageManager.DNF: f"sudo dnf install -y {' '.join(packages)}",
                PackageManager.PACMAN: f"sudo pacman -S --noconfirm {' '.join(packages)}"
            }

            cmd = cmd_map.get(info.package_manager)
            if cmd:
                subprocess.run(cmd.split(), capture_output=True)

    def display_system_info(self) -> None:
        """Display comprehensive Linux system information"""
        info = self.system_info

        # Main system info table
        table = Table(title="🐧 Linux System Information")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        table.add_row("Distribution", f"{info.distro_name} {info.distro_version}")
        table.add_row("Kernel", info.kernel_version)
        table.add_row("Architecture", info.architecture)
        table.add_row("Desktop Environment", info.desktop_environment or "None (CLI)")
        table.add_row("Package Manager", info.package_manager.value)
        table.add_row("CPU Cores", str(info.cpu_cores))
        table.add_row("Memory", f"{info.memory_gb:.1f} GB")

        # Capabilities
        capabilities = []
        if info.systemd_available:
            capabilities.append("systemd")
        if info.docker_available:
            capabilities.append("Docker")
        if info.x11_available:
            capabilities.append("X11")
        if info.wayland_available:
            capabilities.append("Wayland")
        if DBUS_AVAILABLE:
            capabilities.append("D-Bus")
        if INOTIFY_AVAILABLE:
            capabilities.append("inotify")

        table.add_row("Capabilities", ", ".join(capabilities))

        self.console.print(table)

        # Disk usage table
        if info.disk_usage:
            disk_table = Table(title="💾 Disk Usage")
            disk_table.add_column("Mount Point", style="cyan")
            disk_table.add_column("Total", style="blue")
            disk_table.add_column("Used", style="red")
            disk_table.add_column("Free", style="green")
            disk_table.add_column("Usage %", style="yellow")

            for mount_point, usage in info.disk_usage.items():
                disk_table.add_row(
                    mount_point,
                    f"{usage['total']:.1f} GB",
                    f"{usage['used']:.1f} GB",
                    f"{usage['free']:.1f} GB",
                    f"{usage['percent']:.1f}%"
                )

            self.console.print(disk_table)

    def check_linux_compatibility(self) -> bool:
        """Check Terminal Coder compatibility with current Linux system"""
        info = self.system_info
        issues = []

        # Check Python version
        if sys.version_info < (3, 10):
            issues.append("Python 3.10+ required")

        # Check essential commands
        essential_commands = ["git", "curl", "wget"]
        for cmd in essential_commands:
            if not self._command_exists(cmd):
                issues.append(f"Missing command: {cmd}")

        # Check development libraries
        if info.package_manager == PackageManager.APT:
            # Check for common dev packages
            dev_check = subprocess.run(
                ["dpkg", "-l", "python3-dev", "build-essential"],
                capture_output=True
            )
            if dev_check.returncode != 0:
                issues.append("Development packages not installed")

        # Display results
        if issues:
            self.console.print("[red]❌ Compatibility Issues Found:[/red]")
            for issue in issues:
                self.console.print(f"  • {issue}")
            return False
        else:
            self.console.print("[green]✅ System is fully compatible with Terminal Coder![/green]")
            return True


# Utility functions
def get_linux_system_manager() -> LinuxSystemManager:
    """Get Linux system manager instance"""
    return LinuxSystemManager()


def is_linux_compatible() -> bool:
    """Quick check if running on compatible Linux system"""
    try:
        if platform.system() != "Linux":
            return False

        manager = LinuxSystemManager()
        return manager.check_linux_compatibility()
    except Exception:
        return False