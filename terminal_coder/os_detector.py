#!/usr/bin/env python3
"""
OS Detection and Platform Router for Terminal Coder
Automatically detects the operating system and routes to appropriate implementation
"""

import os
import sys
import platform
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum


class SupportedOS(Enum):
    """Supported operating systems"""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class OSInfo:
    """Operating system information"""
    name: SupportedOS
    version: str
    architecture: str
    platform_name: str
    is_64bit: bool
    python_version: str
    distribution: Optional[str] = None  # For Linux distributions


class OSDetector:
    """Advanced OS detection and platform routing"""

    def __init__(self):
        self.os_info = self._detect_os()
        self.platform_module = None
        self._load_platform_module()

    def _detect_os(self) -> OSInfo:
        """Detect the current operating system with detailed information"""
        system = platform.system().lower()
        version = platform.version()
        architecture = platform.machine()
        platform_name = platform.platform()
        is_64bit = platform.architecture()[0] == '64bit'
        python_version = platform.python_version()
        distribution = None

        # Determine OS type
        if system == "windows":
            os_type = SupportedOS.WINDOWS
        elif system == "linux":
            os_type = SupportedOS.LINUX
            # Detect Linux distribution
            distribution = self._detect_linux_distribution()
        elif system == "darwin":
            os_type = SupportedOS.MACOS
        else:
            os_type = SupportedOS.UNKNOWN

        return OSInfo(
            name=os_type,
            version=version,
            architecture=architecture,
            platform_name=platform_name,
            is_64bit=is_64bit,
            python_version=python_version,
            distribution=distribution
        )

    def _detect_linux_distribution(self) -> Optional[str]:
        """Detect Linux distribution"""
        try:
            import distro
            return distro.name()
        except ImportError:
            # Fallback methods
            try:
                # Try reading /etc/os-release
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('NAME='):
                            return line.split('=')[1].strip().strip('"')
            except (FileNotFoundError, IOError):
                pass

            # Try platform.linux_distribution() (deprecated but might work)
            try:
                return platform.linux_distribution()[0]
            except AttributeError:
                pass

            return "Unknown Linux"

    def _load_platform_module(self):
        """Load the appropriate platform-specific module"""
        if self.os_info.name == SupportedOS.WINDOWS:
            try:
                self.platform_module = importlib.import_module('windows.main')
            except ImportError as e:
                raise ImportError(f"Failed to load Windows module: {e}")

        elif self.os_info.name == SupportedOS.LINUX:
            try:
                self.platform_module = importlib.import_module('linux.main')
            except ImportError as e:
                raise ImportError(f"Failed to load Linux module: {e}")

        elif self.os_info.name == SupportedOS.MACOS:
            # For now, use Linux implementation for macOS (Unix-like)
            try:
                self.platform_module = importlib.import_module('linux.main')
            except ImportError as e:
                raise ImportError(f"Failed to load macOS module (using Linux): {e}")

        else:
            raise OSError(f"Unsupported operating system: {self.os_info.name}")

    def get_platform_main(self):
        """Get the main application class for the current platform"""
        if not self.platform_module:
            raise RuntimeError("Platform module not loaded")

        return getattr(self.platform_module, 'TerminalCoderApp')

    def get_os_info(self) -> OSInfo:
        """Get detailed OS information"""
        return self.os_info

    def is_windows(self) -> bool:
        """Check if running on Windows"""
        return self.os_info.name == SupportedOS.WINDOWS

    def is_linux(self) -> bool:
        """Check if running on Linux"""
        return self.os_info.name == SupportedOS.LINUX

    def is_macos(self) -> bool:
        """Check if running on macOS"""
        return self.os_info.name == SupportedOS.MACOS

    def get_platform_features(self) -> Dict[str, bool]:
        """Get platform-specific feature availability"""
        base_features = {
            'file_monitoring': True,
            'process_management': True,
            'network_access': True,
            'file_system_access': True,
            'environment_variables': True,
            'subprocess_execution': True,
        }

        if self.is_windows():
            windows_features = {
                'windows_registry': True,
                'windows_services': True,
                'powershell_integration': True,
                'wmi_access': True,
                'windows_api': True,
                'administrator_privileges': True,
                'windows_defender_integration': False,  # Requires special permissions
                'com_objects': True,
            }
            return base_features | windows_features

        elif self.is_linux():
            linux_features = {
                'systemd_integration': True,
                'dbus_communication': True,
                'linux_kernel_access': True,
                'package_management': True,
                'cgroup_management': True,
                'inotify_monitoring': True,
                'udev_integration': True,
                'apparmor_selinux': True,
                'container_support': True,
            }
            return base_features | linux_features

        elif self.is_macos():
            macos_features = {
                'applescript_integration': True,
                'cocoa_framework': True,
                'launchd_integration': True,
                'keychain_access': True,
                'spotlight_integration': True,
                'notification_center': True,
            }
            return base_features | macos_features

        return base_features

    def get_recommended_dependencies(self) -> list[str]:
        """Get platform-specific recommended dependencies"""
        common_deps = [
            'rich>=13.9.4',
            'click>=8.1.7',
            'aiohttp>=3.10.11',
            'aiofiles>=24.1.0',
            'pydantic>=2.10.3',
            'cryptography>=44.0.0',
            'watchdog>=6.0.0',
            'gitpython>=3.1.47',
            'typer>=0.15.1',
            'textual>=0.89.0',
        ]

        if self.is_windows():
            windows_deps = [
                'pywin32>=308',
                'wmi>=1.5.1',
                'psutil>=6.1.0',
                'colorama>=0.4.6',
            ]
            return common_deps + windows_deps

        elif self.is_linux():
            linux_deps = [
                'dbus-python>=1.3.2',
                'psutil>=6.1.0',
                'distro>=1.9.0',
            ]
            return common_deps + linux_deps

        elif self.is_macos():
            macos_deps = [
                'pyobjc-core>=10.3.1',
                'pyobjc-framework-Cocoa>=10.3.1',
                'psutil>=6.1.0',
            ]
            return common_deps + macos_deps

        return common_deps

    def print_system_info(self):
        """Print detailed system information"""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # System info table
        table = Table(title="🖥️ System Information", style="cyan")
        table.add_column("Property", style="magenta", width=20)
        table.add_column("Value", style="white")

        table.add_row("Operating System", self.os_info.name.value.title())
        table.add_row("Version", self.os_info.version)
        table.add_row("Architecture", self.os_info.architecture)
        table.add_row("64-bit", "Yes" if self.os_info.is_64bit else "No")
        table.add_row("Python Version", self.os_info.python_version)
        table.add_row("Platform", self.os_info.platform_name)

        if self.os_info.distribution:
            table.add_row("Distribution", self.os_info.distribution)

        console.print(table)

        # Platform features
        features_table = Table(title="🔧 Available Features", style="green")
        features_table.add_column("Feature", style="cyan")
        features_table.add_column("Available", style="white")

        features = self.get_platform_features()
        for feature, available in features.items():
            status = "✅ Yes" if available else "❌ No"
            features_table.add_row(feature.replace('_', ' ').title(), status)

        console.print(features_table)


def get_os_detector() -> OSDetector:
    """Get the global OS detector instance"""
    if not hasattr(get_os_detector, '_instance'):
        get_os_detector._instance = OSDetector()
    return get_os_detector._instance


def main():
    """Test the OS detector"""
    detector = get_os_detector()
    detector.print_system_info()

    print(f"\nPlatform module: {detector.platform_module}")
    print(f"Recommended dependencies: {len(detector.get_recommended_dependencies())} packages")


if __name__ == "__main__":
    main()