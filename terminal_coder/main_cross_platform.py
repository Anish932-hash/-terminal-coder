#!/usr/bin/env python3
"""
Terminal Coder - Cross-Platform AI-Powered Development Terminal
Advanced development environment with Windows and Linux support
Automatically detects OS and routes to appropriate implementation
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Rich for terminal UI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import rich.traceback

# Import OS detector
from os_detector import get_os_detector

# Install rich if not available
try:
    import rich
except ImportError:
    import os
    os.system("pip install rich")
    import rich

rich.traceback.install(show_locals=True)


class TerminalCoderLauncher:
    """Cross-platform launcher for Terminal Coder"""

    APP_VERSION = "2.0.0-Universal"

    def __init__(self):
        self.console = Console()
        self.os_detector = get_os_detector()

    def display_startup_banner(self):
        """Display startup banner with OS information"""
        os_info = self.os_detector.get_os_info()

        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🚀 TERMINAL CODER v2.0-UNIVERSAL                        ║
║               Cross-Platform AI-Powered Development Terminal                 ║
║                        Windows & Linux Compatible                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

🖥️  Platform: {os_info.name.value.title()}
📍 Architecture: {os_info.architecture}
🐍 Python: {os_info.python_version}
"""

        if os_info.distribution:
            banner += f"🐧 Distribution: {os_info.distribution}\n"

        features = self.os_detector.get_platform_features()
        feature_count = sum(1 for available in features.values() if available)
        banner += f"✨ Features: {feature_count} available\n"

        self.console.print(Panel(
            banner,
            style="bold cyan",
            border_style="bright_blue"
        ))

    def check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            recommended_deps = self.os_detector.get_recommended_dependencies()
            missing_deps = []

            for dep in recommended_deps[:10]:  # Check first 10 critical deps
                package_name = dep.split('>=')[0].split('==')[0]
                try:
                    __import__(package_name.replace('-', '_'))
                except ImportError:
                    missing_deps.append(package_name)

            if missing_deps:
                self.console.print("[yellow]⚠️  Missing optional dependencies:[/yellow]")
                for dep in missing_deps:
                    self.console.print(f"   • {dep}")
                self.console.print("[dim]Install with: pip install -r requirements.txt[/dim]\n")

        except Exception:
            pass  # Don't fail if dependency check fails

    def run(self, args):
        """Run the appropriate platform implementation"""
        try:
            # Display startup information
            self.display_startup_banner()

            # Check dependencies
            self.check_dependencies()

            # Show system information if requested
            if args.system_info:
                self.os_detector.print_system_info()
                return True

            # Get the platform-specific main application class
            app_class = self.os_detector.get_platform_main()

            # Create and run the platform-specific application
            self.console.print(f"[green]Launching {self.os_detector.get_os_info().name.value.title()} implementation...[/green]\n")

            app = app_class()

            # Run the application
            import asyncio
            return asyncio.run(app.run_interactive_mode())

        except ImportError as e:
            self.console.print(f"[red]Error: Could not load platform implementation: {e}[/red]")
            self.console.print("[yellow]Please ensure platform-specific dependencies are installed.[/yellow]")

            # Show installation suggestions
            os_name = self.os_detector.get_os_info().name.value
            if os_name == 'windows':
                self.console.print("[dim]Windows dependencies: pip install pywin32 wmi[/dim]")
            elif os_name == 'linux':
                self.console.print("[dim]Linux dependencies: pip install dbus-python distro[/dim]")

            return False

        except KeyboardInterrupt:
            self.console.print("\n[green]Goodbye! 👋[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Fatal error: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Terminal Coder - Cross-Platform AI Development Terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python main_cross_platform.py                    # Start Terminal Coder
  python main_cross_platform.py --system-info      # Show system information
  python main_cross_platform.py --version          # Show version

The application automatically detects your operating system and loads
the appropriate Windows or Linux implementation with all features.

Supported Platforms:
  • Windows 10/11 with PowerShell, WSL, and Windows Terminal support
  • Linux with systemd, D-Bus, and package manager integration
  • macOS (using Linux implementation)

For more information, visit: https://github.com/terminal-coder/terminal-coder
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'Terminal Coder {TerminalCoderLauncher.APP_VERSION}'
    )

    parser.add_argument(
        '--system-info',
        action='store_true',
        help='Show detailed system information and exit'
    )

    parser.add_argument(
        '--config',
        help='Path to custom configuration file'
    )

    parser.add_argument(
        '--project',
        help='Open specific project on startup'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )

    args = parser.parse_args()

    # Set up debug logging if requested
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    # Create and run the launcher
    launcher = TerminalCoderLauncher()
    success = launcher.run(args)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)