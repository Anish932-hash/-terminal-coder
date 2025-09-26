#!/usr/bin/env python3
"""
Terminal Coder Launcher
Quick start script for the Terminal Coder application
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Terminal Coder requires Python 3.8 or higher")
        print(f"Current version: {sys.version}")
        print("Please upgrade Python and try again.")
        return False
    return True


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'rich',
        'aiohttp',
        'pydantic',
        'cryptography',
        'watchdog'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    return missing_packages


def install_dependencies(missing_packages):
    """Install missing dependencies"""
    print(f"📦 Installing missing dependencies: {', '.join(missing_packages)}")

    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--upgrade'
        ] + missing_packages)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please install manually using: pip install -r requirements.txt")
        return False


def setup_directories():
    """Setup required directories"""
    config_dir = Path.home() / ".terminal_coder"
    workspace_dir = Path.home() / "terminal_coder_workspace"

    config_dir.mkdir(exist_ok=True)
    workspace_dir.mkdir(exist_ok=True)

    print(f"📁 Config directory: {config_dir}")
    print(f"📁 Workspace directory: {workspace_dir}")


def run_application(args):
    """Run the main application"""
    try:
        # Import and run the main application
        from main import main
        main()
    except ImportError as e:
        print(f"❌ Error importing main application: {e}")
        print("Please ensure all dependencies are installed correctly.")
        return False
    except KeyboardInterrupt:
        print("\n👋 Terminal Coder stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Terminal Coder - Advanced AI-Powered Development Terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Start Terminal Coder normally
  python run.py --install-deps     # Install dependencies and start
  python run.py --check            # Check system requirements
  python run.py --setup            # Setup directories only

For more information, visit: https://github.com/terminal-coder/terminal-coder
        """
    )

    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install missing dependencies before starting'
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Check system requirements and dependencies'
    )

    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup directories and configuration'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Terminal Coder v1.0.0'
    )

    args = parser.parse_args()

    # Print banner
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🚀 Terminal Coder v1.0                   ║
║            Advanced AI-Powered Development Terminal          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # Check Python version
    if not check_python_version():
        return 1

    # Check dependencies
    missing_packages = check_dependencies()

    if args.check:
        # Just check and report
        print("🔍 System Check:")
        print(f"✅ Python version: {sys.version.split()[0]}")

        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("Run with --install-deps to install them automatically")
        else:
            print("✅ All dependencies are installed")

        return 0

    if args.setup:
        # Setup directories
        setup_directories()
        print("✅ Setup completed!")
        return 0

    # Install dependencies if requested or if missing
    if args.install_deps or missing_packages:
        if missing_packages:
            if not install_dependencies(missing_packages):
                return 1

    # Setup directories
    setup_directories()

    # Run the application
    print("🚀 Starting Terminal Coder...")
    success = run_application(args)
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)