#!/usr/bin/env python3
"""
Terminal Coder - Setup Script
Advanced cross-platform setup and build configuration
"""

import sys
import os
import platform
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install
import subprocess

# Read version from __init__.py
def get_version():
    init_file = Path(__file__).parent / "terminal_coder" / "__init__.py"
    if init_file.exists():
        with open(init_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('\'"')
    return "2.0.0"

# Read long description from README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "Terminal Coder - AI-Powered Development Terminal"

# Read requirements from requirements.txt
def get_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    return []

# Platform-specific requirements
def get_platform_requirements():
    system = platform.system().lower()
    requirements = []

    if system == 'windows':
        requirements.extend([
            'pywin32>=306',
            'wmi>=1.5.1',
            'winshell>=0.6'
        ])
    elif system == 'linux':
        requirements.extend([
            'distro>=1.8.0',
            'dbus-python>=1.3.0',
            'keyring>=24.0.0',
            'secretstorage>=3.3.0'
        ])
    elif system == 'darwin':
        requirements.extend([
            'pyobjc-core>=9.0',
            'pyobjc-framework-Cocoa>=9.0'
        ])

    return requirements

# Custom build command
class CustomBuildPy(build_py):
    """Custom build command that handles platform-specific builds"""

    def run(self):
        super().run()
        self.build_platform_specific()

    def build_platform_specific(self):
        """Build platform-specific components"""
        system = platform.system().lower()
        print(f"Building for {system}...")

        # Create platform-specific directories
        build_lib = Path(self.build_lib)
        platform_dir = build_lib / "terminal_coder" / system
        platform_dir.mkdir(parents=True, exist_ok=True)

        # Copy platform-specific files
        source_dir = Path(__file__).parent / system
        if source_dir.exists():
            import shutil
            for file in source_dir.rglob("*.py"):
                dest_file = platform_dir / file.relative_to(source_dir)
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dest_file)
                print(f"Copied {file} to {dest_file}")

# Custom install command
class CustomInstall(install):
    """Custom install command with post-install setup"""

    def run(self):
        super().run()
        self.post_install_setup()

    def post_install_setup(self):
        """Post-installation setup"""
        print("Running post-installation setup...")

        # Create application directories
        self.create_app_directories()

        # Setup platform integration
        self.setup_platform_integration()

        print("Post-installation setup completed!")

    def create_app_directories(self):
        """Create application directories"""
        system = platform.system().lower()

        if system == 'linux':
            # Use XDG directories
            config_dir = Path.home() / '.config' / 'terminal-coder'
            data_dir = Path.home() / '.local' / 'share' / 'terminal-coder'
        elif system == 'windows':
            # Use AppData
            config_dir = Path.home() / 'AppData' / 'Roaming' / 'terminal-coder'
            data_dir = config_dir
        elif system == 'darwin':
            # Use macOS conventions
            config_dir = Path.home() / 'Library' / 'Application Support' / 'terminal-coder'
            data_dir = config_dir
        else:
            config_dir = Path.home() / '.terminal-coder'
            data_dir = config_dir

        # Create directories
        config_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / 'logs').mkdir(exist_ok=True)
        (data_dir / 'projects').mkdir(exist_ok=True)

        print(f"Created directories: {config_dir}, {data_dir}")

    def setup_platform_integration(self):
        """Setup platform-specific integration"""
        system = platform.system().lower()

        try:
            if system == 'linux':
                self.setup_linux_integration()
            elif system == 'windows':
                self.setup_windows_integration()
            elif system == 'darwin':
                self.setup_macos_integration()
        except Exception as e:
            print(f"Warning: Could not setup platform integration: {e}")

    def setup_linux_integration(self):
        """Setup Linux desktop integration"""
        desktop_dir = Path.home() / '.local' / 'share' / 'applications'
        desktop_dir.mkdir(parents=True, exist_ok=True)

        desktop_file = desktop_dir / 'terminal-coder.desktop'
        desktop_content = f"""[Desktop Entry]
Name=Terminal Coder
Comment=AI-Powered Development Terminal
GenericName=Development Terminal
Exec={sys.executable} -m terminal_coder
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
        print(f"Created desktop entry: {desktop_file}")

    def setup_windows_integration(self):
        """Setup Windows integration"""
        # This would typically create Start Menu shortcuts
        # For now, just print a message
        print("Windows integration: Start Menu shortcuts can be created manually")

    def setup_macos_integration(self):
        """Setup macOS integration"""
        # This would typically create .app bundles
        # For now, just print a message
        print("macOS integration: Application bundle can be created manually")

# Package metadata
PACKAGE_NAME = "terminal-coder"
VERSION = get_version()
DESCRIPTION = "AI-Powered Development Terminal with Advanced Features"
LONG_DESCRIPTION = get_long_description()
AUTHOR = "Terminal Coder Team"
AUTHOR_EMAIL = "team@terminal-coder.ai"
URL = "https://github.com/terminal-coder/terminal-coder"
LICENSE = "MIT"

# Requirements
INSTALL_REQUIRES = [
    # Core dependencies
    "rich>=13.0.0",
    "asyncio-throttle>=1.0.0",
    "aiofiles>=23.0.0",
    "psutil>=5.9.0",
    "requests>=2.28.0",
    "cryptography>=41.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
    "toml>=0.10.0",
    "click>=8.1.0",
    "typer>=0.9.0",
    "httpx>=0.24.0",
    "numpy>=1.24.0",
] + get_platform_requirements()

# Optional dependencies
EXTRAS_REQUIRE = {
    'ai': [
        'openai>=1.0.0',
        'anthropic>=0.7.0',
        'google-generativeai>=0.3.0',
        'cohere>=4.0.0',
        'tiktoken>=0.5.0',
    ],
    'ml': [
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'pandas>=2.0.0',
        'networkx>=3.0.0',
        'pillow>=10.0.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-asyncio>=0.21.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'mypy>=1.5.0',
        'pre-commit>=3.0.0',
        'coverage>=7.0.0',
        'sphinx>=7.0.0',
    ],
    'build': [
        'PyInstaller>=6.0.0',
        'pyinstaller-hooks-contrib>=2023.8',
        'setuptools>=68.0.0',
        'wheel>=0.41.0',
        'build>=0.10.0',
    ]
}

# Add 'all' extra that includes everything
EXTRAS_REQUIRE['all'] = []
for extra_deps in EXTRAS_REQUIRE.values():
    EXTRAS_REQUIRE['all'].extend(extra_deps)

# Console scripts
CONSOLE_SCRIPTS = [
    'terminal-coder=terminal_coder.__main__:main',
    'tc=terminal_coder.__main__:main',
]

# Classifiers
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Environment :: X11 Applications",
    "Environment :: Win32 (MS Windows)",
    "Environment :: MacOS X",
    "Intended Audience :: Developers",
    "Intended Audience :: System Administrators",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development",
    "Topic :: System :: Systems Administration",
    "Topic :: Terminals",
    "Topic :: Text Editors :: Integrated Development Environments (IDE)",
    "Topic :: Utilities",
]

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Package data
PACKAGE_DATA = {
    'terminal_coder': [
        'linux/*.py',
        'windows/*.py',
        'config/*.json',
        'templates/*',
        'assets/*',
    ]
}

# Data files for different platforms
DATA_FILES = []
system = platform.system().lower()

if system == 'linux':
    DATA_FILES.extend([
        ('share/applications', ['assets/terminal-coder.desktop']),
        ('share/pixmaps', ['assets/terminal-coder.png']),
        ('share/man/man1', ['docs/terminal-coder.1']),
    ])
elif system == 'windows':
    DATA_FILES.extend([
        ('Scripts', ['scripts/terminal-coder.bat']),
    ])

# Setup configuration
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,

    # Package discovery
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    data_files=DATA_FILES,
    include_package_data=True,

    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=PYTHON_REQUIRES,

    # Console scripts
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },

    # Metadata
    classifiers=CLASSIFIERS,
    keywords="terminal ai development coding assistant cross-platform",
    project_urls={
        "Documentation": "https://terminal-coder.ai/docs",
        "Source": "https://github.com/terminal-coder/terminal-coder",
        "Tracker": "https://github.com/terminal-coder/terminal-coder/issues",
        "Funding": "https://github.com/sponsors/terminal-coder",
    },

    # Custom commands
    cmdclass={
        'build_py': CustomBuildPy,
        'install': CustomInstall,
    },

    # Build system
    setup_requires=[
        'setuptools>=68.0.0',
        'wheel>=0.41.0',
    ],

    # Zip safety
    zip_safe=False,
)