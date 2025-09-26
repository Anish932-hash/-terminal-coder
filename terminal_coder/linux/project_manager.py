#!/usr/bin/env python3
"""
Linux Project Manager
Handles project creation, management, and Linux-specific features
"""

import os
import sys
import json
import subprocess
import shutil
import distro
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import tempfile
import tarfile

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, track
from rich.prompt import Prompt, Confirm
from rich.tree import Tree


@dataclass
class LinuxProjectConfig:
    """Linux-specific project configuration"""
    use_systemd_service: bool = False
    create_desktop_file: bool = False
    enable_dbus_integration: bool = False
    create_man_page: bool = False
    use_package_manager: bool = True
    enable_container_support: bool = True
    distribution: str = field(default_factory=lambda: distro.name())


@dataclass
class Project:
    """Enhanced project configuration for Linux"""
    name: str
    path: str
    language: str
    framework: Optional[str] = None
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    linux_config: LinuxProjectConfig = field(default_factory=LinuxProjectConfig)
    dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    scripts: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create project from dictionary"""
        if 'linux_config' in data and isinstance(data['linux_config'], dict):
            data['linux_config'] = LinuxProjectConfig(**data['linux_config'])
        return cls(**data)


class LinuxProjectManager:
    """Advanced project management with Linux integration"""

    def __init__(self):
        self.console = Console()

        # Linux-specific paths (XDG Base Directory)
        xdg_config = os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
        xdg_data = os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')

        self.config_dir = Path(xdg_config) / 'terminal-coder'
        self.data_dir = Path(xdg_data) / 'terminal-coder'
        self.projects_file = self.data_dir / 'projects.json'
        self.templates_dir = self.data_dir / 'templates'

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)

        # Default workspace
        self.default_workspace = Path.home() / 'Projects'
        self.default_workspace.mkdir(exist_ok=True)

        # System information
        self.distribution = distro.name()
        self.package_manager = self._detect_package_manager()

        # Load projects
        self.projects = self._load_projects()

        # Initialize templates
        self._ensure_default_templates()

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

    def _load_projects(self) -> List[Project]:
        """Load projects from file"""
        if not self.projects_file.exists():
            return []

        try:
            with open(self.projects_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            projects = []
            for project_data in data:
                try:
                    projects.append(Project.from_dict(project_data))
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Skipping invalid project: {e}[/yellow]")

            return projects
        except Exception as e:
            self.console.print(f"[red]Error loading projects: {e}[/red]")
            return []

    def _save_projects(self):
        """Save projects to file"""
        try:
            data = [project.to_dict() for project in self.projects]

            with open(self.projects_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Set user-only permissions
            self.projects_file.chmod(0o600)

        except Exception as e:
            self.console.print(f"[red]Error saving projects: {e}[/red]")

    def _ensure_default_templates(self):
        """Ensure default project templates exist"""
        templates = {
            'python-basic': self._get_python_basic_template(),
            'python-flask': self._get_python_flask_template(),
            'python-fastapi': self._get_python_fastapi_template(),
            'python-django': self._get_python_django_template(),
            'javascript-node': self._get_javascript_node_template(),
            'rust-binary': self._get_rust_binary_template(),
            'go-module': self._get_go_module_template(),
            'cpp-cmake': self._get_cpp_cmake_template(),
            'python-systemd': self._get_python_systemd_template(),
            'bash-script': self._get_bash_script_template(),
        }

        for template_name, template_config in templates.items():
            template_file = self.templates_dir / f"{template_name}.json"
            if not template_file.exists():
                with open(template_file, 'w', encoding='utf-8') as f:
                    json.dump(template_config, f, indent=2, ensure_ascii=False)

    def _get_python_basic_template(self) -> Dict[str, Any]:
        """Get Python basic project template for Linux"""
        return {
            'name': 'Python Basic Project (Linux)',
            'language': 'python',
            'framework': None,
            'files': {
                'main.py': '''#!/usr/bin/env python3
"""
{project_name} - Main Application
Linux-optimized Python application
"""

import os
import sys
import signal
from pathlib import Path

def signal_handler(signum, frame):
    """Handle system signals gracefully"""
    print(f"\\nReceived signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    print("Hello from {project_name} on Linux!")
    print(f"Running on: {{os.uname().sysname}} {{os.uname().release}}")
    print(f"Python version: {{sys.version}}")

if __name__ == "__main__":
    main()
''',
                'requirements.txt': '''# Linux-optimized dependencies
# Add your dependencies here
''',
                'setup.py': '''#!/usr/bin/env python3
"""
Setup script for {project_name}
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="{project_name_slug}",
    version="{version}",
    author="{author}",
    description="{description}",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: {license} License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    entry_points={{
        "console_scripts": [
            "{project_name_slug}=main:main",
        ],
    }},
)
''',
                'README.md': '''# {project_name}

## Description
{description}

## Installation

### Using pip
```bash
pip install {project_name_slug}
```

### From source
```bash
git clone <repository-url>
cd {project_name_slug}
pip install -e .
```

### Linux package managers
```bash
# Ubuntu/Debian
sudo apt install {project_name_slug}

# Fedora
sudo dnf install {project_name_slug}

# Arch Linux
yay -S {project_name_slug}
```

## Usage

```bash
{project_name_slug}
```

## Development

### Setup development environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running tests
```bash
pytest
```

### Building package
```bash
python setup.py sdist bdist_wheel
```

## Linux Integration

This application is optimized for Linux systems and includes:
- Signal handling for graceful shutdown
- XDG Base Directory compliance
- systemd service integration (optional)
- Native Linux packaging support

## License
{license}
''',
                'Makefile': '''# Makefile for {project_name}

.PHONY: install test clean build package

PYTHON = python3
PIP = pip3

install:
\t$(PIP) install -e .

install-dev:
\t$(PIP) install -e .[dev]

test:
\t$(PYTHON) -m pytest

clean:
\tbuild/
\tdist/
\t*.egg-info/
\t__pycache__/
\t*.pyc

build:
\t$(PYTHON) setup.py sdist bdist_wheel

package: clean build
\ttwine check dist/*

install-system:
\tsudo $(PIP) install .

uninstall:
\t$(PIP) uninstall {project_name_slug}
''',
                '.gitignore': '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Linux
.directory
Thumbs.db
ehthumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
''',
            },
            'directories': ['src', 'tests', 'docs', 'scripts'],
            'dependencies': [],
            'dev_dependencies': ['pytest', 'black', 'flake8', 'mypy'],
            'scripts': {
                'test': 'python -m pytest',
                'format': 'black .',
                'lint': 'flake8 .',
                'type-check': 'mypy .',
                'build': 'python setup.py sdist bdist_wheel',
            }
        }

    def _get_python_systemd_template(self) -> Dict[str, Any]:
        """Get Python systemd service template"""
        return {
            'name': 'Python Systemd Service',
            'language': 'python',
            'framework': 'systemd',
            'files': {
                'main.py': '''#!/usr/bin/env python3
"""
{project_name} - Systemd Service
Linux systemd service written in Python
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

# Configure logging for systemd
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class {project_name_class}Service:
    """Main service class"""

    def __init__(self):
        self.running = False

    async def start(self):
        """Start the service"""
        logger.info("Starting {project_name} service")
        self.running = True

        try:
            while self.running:
                # Main service loop
                logger.debug("Service heartbeat")
                await asyncio.sleep(60)  # Sleep for 1 minute

        except Exception as e:
            logger.error(f"Service error: {{e}}")
            raise

    async def stop(self):
        """Stop the service"""
        logger.info("Stopping {project_name} service")
        self.running = False

# Global service instance
service = {project_name_class}Service()

def signal_handler(signum, frame):
    """Handle system signals"""
    logger.info(f"Received signal {{signum}}")
    if signum in [signal.SIGTERM, signal.SIGINT]:
        asyncio.create_task(service.stop())

async def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Service interrupted")
    except Exception as e:
        logger.error(f"Service failed: {{e}}")
        sys.exit(1)
    finally:
        logger.info("Service shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
''',
                f'{project_name_slug}.service': '''[Unit]
Description={project_name} - {description}
After=network.target
Wants=network.target

[Service]
Type=exec
User=nobody
Group=nogroup
ExecStart=/usr/bin/python3 /usr/local/bin/{project_name_slug}
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier={project_name_slug}

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/{project_name_slug}

[Install]
WantedBy=multi-user.target
''',
                'install_service.sh': '''#!/bin/bash
# Install systemd service for {project_name}

set -e

SERVICE_FILE="{project_name_slug}.service"
SERVICE_DIR="/etc/systemd/system"
BIN_FILE="/usr/local/bin/{project_name_slug}"

echo "Installing {project_name} systemd service..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Copy service file
cp "$SERVICE_FILE" "$SERVICE_DIR/"

# Copy binary
cp "main.py" "$BIN_FILE"
chmod +x "$BIN_FILE"

# Create data directory
mkdir -p "/var/lib/{project_name_slug}"
chown nobody:nogroup "/var/lib/{project_name_slug}"

# Reload systemd
systemctl daemon-reload

# Enable service
systemctl enable {project_name_slug}.service

echo "Service installed successfully!"
echo "Start with: sudo systemctl start {project_name_slug}"
echo "Check status: sudo systemctl status {project_name_slug}"
echo "View logs: sudo journalctl -u {project_name_slug} -f"
''',
            },
            'directories': ['lib', 'etc', 'logs'],
            'dependencies': [],
            'dev_dependencies': ['pytest', 'systemd-python'],
        }

    def _get_bash_script_template(self) -> Dict[str, Any]:
        """Get Bash script template"""
        return {
            'name': 'Bash Script Project',
            'language': 'bash',
            'framework': 'shell',
            'files': {
                f'{project_name_slug}.sh': '''#!/bin/bash
# {project_name} - {description}
# Linux bash script

set -euo pipefail

# Script configuration
SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LOG_FILE="/tmp/${{SCRIPT_NAME}}.log"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Logging functions
log() {{
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" | tee -a "$LOG_FILE"
}}

info() {{
    log "INFO" "$1"
    echo -e "${{BLUE}}[INFO]${{NC}} $1"
}}

warn() {{
    log "WARN" "$1"
    echo -e "${{YELLOW}}[WARN]${{NC}} $1" >&2
}}

error() {{
    log "ERROR" "$1"
    echo -e "${{RED}}[ERROR]${{NC}} $1" >&2
}}

success() {{
    log "SUCCESS" "$1"
    echo -e "${{GREEN}}[SUCCESS]${{NC}} $1"
}}

# Cleanup function
cleanup() {{
    info "Cleaning up..."
    # Add cleanup code here
}}

# Signal handlers
trap cleanup EXIT
trap 'error "Script interrupted"; exit 130' INT
trap 'error "Script terminated"; exit 143' TERM

# Help function
show_help() {{
    cat << EOF
{project_name} - {description}

Usage: $SCRIPT_NAME [OPTIONS]

Options:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -q, --quiet     Suppress output
    --version       Show version information

Examples:
    $SCRIPT_NAME
    $SCRIPT_NAME --verbose

EOF
}}

# Version information
show_version() {{
    echo "{project_name} version {version}"
    echo "Running on: $(uname -s) $(uname -r)"
    echo "Bash version: ${{BASH_VERSION}}"
}}

# Main function
main() {{
    local verbose=false
    local quiet=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            -q|--quiet)
                quiet=true
                shift
                ;;
            --version)
                show_version
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Check if running on Linux
    if [[ "$(uname -s)" != "Linux" ]]; then
        warn "This script is optimized for Linux systems"
    fi

    info "Starting {project_name}..."

    # Main script logic here
    success "Hello from {project_name}!"
    info "Script completed successfully"
}}

# Run main function with all arguments
main "$@"
''',
                'README.md': '''# {project_name}

{description}

## Installation

```bash
# Make executable
chmod +x {project_name_slug}.sh

# Install to system
sudo cp {project_name_slug}.sh /usr/local/bin/{project_name_slug}
```

## Usage

```bash
./{project_name_slug}.sh --help
```

## Features

- Linux-optimized bash script
- Proper error handling and logging
- Signal handling for graceful shutdown
- Colorized output
- Comprehensive argument parsing

## Requirements

- Bash 4.0+
- Linux environment
- Standard Unix utilities

## License

{license}
''',
                'install.sh': '''#!/bin/bash
# Installation script for {project_name}

set -euo pipefail

SCRIPT_NAME="{project_name_slug}.sh"
INSTALL_DIR="/usr/local/bin"
INSTALL_PATH="$INSTALL_DIR/{project_name_slug}"

echo "Installing {project_name}..."

# Check if script exists
if [[ ! -f "$SCRIPT_NAME" ]]; then
    echo "Error: $SCRIPT_NAME not found"
    exit 1
fi

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    echo "Please run as root (use sudo)"
    exit 1
fi

# Copy script
cp "$SCRIPT_NAME" "$INSTALL_PATH"
chmod +x "$INSTALL_PATH"

echo "Successfully installed to $INSTALL_PATH"
echo "Run with: {project_name_slug}"
''',
            },
            'directories': ['lib', 'tests', 'docs'],
        }

    # Include other template methods...
    def _get_python_flask_template(self) -> Dict[str, Any]:
        """Get Flask template"""
        return {
            'name': 'Python Flask Web Application',
            'language': 'python',
            'framework': 'flask',
            'files': {
                'app.py': '''#!/usr/bin/env python3
"""
{project_name} - Flask Web Application
Linux-optimized Flask application
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html',
                         project_name='{project_name}',
                         system_info=get_system_info())

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({{
        'status': 'healthy',
        'project': '{project_name}',
        'version': '{version}'
    }})

def get_system_info():
    """Get Linux system information"""
    return {{
        'system': os.uname().sysname,
        'release': os.uname().release,
        'python_version': sys.version.split()[0]
    }}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    app.run(host='0.0.0.0', port=port, debug=debug)
''',
                'requirements.txt': '''Flask>=3.0.0
gunicorn>=21.2.0
''',
                'templates/index.html': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ project_name }}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .system-info {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ project_name }}</h1>
        <p>Welcome to your Flask application running on Linux!</p>

        <div class="system-info">
            <h3>System Information</h3>
            <p><strong>System:</strong> {{ system_info.system }}</p>
            <p><strong>Release:</strong> {{ system_info.release }}</p>
            <p><strong>Python:</strong> {{ system_info.python_version }}</p>
        </div>
    </div>
</body>
</html>
''',
                'Dockerfile': '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 5000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
''',
            },
            'directories': ['templates', 'static', 'tests'],
            'dependencies': ['Flask>=3.0.0', 'gunicorn>=21.2.0'],
        }

    # Add more template methods as needed...

    async def create_project(self) -> Optional[Project]:
        """Create a new project interactively"""
        self.console.print(Panel("🚀 Create New Linux Project", style="green"))

        # Get basic project information
        name = Prompt.ask("[cyan]Project name[/cyan]")
        if not name:
            self.console.print("[red]Project name is required[/red]")
            return None

        # Check if project already exists
        if any(p.name == name for p in self.projects):
            if not Confirm.ask(f"[yellow]Project '{name}' already exists. Continue anyway?[/yellow]"):
                return None

        description = Prompt.ask("[cyan]Project description[/cyan]", default="")
        author = Prompt.ask("[cyan]Author[/cyan]", default=os.environ.get('USER', ''))

        # Choose project template
        available_templates = self._get_available_templates()
        self.console.print("\n[cyan]Available project templates:[/cyan]")

        for i, (template_id, template_info) in enumerate(available_templates.items(), 1):
            self.console.print(f"{i}. {template_info['name']} ({template_info['language']})")

        template_choice = Prompt.ask(
            "Choose template",
            choices=[str(i) for i in range(1, len(available_templates) + 1)],
            default="1"
        )

        template_id = list(available_templates.keys())[int(template_choice) - 1]
        template = available_templates[template_id]

        # Get project location
        default_path = self.default_workspace / name
        project_path = Path(Prompt.ask(
            f"[cyan]Project location[/cyan]",
            default=str(default_path)
        ))

        # Linux-specific options
        linux_config = LinuxProjectConfig()

        if Confirm.ask("[cyan]Configure Linux-specific options?[/cyan]"):
            linux_config.use_systemd_service = Confirm.ask(
                "Create systemd service?", default=False
            )
            linux_config.create_desktop_file = Confirm.ask(
                "Create .desktop file?", default=False
            )
            linux_config.enable_dbus_integration = Confirm.ask(
                "Enable D-Bus integration?", default=False
            )
            linux_config.create_man_page = Confirm.ask(
                "Create manual page?", default=False
            )
            linux_config.enable_container_support = Confirm.ask(
                "Enable container support?", default=True
            )

        # Create project
        project = Project(
            name=name,
            path=str(project_path),
            language=template['language'],
            framework=template.get('framework'),
            description=description,
            author=author,
            linux_config=linux_config,
            dependencies=template.get('dependencies', []),
            dev_dependencies=template.get('dev_dependencies', []),
            scripts=template.get('scripts', {})
        )

        # Create project files and structure
        success = await self._create_project_structure(project, template)

        if success:
            self.projects.append(project)
            self._save_projects()

            self.console.print(Panel(
                f"✅ Project '{name}' created successfully!\n\n"
                f"📁 Location: {project_path}\n"
                f"💻 Language: {template['language']}\n"
                f"🐧 Linux optimized: Yes\n"
                f"📦 Package manager: {self.package_manager}",
                title="Project Created",
                style="green"
            ))

            return project
        else:
            self.console.print("[red]Failed to create project[/red]")
            return None

    def _get_available_templates(self) -> Dict[str, Dict]:
        """Get available project templates"""
        templates = {}

        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                    templates[template_file.stem] = template_data
            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not load template {template_file}: {e}[/yellow]")

        return templates

    async def _create_project_structure(self, project: Project, template: Dict[str, Any]) -> bool:
        """Create project directory structure and files"""
        try:
            project_path = Path(project.path)

            # Create project directory
            project_path.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            directories = template.get('directories', [])
            for directory in directories:
                (project_path / directory).mkdir(exist_ok=True)

            # Create files from template
            files = template.get('files', {})
            for filename, content in files.items():
                # Replace template variables
                content = self._replace_template_variables(content, project, template)
                filename = self._replace_template_variables(filename, project, template)

                file_path = project_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # Make scripts executable
                if filename.endswith(('.sh', '.py')) or 'bin' in str(file_path):
                    file_path.chmod(0o755)

            # Create Linux-specific files
            await self._create_linux_specific_files(project)

            # Set up version control
            if Confirm.ask("Initialize Git repository?", default=True):
                await self._initialize_git_repo(project_path)

            return True

        except Exception as e:
            self.console.print(f"[red]Error creating project structure: {e}[/red]")
            return False

    def _replace_template_variables(self, content: str, project: Project, template: Dict[str, Any]) -> str:
        """Replace template variables in content"""
        import re
        import uuid

        # Generate derived variables
        project_name_slug = re.sub(r'[^a-zA-Z0-9]', '-', project.name.lower()).strip('-')
        project_name_class = re.sub(r'[^a-zA-Z0-9]', '', project.name.title())
        project_name_package = re.sub(r'[^a-zA-Z0-9]', '_', project.name.lower()).strip('_')
        current_year = datetime.now().year

        replacements = {
            '{project_name}': project.name,
            '{project_name_slug}': project_name_slug,
            '{project_name_class}': project_name_class,
            '{project_name_package}': project_name_package,
            '{description}': project.description,
            '{author}': project.author,
            '{version}': project.version,
            '{license}': project.license,
            '{year}': str(current_year),
            '{distribution}': self.distribution,
            '{package_manager}': self.package_manager,
        }

        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)

        return content

    async def _create_linux_specific_files(self, project: Project):
        """Create Linux-specific project files"""
        project_path = Path(project.path)

        # Create .gitignore
        gitignore_content = '''# Linux specific
*~
.directory
.DS_Store
Thumbs.db

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
venv/
env/
.env

# Node.js
node_modules/
npm-debug.log*

# Rust
target/
Cargo.lock

# C/C++
*.o
*.so
*.a
build/

# Go
vendor/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Build outputs
dist/
*.deb
*.rpm
*.tar.gz
*.AppImage

# Logs
*.log
logs/

# Runtime
*.pid
*.sock
'''

        with open(project_path / '.gitignore', 'w') as f:
            f.write(gitignore_content)

        # Create .desktop file if requested
        if project.linux_config.create_desktop_file:
            await self._create_desktop_file(project)

        # Create systemd service if requested
        if project.linux_config.use_systemd_service:
            await self._create_systemd_service(project)

        # Create man page if requested
        if project.linux_config.create_man_page:
            await self._create_man_page(project)

    async def _create_desktop_file(self, project: Project):
        """Create .desktop file for the application"""
        project_path = Path(project.path)
        project_name_slug = re.sub(r'[^a-zA-Z0-9]', '-', project.name.lower()).strip('-')

        desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name={project.name}
Comment={project.description}
Exec={project_path / 'main.py' if project.language == 'python' else project_path / f'{project_name_slug}'}
Icon=applications-development
Terminal=false
Categories=Development;Programming;
Keywords=development;programming;
StartupNotify=true
"""

        desktop_file = project_path / f"{project_name_slug}.desktop"
        with open(desktop_file, 'w') as f:
            f.write(desktop_content)

        desktop_file.chmod(0o644)

    async def _create_systemd_service(self, project: Project):
        """Create systemd service file"""
        # This would be implemented based on the project type
        pass

    async def _create_man_page(self, project: Project):
        """Create manual page for the application"""
        project_path = Path(project.path)
        man_dir = project_path / 'man'
        man_dir.mkdir(exist_ok=True)

        project_name_slug = re.sub(r'[^a-zA-Z0-9]', '-', project.name.lower()).strip('-')

        man_content = f'''.TH {project_name_slug.upper()} 1 "{datetime.now().strftime('%B %Y')}" "version {project.version}" "User Commands"
.SH NAME
{project_name_slug} \\- {project.description}
.SH SYNOPSIS
.B {project_name_slug}
[\\fIOPTIONS\\fR]
.SH DESCRIPTION
{project.description}

This manual page documents the
.B {project_name_slug}
command.
.SH OPTIONS
.TP
\\fB\\-h\\fR, \\fB\\-\\-help\\fR
Display help message and exit.
.TP
\\fB\\-v\\fR, \\fB\\-\\-version\\fR
Display version information and exit.
.SH EXAMPLES
.TP
{project_name_slug}
Run the application with default settings.
.SH AUTHOR
{project.author}
.SH SEE ALSO
Documentation may be available at /usr/share/doc/{project_name_slug}/
'''

        man_file = man_dir / f"{project_name_slug}.1"
        with open(man_file, 'w') as f:
            f.write(man_content)

    async def _initialize_git_repo(self, project_path: Path):
        """Initialize Git repository"""
        try:
            result = subprocess.run(
                ['git', 'init'],
                cwd=project_path,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Create initial commit
                subprocess.run(['git', 'add', '.'], cwd=project_path)
                subprocess.run([
                    'git', 'commit', '-m', 'Initial commit - Linux optimized project'
                ], cwd=project_path)

                self.console.print("[green]Git repository initialized[/green]")
            else:
                self.console.print(f"[yellow]Warning: Git init failed: {result.stderr}[/yellow]")

        except FileNotFoundError:
            self.console.print("[yellow]Warning: Git not found. Repository not initialized.[/yellow]")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Git initialization failed: {e}[/yellow]")

    async def open_project(self) -> Optional[Project]:
        """Open existing project"""
        if not self.projects:
            self.console.print("[yellow]No projects found. Create a project first.[/yellow]")
            return None

        self.console.print(Panel("📂 Open Existing Project", style="blue"))

        # Display projects table
        table = Table(title="Available Linux Projects", style="cyan")
        table.add_column("ID", style="magenta", width=5)
        table.add_column("Name", style="white")
        table.add_column("Language", style="green")
        table.add_column("Framework", style="yellow")
        table.add_column("Last Modified", style="blue")
        table.add_column("Path", style="dim")

        for i, project in enumerate(self.projects, 1):
            last_modified = datetime.fromisoformat(project.last_modified).strftime("%Y-%m-%d %H:%M")
            framework = project.framework or "None"
            path_display = str(Path(project.path).name)

            table.add_row(
                str(i),
                project.name,
                project.language.title(),
                framework,
                last_modified,
                path_display
            )

        self.console.print(table)

        # Let user choose project
        try:
            choice = int(Prompt.ask(
                "Enter project ID",
                choices=[str(i) for i in range(1, len(self.projects) + 1)]
            ))

            selected_project = self.projects[choice - 1]

            # Verify project path exists
            if not Path(selected_project.path).exists():
                if Confirm.ask(f"[yellow]Project path does not exist: {selected_project.path}\nRemove from list?[/yellow]"):
                    self.projects.remove(selected_project)
                    self._save_projects()
                return None

            # Update last accessed time
            selected_project.last_modified = datetime.now().isoformat()
            self._save_projects()

            self.console.print(f"[green]✅ Opened project: {selected_project.name}[/green]")
            return selected_project

        except (ValueError, IndexError):
            self.console.print("[red]Invalid project ID[/red]")
            return None