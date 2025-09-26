#!/usr/bin/env python3
"""
Windows Project Manager
Handles project creation, management, and Windows-specific features
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import tempfile
import zipfile

# Windows-specific imports
import winreg
import win32api
import win32con

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, track
from rich.prompt import Prompt, Confirm
from rich.tree import Tree


@dataclass
class WindowsProjectConfig:
    """Windows-specific project configuration"""
    use_powershell_scripts: bool = True
    create_batch_files: bool = True
    register_file_associations: bool = False
    create_desktop_shortcut: bool = False
    enable_windows_defender_exclusion: bool = False
    use_windows_terminal_profiles: bool = True


@dataclass
class Project:
    """Enhanced project configuration for Windows"""
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
    windows_config: WindowsProjectConfig = field(default_factory=WindowsProjectConfig)
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
        if 'windows_config' in data and isinstance(data['windows_config'], dict):
            data['windows_config'] = WindowsProjectConfig(**data['windows_config'])
        return cls(**data)


class WindowsProjectManager:
    """Advanced project management with Windows integration"""

    def __init__(self):
        self.console = Console()

        # Windows-specific paths
        appdata = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        self.config_dir = appdata / 'TerminalCoder'
        self.projects_file = self.config_dir / 'projects.json'
        self.templates_dir = self.config_dir / 'templates'

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)

        # Default workspace in Documents
        self.default_workspace = Path.home() / 'Documents' / 'TerminalCoderProjects'
        self.default_workspace.mkdir(exist_ok=True)

        # Load projects
        self.projects = self._load_projects()

        # Initialize templates
        self._ensure_default_templates()

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

        except Exception as e:
            self.console.print(f"[red]Error saving projects: {e}[/red]")

    def _ensure_default_templates(self):
        """Ensure default project templates exist"""
        templates = {
            'python-basic': self._get_python_basic_template(),
            'python-fastapi': self._get_python_fastapi_template(),
            'javascript-node': self._get_javascript_node_template(),
            'csharp-console': self._get_csharp_console_template(),
            'powershell-module': self._get_powershell_module_template(),
            'python-gui': self._get_python_gui_template(),
        }

        for template_name, template_config in templates.items():
            template_file = self.templates_dir / f"{template_name}.json"
            if not template_file.exists():
                with open(template_file, 'w', encoding='utf-8') as f:
                    json.dump(template_config, f, indent=2, ensure_ascii=False)

    def _get_python_basic_template(self) -> Dict[str, Any]:
        """Get Python basic project template"""
        return {
            'name': 'Python Basic Project',
            'language': 'python',
            'framework': None,
            'files': {
                'main.py': '''#!/usr/bin/env python3
"""
{project_name} - Main Application
"""

def main():
    """Main entry point"""
    print("Hello from {project_name}!")

if __name__ == "__main__":
    main()
''',
                'requirements.txt': '''# Project dependencies
# Add your dependencies here
''',
                'README.md': '''# {project_name}

## Description
{description}

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python main.py
```

## Development

### Windows PowerShell
```powershell
# Create virtual environment
python -m venv venv
.\\venv\\Scripts\\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Command Prompt
```cmd
# Create virtual environment
python -m venv venv
venv\\Scripts\\activate.bat

# Install dependencies
pip install -r requirements.txt
```

## License
{license}
''',
                'setup.bat': '''@echo off
echo Setting up {project_name}...
python -m venv venv
call venv\\Scripts\\activate.bat
pip install -r requirements.txt
echo Setup complete!
pause
''',
                'run.bat': '''@echo off
call venv\\Scripts\\activate.bat
python main.py
pause
''',
                'setup.ps1': '''# PowerShell setup script for {project_name}
Write-Host "Setting up {project_name}..." -ForegroundColor Green

# Create virtual environment
python -m venv venv

# Activate virtual environment
& .\\venv\\Scripts\\Activate.ps1

# Install dependencies
pip install -r requirements.txt

Write-Host "Setup complete!" -ForegroundColor Green
''',
            },
            'directories': ['src', 'tests', 'docs', 'data'],
            'dependencies': [],
            'dev_dependencies': ['pytest', 'black', 'flake8'],
            'scripts': {
                'test': 'python -m pytest',
                'format': 'black .',
                'lint': 'flake8 .',
            }
        }

    def _get_python_fastapi_template(self) -> Dict[str, Any]:
        """Get Python FastAPI project template"""
        return {
            'name': 'Python FastAPI Project',
            'language': 'python',
            'framework': 'fastapi',
            'files': {
                'main.py': '''#!/usr/bin/env python3
"""
{project_name} - FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="{project_name}",
    description="{description}",
    version="{version}"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {{"message": "Hello from {project_name}!"}}

@app.get("/health")
async def health():
    return {{"status": "healthy"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
                'requirements.txt': '''fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.4.2
python-multipart>=0.0.6
''',
                'Dockerfile': '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
''',
                'docker-compose.yml': '''version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - PYTHONPATH=/app
''',
                'start.bat': '''@echo off
echo Starting {project_name} API server...
call venv\\Scripts\\activate.bat
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
''',
                'start.ps1': '''# PowerShell start script for {project_name}
Write-Host "Starting {project_name} API server..." -ForegroundColor Green

# Activate virtual environment
& .\\venv\\Scripts\\Activate.ps1

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
''',
            },
            'directories': ['src', 'tests', 'docs', 'static', 'templates'],
            'dependencies': ['fastapi', 'uvicorn[standard]', 'pydantic', 'python-multipart'],
            'dev_dependencies': ['pytest', 'httpx', 'black', 'flake8'],
            'scripts': {
                'start': 'uvicorn main:app --reload',
                'test': 'python -m pytest',
                'format': 'black .',
                'lint': 'flake8 .',
            }
        }

    def _get_javascript_node_template(self) -> Dict[str, Any]:
        """Get JavaScript Node.js project template"""
        return {
            'name': 'JavaScript Node.js Project',
            'language': 'javascript',
            'framework': 'node',
            'files': {
                'index.js': '''// {project_name} - Main Application

console.log('Hello from {project_name}!');

// Example Express.js server (uncomment to use)
/*
const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
    res.json({ message: 'Hello from {project_name}!' });
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
*/
''',
                'package.json': '''{
  "name": "{project_name_slug}",
  "version": "{version}",
  "description": "{description}",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js",
    "test": "jest"
  },
  "keywords": [],
  "author": "{author}",
  "license": "{license}",
  "dependencies": {},
  "devDependencies": {
    "nodemon": "^3.0.1",
    "jest": "^29.7.0"
  }
}
''',
                'start.bat': '''@echo off
echo Starting {project_name}...
npm start
pause
''',
                'setup.bat': '''@echo off
echo Setting up {project_name}...
npm install
echo Setup complete!
pause
''',
                'dev.bat': '''@echo off
echo Starting {project_name} in development mode...
npm run dev
pause
''',
            },
            'directories': ['src', 'tests', 'public', 'docs'],
            'dependencies': [],
            'dev_dependencies': ['nodemon', 'jest'],
        }

    def _get_csharp_console_template(self) -> Dict[str, Any]:
        """Get C# Console project template"""
        return {
            'name': 'C# Console Project',
            'language': 'csharp',
            'framework': 'dotnet',
            'files': {
                'Program.cs': '''using System;

namespace {project_name_namespace}
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello from {project_name}!");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}
''',
                f'{{}}.csproj': '''<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
  </PropertyGroup>

</Project>
''',
                'build.bat': '''@echo off
echo Building {project_name}...
dotnet build
pause
''',
                'run.bat': '''@echo off
echo Running {project_name}...
dotnet run
pause
''',
                'publish.bat': '''@echo off
echo Publishing {project_name}...
dotnet publish -c Release -o publish
echo Published to publish folder
pause
''',
            },
            'directories': ['Properties'],
        }

    def _get_powershell_module_template(self) -> Dict[str, Any]:
        """Get PowerShell module project template"""
        return {
            'name': 'PowerShell Module Project',
            'language': 'powershell',
            'framework': 'powershell',
            'files': {
                f'{{}}.psm1': '''# {project_name} PowerShell Module

# Export functions
Export-ModuleMember -Function *

# Module initialization
Write-Host "Loading {project_name} module..." -ForegroundColor Green

function Get-{project_name_function}Info {{
    <#
    .SYNOPSIS
    Gets information about {project_name}

    .DESCRIPTION
    This function returns basic information about the {project_name} module.

    .EXAMPLE
    Get-{project_name_function}Info
    #>

    return @{{
        Name = "{project_name}"
        Version = "{version}"
        Author = "{author}"
        Description = "{description}"
    }}
}}

function Invoke-{project_name_function}Action {{
    <#
    .SYNOPSIS
    Performs a sample action

    .DESCRIPTION
    This function demonstrates a basic action in the {project_name} module.

    .PARAMETER Message
    The message to display

    .EXAMPLE
    Invoke-{project_name_function}Action -Message "Hello World"
    #>

    param(
        [string]$Message = "Hello from {project_name}!"
    )

    Write-Host $Message -ForegroundColor Cyan
}}
''',
                f'{{}}.psd1': '''@{{
    # Module manifest for {project_name}

    RootModule = '{project_name}.psm1'
    ModuleVersion = '{version}'
    GUID = '{guid}'
    Author = '{author}'
    CompanyName = 'Unknown'
    Copyright = '(c) {year} {author}. All rights reserved.'
    Description = '{description}'
    PowerShellVersion = '5.1'

    FunctionsToExport = @('Get-{project_name_function}Info', 'Invoke-{project_name_function}Action')
    CmdletsToExport = @()
    VariablesToExport = @()
    AliasesToExport = @()

    PrivateData = @{{
        PSData = @{{
            Tags = @('PowerShell', 'Module')
            LicenseUri = ''
            ProjectUri = ''
            IconUri = ''
            ReleaseNotes = ''
        }}
    }}
}}
''',
                'install.ps1': '''# Install {project_name} module

$ModuleName = "{project_name}"
$ModulePath = "$env:USERPROFILE\\Documents\\WindowsPowerShell\\Modules\\$ModuleName"

Write-Host "Installing $ModuleName module..." -ForegroundColor Green

# Create module directory
if (!(Test-Path $ModulePath)) {{
    New-Item -Path $ModulePath -ItemType Directory -Force
}}

# Copy module files
Copy-Item -Path ".\\$ModuleName.psm1" -Destination $ModulePath
Copy-Item -Path ".\\$ModuleName.psd1" -Destination $ModulePath

Write-Host "Module installed to: $ModulePath" -ForegroundColor Green
Write-Host "Import with: Import-Module $ModuleName" -ForegroundColor Cyan
''',
            },
            'directories': ['Functions', 'Tests'],
        }

    def _get_python_gui_template(self) -> Dict[str, Any]:
        """Get Python GUI project template"""
        return {
            'name': 'Python GUI Project',
            'language': 'python',
            'framework': 'tkinter',
            'files': {
                'main.py': '''#!/usr/bin/env python3
"""
{project_name} - GUI Application
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sys
from pathlib import Path

class {project_name_class}App:
    """Main application class"""

    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_widgets()

    def setup_window(self):
        """Setup main window"""
        self.root.title("{project_name}")
        self.root.geometry("800x600")
        self.root.minsize(400, 300)

        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{{x}}+{{y}}")

    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="{project_name}",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Input field
        ttk.Label(main_frame, text="Enter text:").grid(row=1, column=0, sticky=tk.W)
        self.text_entry = ttk.Entry(main_frame, width=50)
        self.text_entry.grid(row=1, column=1, padx=(10, 0), pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Process",
                  command=self.process_text).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear",
                  command=self.clear_text).pack(side=tk.LEFT, padx=(5, 0))

        # Output area
        ttk.Label(main_frame, text="Output:").grid(row=3, column=0, sticky=tk.NW)

        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=3, column=1, padx=(10, 0), pady=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))

        self.output_text = tk.Text(output_frame, height=15, width=60)
        scrollbar = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)

        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

    def process_text(self):
        """Process the input text"""
        text = self.text_entry.get()
        if text:
            result = f"Processed: {{text}}\\n"
            self.output_text.insert(tk.END, result)
            self.output_text.see(tk.END)
        else:
            messagebox.showwarning("Warning", "Please enter some text first.")

    def clear_text(self):
        """Clear input and output"""
        self.text_entry.delete(0, tk.END)
        self.output_text.delete(1.0, tk.END)

def main():
    """Main entry point"""
    root = tk.Tk()
    app = {project_name_class}App(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.quit()

if __name__ == "__main__":
    main()
''',
                'requirements.txt': '''# GUI dependencies
# tkinter is included with Python
# Add additional GUI libraries as needed
''',
                'build_exe.py': '''# Build executable with PyInstaller
import PyInstaller.__main__
import sys
from pathlib import Path

# Build configuration
PyInstaller.__main__.run([
    'main.py',
    '--name={project_name}',
    '--onefile',
    '--windowed',
    '--icon=icon.ico',
    '--add-data=README.md;.',
    '--hidden-import=tkinter',
    '--hidden-import=tkinter.ttk',
])
''',
                'build.bat': '''@echo off
echo Building {project_name} executable...
call venv\\Scripts\\activate.bat
python build_exe.py
echo Build complete! Check dist folder.
pause
''',
            },
            'directories': ['src', 'tests', 'docs', 'assets'],
            'dependencies': [],
            'dev_dependencies': ['pyinstaller'],
        }

    async def create_project(self) -> Optional[Project]:
        """Create a new project interactively"""
        self.console.print(Panel("🚀 Create New Windows Project", style="green"))

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
        author = Prompt.ask("[cyan]Author[/cyan]", default=os.environ.get('USERNAME', ''))

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

        # Windows-specific options
        windows_config = WindowsProjectConfig()

        if Confirm.ask("[cyan]Configure Windows-specific options?[/cyan]"):
            windows_config.use_powershell_scripts = Confirm.ask(
                "Create PowerShell scripts?", default=True
            )
            windows_config.create_batch_files = Confirm.ask(
                "Create batch files?", default=True
            )
            windows_config.create_desktop_shortcut = Confirm.ask(
                "Create desktop shortcut?", default=False
            )
            windows_config.use_windows_terminal_profiles = Confirm.ask(
                "Create Windows Terminal profile?", default=True
            )

        # Create project
        project = Project(
            name=name,
            path=str(project_path),
            language=template['language'],
            framework=template.get('framework'),
            description=description,
            author=author,
            windows_config=windows_config,
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
                f"🖥️  Windows optimized: Yes",
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

            # Create Windows-specific files
            await self._create_windows_specific_files(project)

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
        project_name_slug = re.sub(r'[^a-zA-Z0-9]', '-', project.name.lower())
        project_name_class = re.sub(r'[^a-zA-Z0-9]', '', project.name.title())
        project_name_function = re.sub(r'[^a-zA-Z0-9]', '', project.name.title())
        project_name_namespace = re.sub(r'[^a-zA-Z0-9]', '_', project.name)
        current_year = datetime.now().year
        guid = str(uuid.uuid4())

        replacements = {
            '{project_name}': project.name,
            '{project_name_slug}': project_name_slug,
            '{project_name_class}': project_name_class,
            '{project_name_function}': project_name_function,
            '{project_name_namespace}': project_name_namespace,
            '{description}': project.description,
            '{author}': project.author,
            '{version}': project.version,
            '{license}': project.license,
            '{year}': str(current_year),
            '{guid}': guid,
            '{}': project.name,  # For filename replacements
        }

        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)

        return content

    async def _create_windows_specific_files(self, project: Project):
        """Create Windows-specific project files"""
        project_path = Path(project.path)

        # Create .gitignore
        gitignore_content = '''# Windows specific
Thumbs.db
ehthumbs.db
Desktop.ini
$RECYCLE.BIN/
*.tmp
*.temp

# Visual Studio
.vs/
*.user
*.sln.docstates
bin/
obj/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
.venv/
env/
.env/

# Node.js
node_modules/
npm-debug.log*
.npm

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Build outputs
dist/
build/
*.exe
*.msi

# Logs
*.log
logs/
'''

        with open(project_path / '.gitignore', 'w') as f:
            f.write(gitignore_content)

        # Create Windows Terminal profile (if requested)
        if project.windows_config.use_windows_terminal_profiles:
            await self._create_windows_terminal_profile(project)

        # Create desktop shortcut (if requested)
        if project.windows_config.create_desktop_shortcut:
            await self._create_desktop_shortcut(project)

    async def _create_windows_terminal_profile(self, project: Project):
        """Create Windows Terminal profile for the project"""
        try:
            # Windows Terminal settings location
            terminal_settings_dir = Path(os.environ.get('LOCALAPPDATA', '')) / 'Packages' / 'Microsoft.WindowsTerminal_8wekyb3d8bbwe' / 'LocalState'

            if terminal_settings_dir.exists():
                profile = {
                    "guid": f"{{{project.name.upper()}-{datetime.now().strftime('%Y%m%d')}}}",
                    "name": f"{project.name} - Development",
                    "commandline": f"cmd.exe /k cd /d \"{project.path}\"",
                    "startingDirectory": project.path,
                    "icon": "📁"
                }

                # Save profile info for user to manually add
                profile_file = Path(project.path) / 'windows_terminal_profile.json'
                with open(profile_file, 'w', encoding='utf-8') as f:
                    json.dump(profile, f, indent=2)

                self.console.print(f"[green]Windows Terminal profile saved to {profile_file}[/green]")
                self.console.print("[yellow]Add this profile to your Windows Terminal settings manually[/yellow]")

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not create Windows Terminal profile: {e}[/yellow]")

    async def _create_desktop_shortcut(self, project: Project):
        """Create desktop shortcut for the project"""
        try:
            import win32com.client

            desktop = Path.home() / 'Desktop'
            shortcut_path = desktop / f"{project.name}.lnk"

            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = "explorer.exe"
            shortcut.Arguments = f'"{project.path}"'
            shortcut.WorkingDirectory = project.path
            shortcut.IconLocation = "shell32.dll,3"  # Folder icon
            shortcut.save()

            self.console.print(f"[green]Desktop shortcut created: {shortcut_path}[/green]")

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not create desktop shortcut: {e}[/yellow]")

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
                    'git', 'commit', '-m', 'Initial commit'
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
        table = Table(title="Available Projects", style="cyan")
        table.add_column("ID", style="magenta", width=5)
        table.add_column("Name", style="white")
        table.add_column("Language", style="green")
        table.add_column("Framework", style="yellow")
        table.add_column("Last Modified", style="blue")
        table.add_column("Path", style="dim")

        for i, project in enumerate(self.projects, 1):
            last_modified = datetime.fromisoformat(project.last_modified).strftime("%Y-%m-%d %H:%M")
            framework = project.framework or "None"
            path_display = str(Path(project.path).name)  # Show only folder name

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

    def display_projects(self):
        """Display all projects"""
        if not self.projects:
            self.console.print("[yellow]No projects found[/yellow]")
            return

        table = Table(title="🚀 Terminal Coder Projects", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Language", style="green")
        table.add_column("Framework", style="yellow")
        table.add_column("Created", style="blue")
        table.add_column("Location", style="magenta")

        for project in self.projects:
            created = datetime.fromisoformat(project.created_at).strftime("%Y-%m-%d")
            framework = project.framework or "None"

            table.add_row(
                project.name,
                project.language.title(),
                framework,
                created,
                str(Path(project.path).parent.name)
            )

        self.console.print(table)

    async def delete_project(self, project_name: str) -> bool:
        """Delete a project"""
        project = next((p for p in self.projects if p.name == project_name), None)

        if not project:
            self.console.print(f"[red]Project '{project_name}' not found[/red]")
            return False

        if not Confirm.ask(f"[red]Delete project '{project_name}' and all its files?[/red]"):
            return False

        try:
            # Remove project files
            project_path = Path(project.path)
            if project_path.exists():
                shutil.rmtree(project_path)

            # Remove from projects list
            self.projects.remove(project)
            self._save_projects()

            self.console.print(f"[green]✅ Project '{project_name}' deleted successfully[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]Error deleting project: {e}[/red]")
            return False

    def get_project_by_name(self, name: str) -> Optional[Project]:
        """Get project by name"""
        return next((p for p in self.projects if p.name == name), None)