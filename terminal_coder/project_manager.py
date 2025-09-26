"""
Advanced Project Management System
Comprehensive project handling with templates, dependencies, and intelligent features
"""

import os
import json
import shutil
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import zipfile
import logging


class ProjectType(Enum):
    """Project type categories"""
    WEB_APP = "web_app"
    API = "api"
    CLI_TOOL = "cli_tool"
    LIBRARY = "library"
    MOBILE_APP = "mobile_app"
    DESKTOP_APP = "desktop_app"
    DATA_SCIENCE = "data_science"
    MACHINE_LEARNING = "machine_learning"
    GAME = "game"
    CUSTOM = "custom"


class ProjectStatus(Enum):
    """Project status states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    SUSPENDED = "suspended"


@dataclass
class Dependency:
    """Project dependency information"""
    name: str
    version: str
    type: str  # "pip", "npm", "gem", etc.
    required: bool = True
    dev_only: bool = False


@dataclass
class ProjectTemplate:
    """Project template definition"""
    name: str
    description: str
    language: str
    framework: Optional[str]
    project_type: ProjectType
    files: Dict[str, str]  # filename: content
    directories: List[str]
    dependencies: List[Dependency]
    setup_commands: List[str]
    additional_config: Dict[str, Any]


@dataclass
class ProjectMetrics:
    """Project metrics and analytics"""
    lines_of_code: int = 0
    files_count: int = 0
    commit_count: int = 0
    last_activity: Optional[datetime] = None
    test_coverage: float = 0.0
    complexity_score: int = 0
    tech_debt_score: int = 0
    security_score: int = 100


@dataclass
class Project:
    """Enhanced project data structure"""
    id: str
    name: str
    description: str
    path: str
    language: str
    framework: Optional[str]
    project_type: ProjectType
    status: ProjectStatus
    created_at: datetime
    modified_at: datetime

    # Enhanced fields
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"
    repository_url: str = ""
    dependencies: List[Dependency] = None
    build_config: Dict[str, Any] = None
    test_config: Dict[str, Any] = None
    deploy_config: Dict[str, Any] = None
    metrics: ProjectMetrics = None
    tags: List[str] = None
    ai_provider: str = "openai"
    ai_model: str = "gpt-4"
    custom_settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.build_config is None:
            self.build_config = {}
        if self.test_config is None:
            self.test_config = {}
        if self.deploy_config is None:
            self.deploy_config = {}
        if self.metrics is None:
            self.metrics = ProjectMetrics()
        if self.tags is None:
            self.tags = []
        if self.custom_settings is None:
            self.custom_settings = {}


class ProjectTemplateManager:
    """Manages project templates"""

    def __init__(self):
        self.templates: Dict[str, ProjectTemplate] = {}
        self._initialize_built_in_templates()

    def _initialize_built_in_templates(self):
        """Initialize built-in project templates"""

        # Python FastAPI Template
        self.templates["python_fastapi"] = ProjectTemplate(
            name="Python FastAPI API",
            description="Modern Python API with FastAPI framework",
            language="python",
            framework="fastapi",
            project_type=ProjectType.API,
            files={
                "main.py": '''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="{project_name}", description="API built with Terminal Coder")

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

@app.get("/")
async def root():
    return {"message": "Hello from {project_name}!"}

@app.get("/items/{{item_id}}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
async def create_item(item: Item):
    return item

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
''',
                "requirements.txt": '''fastapi>=0.68.0
uvicorn[standard]>=0.15.0
pydantic>=1.8.0
pytest>=6.0.0
httpx>=0.24.0
''',
                "README.md": '''# {project_name}

{project_description}

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload
```

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## Testing

```bash
pytest
```
''',
                ".gitignore": '''__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
''',
                "Dockerfile": '''FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
            },
            directories=["tests", "app", "docs"],
            dependencies=[
                Dependency("fastapi", ">=0.68.0", "pip"),
                Dependency("uvicorn", ">=0.15.0", "pip"),
                Dependency("pydantic", ">=1.8.0", "pip"),
                Dependency("pytest", ">=6.0.0", "pip", required=False, dev_only=True),
            ],
            setup_commands=[
                "pip install -r requirements.txt",
                "python -m pytest tests/ --verbose"
            ],
            additional_config={
                "port": 8000,
                "auto_reload": True,
                "docs_url": "/docs"
            }
        )

        # React TypeScript Template
        self.templates["react_typescript"] = ProjectTemplate(
            name="React TypeScript App",
            description="Modern React application with TypeScript",
            language="typescript",
            framework="react",
            project_type=ProjectType.WEB_APP,
            files={
                "package.json": '''{
  "name": "{project_name}",
  "version": "0.1.0",
  "description": "{project_description}",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^4.9.5"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@typescript-eslint/eslint-plugin": "^5.57.0",
    "@typescript-eslint/parser": "^5.57.0",
    "eslint": "^8.37.0",
    "eslint-plugin-react": "^7.32.2"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }
}''',
                "src/App.tsx": '''import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>{project_name}</h1>
        <p>{project_description}</p>
        <p>Built with Terminal Coder 🚀</p>
      </header>
    </div>
  );
}

export default App;
''',
                "src/index.tsx": '''import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
''',
                "public/index.html": '''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="{project_description}" />
    <title>{project_name}</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
''',
                "tsconfig.json": '''{
  "compilerOptions": {
    "target": "es5",
    "lib": [
      "dom",
      "dom.iterable",
      "esnext"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx"
  },
  "include": [
    "src"
  ]
}
'''
            },
            directories=["src", "public", "src/components", "src/hooks", "src/utils"],
            dependencies=[
                Dependency("react", "^18.2.0", "npm"),
                Dependency("react-dom", "^18.2.0", "npm"),
                Dependency("typescript", "^4.9.5", "npm"),
            ],
            setup_commands=[
                "npm install",
                "npm start"
            ],
            additional_config={
                "port": 3000,
                "hot_reload": True
            }
        )

        # Python Data Science Template
        self.templates["python_datascience"] = ProjectTemplate(
            name="Python Data Science",
            description="Data science project with Jupyter notebooks",
            language="python",
            framework="jupyter",
            project_type=ProjectType.DATA_SCIENCE,
            files={
                "requirements.txt": '''numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
plotly>=5.0.0
scipy>=1.7.0
''',
                "main.py": '''#!/usr/bin/env python3
"""
{project_name}
{project_description}

Data Science project created with Terminal Coder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """Load data from file"""
    return pd.read_csv(filepath)

def explore_data(df: pd.DataFrame) -> None:
    """Basic data exploration"""
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\\nBasic Statistics:")
    print(df.describe())

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\\nMissing Values:")
        print(missing[missing > 0])

def main():
    """Main analysis pipeline"""
    print("🚀 Starting {project_name} analysis...")

    # Your analysis code here
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
''',
                "notebooks/exploration.ipynb": '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# {project_name}\\n",
    "{project_description}\\n",
    "\\n",
    "Created with Terminal Coder 🚀"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Set plotting style\\n",
    "plt.style.use('default')\\n",
    "sns.set_palette('husl')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
            },
            directories=["data", "notebooks", "scripts", "results", "models"],
            dependencies=[
                Dependency("numpy", ">=1.21.0", "pip"),
                Dependency("pandas", ">=1.3.0", "pip"),
                Dependency("matplotlib", ">=3.4.0", "pip"),
                Dependency("jupyter", ">=1.0.0", "pip"),
            ],
            setup_commands=[
                "pip install -r requirements.txt",
                "jupyter notebook"
            ],
            additional_config={
                "jupyter_port": 8888,
                "data_dir": "data",
                "results_dir": "results"
            }
        )

    def get_template(self, template_name: str) -> Optional[ProjectTemplate]:
        """Get project template by name"""
        return self.templates.get(template_name)

    def list_templates(self) -> List[str]:
        """List available template names"""
        return list(self.templates.keys())

    def add_custom_template(self, template: ProjectTemplate):
        """Add custom project template"""
        self.templates[template.name.lower().replace(" ", "_")] = template


class AdvancedProjectManager:
    """Advanced project management system"""

    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = Path(workspace_dir or Path.home() / "terminal_coder_workspace")
        self.workspace_dir.mkdir(exist_ok=True)

        self.projects_file = self.workspace_dir / "projects.json"
        self.templates_manager = ProjectTemplateManager()
        self.logger = logging.getLogger(__name__)

        # Load existing projects
        self.projects: List[Project] = self._load_projects()

    def _load_projects(self) -> List[Project]:
        """Load projects from storage"""
        if not self.projects_file.exists():
            return []

        try:
            with open(self.projects_file, 'r') as f:
                data = json.load(f)

            projects = []
            for project_data in data:
                # Convert datetime strings back to datetime objects
                project_data['created_at'] = datetime.fromisoformat(project_data['created_at'])
                project_data['modified_at'] = datetime.fromisoformat(project_data['modified_at'])

                # Convert enums
                project_data['project_type'] = ProjectType(project_data['project_type'])
                project_data['status'] = ProjectStatus(project_data['status'])

                # Convert dependencies
                if 'dependencies' in project_data:
                    project_data['dependencies'] = [
                        Dependency(**dep) for dep in project_data['dependencies']
                    ]

                # Convert metrics
                if 'metrics' in project_data and project_data['metrics']:
                    metrics_data = project_data['metrics']
                    if 'last_activity' in metrics_data and metrics_data['last_activity']:
                        metrics_data['last_activity'] = datetime.fromisoformat(metrics_data['last_activity'])
                    project_data['metrics'] = ProjectMetrics(**metrics_data)

                projects.append(Project(**project_data))

            return projects

        except Exception as e:
            self.logger.error(f"Error loading projects: {e}")
            return []

    def _save_projects(self):
        """Save projects to storage"""
        try:
            # Convert projects to serializable format
            data = []
            for project in self.projects:
                project_dict = asdict(project)

                # Convert datetime objects to strings
                project_dict['created_at'] = project.created_at.isoformat()
                project_dict['modified_at'] = project.modified_at.isoformat()

                # Convert enums to strings
                project_dict['project_type'] = project.project_type.value
                project_dict['status'] = project.status.value

                # Handle metrics
                if project.metrics and project.metrics.last_activity:
                    project_dict['metrics']['last_activity'] = project.metrics.last_activity.isoformat()

                data.append(project_dict)

            with open(self.projects_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving projects: {e}")

    def create_project(self, name: str, description: str = "",
                      template_name: str = None, **kwargs) -> Project:
        """Create a new project"""
        # Generate unique ID
        project_id = self._generate_project_id(name)

        # Create project directory
        project_path = self.workspace_dir / name
        project_path.mkdir(exist_ok=True)

        # Get template if specified
        template = None
        if template_name:
            template = self.templates_manager.get_template(template_name)

        # Create project object
        project = Project(
            id=project_id,
            name=name,
            description=description,
            path=str(project_path),
            language=kwargs.get('language', 'python'),
            framework=kwargs.get('framework'),
            project_type=ProjectType(kwargs.get('project_type', 'custom')),
            status=ProjectStatus.ACTIVE,
            created_at=datetime.now(),
            modified_at=datetime.now(),
            **kwargs
        )

        # Apply template if available
        if template:
            self._apply_template(project, template)

        # Add to projects list
        self.projects.append(project)
        self._save_projects()

        self.logger.info(f"Created project: {name} ({project_id})")
        return project

    def _generate_project_id(self, name: str) -> str:
        """Generate unique project ID"""
        base_id = name.lower().replace(" ", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(f"{name}{timestamp}".encode()).hexdigest()[:8]
        return f"{base_id}_{hash_part}"

    def _apply_template(self, project: Project, template: ProjectTemplate):
        """Apply template to project"""
        project_path = Path(project.path)

        # Create directories
        for directory in template.directories:
            (project_path / directory).mkdir(parents=True, exist_ok=True)

        # Create files with content substitution
        substitutions = {
            "project_name": project.name,
            "project_description": project.description,
            "author": project.author,
            "version": project.version
        }

        for filename, content in template.files.items():
            # Substitute placeholders
            for placeholder, value in substitutions.items():
                content = content.replace(f"{{{placeholder}}}", value)

            file_path = project_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                f.write(content)

        # Update project properties from template
        project.dependencies.extend(template.dependencies)
        project.language = template.language
        project.framework = template.framework
        project.project_type = template.project_type
        project.custom_settings.update(template.additional_config)

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        return next((p for p in self.projects if p.id == project_id), None)

    def get_project_by_name(self, name: str) -> Optional[Project]:
        """Get project by name"""
        return next((p for p in self.projects if p.name == name), None)

    def list_projects(self, status: ProjectStatus = None) -> List[Project]:
        """List projects, optionally filtered by status"""
        if status:
            return [p for p in self.projects if p.status == status]
        return self.projects.copy()

    def update_project(self, project_id: str, **updates) -> bool:
        """Update project properties"""
        project = self.get_project(project_id)
        if not project:
            return False

        # Update properties
        for key, value in updates.items():
            if hasattr(project, key):
                setattr(project, key, value)

        project.modified_at = datetime.now()
        self._save_projects()
        return True

    def delete_project(self, project_id: str, remove_files: bool = False) -> bool:
        """Delete project"""
        project = self.get_project(project_id)
        if not project:
            return False

        # Remove from list
        self.projects = [p for p in self.projects if p.id != project_id]

        # Remove files if requested
        if remove_files:
            try:
                shutil.rmtree(project.path)
            except Exception as e:
                self.logger.error(f"Error removing project files: {e}")

        self._save_projects()
        return True

    def archive_project(self, project_id: str) -> bool:
        """Archive a project"""
        return self.update_project(project_id, status=ProjectStatus.ARCHIVED)

    async def analyze_project(self, project_id: str) -> ProjectMetrics:
        """Analyze project and update metrics"""
        project = self.get_project(project_id)
        if not project:
            return None

        project_path = Path(project.path)
        metrics = ProjectMetrics()

        try:
            # Count files and lines of code
            code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php', '.rb'}

            for file_path in project_path.rglob('*'):
                if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                    metrics.files_count += 1

                    if file_path.suffix.lower() in code_extensions:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = f.readlines()
                                # Count non-empty, non-comment lines
                                code_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
                                metrics.lines_of_code += len(code_lines)
                        except Exception:
                            pass

            # Get git information if available
            if (project_path / '.git').exists():
                try:
                    # Get commit count
                    result = subprocess.run(
                        ['git', 'rev-list', '--count', 'HEAD'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        metrics.commit_count = int(result.stdout.strip())

                    # Get last activity
                    result = subprocess.run(
                        ['git', 'log', '-1', '--format=%ci'],
                        cwd=project_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        from dateutil import parser
                        metrics.last_activity = parser.parse(result.stdout.strip())

                except Exception as e:
                    self.logger.warning(f"Error getting git info: {e}")

            # Calculate complexity score (simplified)
            metrics.complexity_score = min(100, metrics.lines_of_code // 100)

            # Update project
            project.metrics = metrics
            project.modified_at = datetime.now()
            self._save_projects()

            return metrics

        except Exception as e:
            self.logger.error(f"Error analyzing project: {e}")
            return metrics

    def search_projects(self, query: str) -> List[Project]:
        """Search projects by name, description, or tags"""
        query_lower = query.lower()
        results = []

        for project in self.projects:
            if (query_lower in project.name.lower() or
                query_lower in project.description.lower() or
                any(query_lower in tag.lower() for tag in project.tags)):
                results.append(project)

        return results

    def get_project_stats(self) -> Dict[str, Any]:
        """Get comprehensive project statistics"""
        total_projects = len(self.projects)

        if total_projects == 0:
            return {"total_projects": 0}

        # Status breakdown
        status_counts = {}
        for status in ProjectStatus:
            status_counts[status.value] = len([p for p in self.projects if p.status == status])

        # Language breakdown
        language_counts = {}
        for project in self.projects:
            lang = project.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        # Project type breakdown
        type_counts = {}
        for project in self.projects:
            ptype = project.project_type.value
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

        # Calculate totals
        total_loc = sum(p.metrics.lines_of_code for p in self.projects if p.metrics)
        total_files = sum(p.metrics.files_count for p in self.projects if p.metrics)
        total_commits = sum(p.metrics.commit_count for p in self.projects if p.metrics)

        # Recent activity
        recent_projects = sorted([p for p in self.projects if p.metrics and p.metrics.last_activity],
                               key=lambda x: x.metrics.last_activity, reverse=True)[:5]

        return {
            "total_projects": total_projects,
            "status_breakdown": status_counts,
            "language_breakdown": language_counts,
            "type_breakdown": type_counts,
            "total_lines_of_code": total_loc,
            "total_files": total_files,
            "total_commits": total_commits,
            "recent_activity": [
                {
                    "name": p.name,
                    "last_activity": p.metrics.last_activity.isoformat(),
                    "lines_of_code": p.metrics.lines_of_code
                }
                for p in recent_projects
            ]
        }

    def export_project(self, project_id: str, export_path: str = None) -> str:
        """Export project as zip file"""
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")

        if not export_path:
            export_path = f"{project.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        try:
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                project_path = Path(project.path)

                for file_path in project_path.rglob('*'):
                    if file_path.is_file():
                        # Skip hidden files and directories
                        if not any(part.startswith('.') for part in file_path.parts):
                            arcname = file_path.relative_to(project_path)
                            zipf.write(file_path, arcname)

                # Add project metadata
                metadata = {
                    "project_info": asdict(project),
                    "export_timestamp": datetime.now().isoformat(),
                    "exported_by": "Terminal Coder"
                }

                zipf.writestr("project_metadata.json", json.dumps(metadata, indent=2, default=str))

            return export_path

        except Exception as e:
            self.logger.error(f"Error exporting project: {e}")
            raise

    def import_project(self, import_path: str) -> Project:
        """Import project from zip file"""
        if not Path(import_path).exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")

        try:
            with zipfile.ZipFile(import_path, 'r') as zipf:
                # Read metadata
                try:
                    metadata_content = zipf.read("project_metadata.json")
                    metadata = json.loads(metadata_content.decode('utf-8'))
                    project_info = metadata["project_info"]
                except KeyError:
                    # No metadata, create basic project info
                    project_name = Path(import_path).stem
                    project_info = {
                        "name": project_name,
                        "description": f"Imported project from {import_path}",
                        "language": "unknown",
                        "project_type": "custom"
                    }

                # Create new project directory
                project_name = project_info["name"]
                project_path = self.workspace_dir / project_name

                # Ensure unique name
                counter = 1
                original_path = project_path
                while project_path.exists():
                    project_path = original_path.parent / f"{original_path.name}_{counter}"
                    counter += 1

                project_path.mkdir(parents=True, exist_ok=True)

                # Extract all files except metadata
                for file_info in zipf.filelist:
                    if file_info.filename != "project_metadata.json":
                        zipf.extract(file_info, project_path)

                # Create project object
                project = Project(
                    id=self._generate_project_id(project_name),
                    name=project_path.name,
                    description=project_info.get("description", ""),
                    path=str(project_path),
                    language=project_info.get("language", "unknown"),
                    framework=project_info.get("framework"),
                    project_type=ProjectType(project_info.get("project_type", "custom")),
                    status=ProjectStatus.ACTIVE,
                    created_at=datetime.now(),
                    modified_at=datetime.now()
                )

                # Add to projects
                self.projects.append(project)
                self._save_projects()

                return project

        except Exception as e:
            self.logger.error(f"Error importing project: {e}")
            raise

    async def run_project_command(self, project_id: str, command: str) -> Dict[str, Any]:
        """Run command in project directory"""
        project = self.get_project(project_id)
        if not project:
            return {"success": False, "error": "Project not found"}

        try:
            result = subprocess.run(
                command,
                cwd=project.path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 5 minutes",
                "command": command
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": command
            }