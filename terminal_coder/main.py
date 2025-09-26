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

        self.console.print(Panel(
            banner,
            style="bold cyan",
            border_style="bright_blue"
        ))

    def run(self, args):
        """Run the appropriate platform implementation"""
        try:
            # Display startup information
            self.display_startup_banner()

            # Get the platform-specific main application class
            app_class = self.os_detector.get_platform_main()

            # Create and run the platform-specific application
            app = app_class()

            # Run the application
            import asyncio
            return asyncio.run(app.run_interactive_mode())

        except ImportError as e:
            self.console.print(f"[red]Error: Could not load platform implementation: {e}[/red]")
            self.console.print("[yellow]Please ensure all dependencies are installed.[/yellow]")
            return False

        except Exception as e:
            self.console.print(f"[red]Fatal error: {e}[/red]")
            return False

    def __init__(self) -> None:
        self.console = Console()
        self._setup_paths()
        self._setup_logging()

        # Load configuration and projects
        self.config = self.load_config()
        self.projects = self.load_projects()
        self.current_project: Project | None = None

        # Initialize AI providers with modern syntax
        self.ai_providers = self._initialize_ai_providers()

    def _setup_paths(self) -> None:
        """Setup application paths using pathlib"""
        self.config_dir = Path.home() / self.DEFAULT_CONFIG_DIR
        self.config_file = self.config_dir / "config.json"
        self.projects_file = self.config_dir / "projects.json"
        self.session_file = self.config_dir / "session.json"

        # Create directories if they don't exist
        self.config_dir.mkdir(exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.config_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "terminal_coder.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_ai_providers(self) -> ProviderDict:
        """Initialize AI providers with updated model lists"""
        return {
            "openai": AIProvider(
                name="OpenAI",
                base_url="https://api.openai.com/v1",
                auth_type="api_key",
                models=["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
                max_tokens=128000,
                supports_streaming=True,
                rate_limit=60
            ),
            "anthropic": AIProvider(
                name="Anthropic",
                base_url="https://api.anthropic.com/v1",
                auth_type="api_key",
                models=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                max_tokens=200000,
                supports_streaming=True,
                rate_limit=50
            ),
            "google": AIProvider(
                name="Google",
                base_url="https://generativelanguage.googleapis.com/v1",
                auth_type="api_key",
                models=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
                max_tokens=32000,
                supports_streaming=True,
                rate_limit=60
            ),
            "cohere": AIProvider(
                name="Cohere",
                base_url="https://api.cohere.ai/v1",
                auth_type="bearer",
                models=["command-r-plus", "command-r", "command"],
                max_tokens=128000,
                supports_streaming=True,
                rate_limit=100
            )
        }

    def load_config(self) -> JSONData:
        """Load application configuration with modern Python features"""
        default_config: JSONData = {
            "theme": "dark",
            "auto_save": True,
            "show_line_numbers": True,
            "syntax_highlighting": True,
            "ai_provider": "openai",
            "model": "gpt-4",
            "max_tokens": 8000,  # Increased default
            "temperature": 0.7,
            "api_keys": {},
            "workspace": str(Path.home() / self.DEFAULT_WORKSPACE),
            "features": {
                "code_completion": True,
                "error_analysis": True,
                "code_review": True,
                "documentation_generation": True,
                "test_generation": True,
                "refactoring_suggestions": True,
                "security_analysis": True,
                "performance_optimization": True,
                "ai_code_explanation": True,  # New feature
                "pattern_recognition": True,  # New feature
                "code_translation": True,    # New feature
            }
        }

        if not self.config_file.exists():
            return default_config

        try:
            with self.config_file.open('r', encoding='utf-8') as f:
                config = json.load(f)

            # Modern dict merge using | operator (Python 3.9+)
            return default_config | config

        except (OSError, json.JSONDecodeError) as e:
            self.console.print(f"[red]Error loading config: {e}[/red]")
            self.logger.error(f"Config loading error: {e}")
            return default_config

    def save_config(self) -> None:
        """Save application configuration with error handling"""
        try:
            with self.config_file.open('w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info("Configuration saved successfully")
        except OSError as e:
            self.console.print(f"[red]Error saving config: {e}[/red]")
            self.logger.error(f"Config save error: {e}")

    def load_projects(self) -> ProjectList:
        """Load saved projects with modern error handling"""
        if not self.projects_file.exists():
            return []

        try:
            with self.projects_file.open('r', encoding='utf-8') as f:
                data = json.load(f)

            # Use list comprehension with error handling
            projects = []
            for project_data in data:
                try:
                    projects.append(Project(**project_data))
                except TypeError as e:
                    self.logger.warning(f"Skipping invalid project data: {e}")
                    continue

            return projects

        except (OSError, json.JSONDecodeError) as e:
            self.console.print(f"[red]Error loading projects: {e}[/red]")
            self.logger.error(f"Projects loading error: {e}")
            return []

    def save_projects(self) -> None:
        """Save projects to file with modern features"""
        try:
            # Use dataclasses.asdict for serialization
            from dataclasses import asdict
            data = [asdict(project) for project in self.projects]

            with self.projects_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(self.projects)} projects")

        except OSError as e:
            self.console.print(f"[red]Error saving projects: {e}[/red]")
            self.logger.error(f"Projects save error: {e}")

    def display_banner(self):
        """Display application banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🚀 TERMINAL CODER v1.0                             ║
║                     Advanced AI-Powered Development Terminal                  ║
║                            50+ Features | Multi-AI Support                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """

        self.console.print(Panel(
            banner,
            style="bold cyan",
            border_style="bright_blue"
        ))

    def show_main_menu(self):
        """Display main menu"""
        table = Table(title="🔧 Terminal Coder - Main Menu", style="cyan")
        table.add_column("Option", style="magenta", width=10)
        table.add_column("Description", style="white")
        table.add_column("Shortcut", style="yellow", width=10)

        menu_items = [
            ("1", "🆕 Create New Project", "Ctrl+N"),
            ("2", "📂 Open Existing Project", "Ctrl+O"),
            ("3", "🤖 AI Assistant", "Ctrl+A"),
            ("4", "⚙️  Configure Settings", "Ctrl+S"),
            ("5", "🔧 Code Tools", "Ctrl+T"),
            ("6", "📊 Project Analytics", "Ctrl+P"),
            ("7", "🧠 AI Model Manager", "Ctrl+M"),
            ("8", "🌐 API Manager", "Ctrl+I"),
            ("9", "📝 Documentation Generator", "Ctrl+D"),
            ("10", "🛡️  Security Scanner", "Ctrl+E"),
            ("11", "🚀 Deploy Assistant", "Ctrl+Y"),
            ("12", "❓ Help & Features", "F1"),
            ("0", "❌ Exit", "Ctrl+Q")
        ]

        for option, description, shortcut in menu_items:
            table.add_row(option, description, shortcut)

        self.console.print(table)

    async def run_interactive_mode(self):
        """Run the interactive terminal interface"""
        self.display_banner()

        while True:
            try:
                self.show_main_menu()

                choice = Prompt.ask(
                    "\n[bold cyan]Select an option[/bold cyan]",
                    choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    default="3"
                )

                if choice == "0":
                    if Confirm.ask("Are you sure you want to exit?"):
                        self.console.print("[green]Thank you for using Terminal Coder! 🚀[/green]")
                        break

                elif choice == "1":
                    await self.create_new_project()
                elif choice == "2":
                    await self.open_existing_project()
                elif choice == "3":
                    await self.ai_assistant()
                elif choice == "4":
                    await self.configure_settings()
                elif choice == "5":
                    await self.code_tools_menu()
                elif choice == "6":
                    await self.project_analytics()
                elif choice == "7":
                    await self.ai_model_manager()
                elif choice == "8":
                    await self.api_manager()
                elif choice == "9":
                    await self.documentation_generator()
                elif choice == "10":
                    await self.security_scanner()
                elif choice == "11":
                    await self.deploy_assistant()
                elif choice == "12":
                    await self.show_help()

            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Exit Terminal Coder?[/yellow]"):
                    break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
                self.logger.error(f"Unexpected error: {e}")

    async def create_new_project(self):
        """Create a new coding project"""
        self.console.print(Panel("🆕 Create New Project", style="green"))

        # Project details
        name = Prompt.ask("[cyan]Project name[/cyan]")
        language = Prompt.ask(
            "[cyan]Programming language[/cyan]",
            choices=["python", "javascript", "typescript", "java", "cpp", "rust", "go", "php", "ruby", "swift", "kotlin", "other"],
            default="python"
        )

        framework = None
        if language == "python":
            framework = Prompt.ask(
                "[cyan]Framework (optional)[/cyan]",
                choices=["django", "flask", "fastapi", "streamlit", "none"],
                default="none"
            )
        elif language in ["javascript", "typescript"]:
            framework = Prompt.ask(
                "[cyan]Framework (optional)[/cyan]",
                choices=["react", "vue", "angular", "express", "nextjs", "none"],
                default="none"
            )

        # AI Provider selection
        provider_names = list(self.ai_providers.keys())
        ai_provider = Prompt.ask(
            "[cyan]AI Provider[/cyan]",
            choices=provider_names,
            default=self.config.get("ai_provider", "openai")
        )

        # Model selection
        available_models = self.ai_providers[ai_provider].models
        model = Prompt.ask(
            "[cyan]AI Model[/cyan]",
            choices=available_models,
            default=available_models[0]
        )

        # Create project directory
        workspace = Path(self.config["workspace"])
        workspace.mkdir(exist_ok=True)
        project_path = workspace / name
        project_path.mkdir(exist_ok=True)

        # Create project
        from datetime import datetime
        project = Project(
            name=name,
            path=str(project_path),
            language=language,
            framework=framework if framework != "none" else None,
            ai_provider=ai_provider,
            model=model,
            created_at=datetime.now().isoformat(),
            last_modified=datetime.now().isoformat()
        )

        self.projects.append(project)
        self.current_project = project
        self.save_projects()

        # Create initial project structure
        await self.create_project_structure(project)

        self.console.print(f"[green]✅ Project '{name}' created successfully![/green]")
        self.console.print(f"[cyan]📁 Location: {project_path}[/cyan]")

    async def create_project_structure(self, project: Project):
        """Create initial project structure based on language and framework"""
        project_path = Path(project.path)

        # Create basic structure
        (project_path / "src").mkdir(exist_ok=True)
        (project_path / "tests").mkdir(exist_ok=True)
        (project_path / "docs").mkdir(exist_ok=True)

        # Create README
        readme_content = f"""# {project.name}

## Description
A {project.language} project created with Terminal Coder.

## Setup
Instructions for setting up the project.

## Usage
Instructions for using the project.

## Contributing
Guidelines for contributing to this project.

## License
Project license information.
"""

        with open(project_path / "README.md", "w") as f:
            f.write(readme_content)

        # Language-specific structure
        if project.language == "python":
            # Create setup.py, requirements.txt, etc.
            with open(project_path / "requirements.txt", "w") as f:
                f.write("# Project dependencies\n")

            with open(project_path / "src" / "main.py", "w") as f:
                f.write('"""Main application entry point"""\n\ndef main():\n    print("Hello, Terminal Coder!")\n\nif __name__ == "__main__":\n    main()\n')

        elif project.language in ["javascript", "typescript"]:
            # Create package.json
            package_json = {
                "name": project.name.lower().replace(" ", "-"),
                "version": "1.0.0",
                "description": f"A {project.language} project created with Terminal Coder",
                "main": "src/main.js" if project.language == "javascript" else "src/main.ts",
                "scripts": {
                    "start": "node src/main.js" if project.language == "javascript" else "ts-node src/main.ts",
                    "test": "npm test"
                }
            }

            with open(project_path / "package.json", "w") as f:
                json.dump(package_json, f, indent=2)

    async def ai_assistant(self):
        """AI-powered coding assistant"""
        if not self.current_project:
            self.console.print("[yellow]⚠️  No project selected. Please create or open a project first.[/yellow]")
            return

        self.console.print(Panel("🤖 AI Assistant", style="blue"))
        self.console.print(f"[cyan]Current Project: {self.current_project.name}[/cyan]")
        self.console.print(f"[cyan]AI Provider: {self.current_project.ai_provider} ({self.current_project.model})[/cyan]")

        while True:
            try:
                query = Prompt.ask("\n[bold green]Ask me anything (or 'exit' to return, '/help' for commands)[/bold green]")

                if query.lower() in ['exit', 'quit', 'back']:
                    break

                # Check for slash commands
                if query.startswith('/') and hasattr(self, 'advanced_cli'):
                    try:
                        response = await self.advanced_cli._handle_slash_command(query)
                        self.console.print(Panel(
                            response,
                            title="🔧 Command Response",
                            border_style="green"
                        ))
                        continue
                    except Exception as e:
                        self.console.print(f"[red]Command error: {e}[/red]")
                        continue

                # Check for file attachments
                files = []
                if '@' in query:
                    # Extract file references
                    import re
                    file_refs = re.findall(r'@([^\s]+)', query)
                    for ref in file_refs:
                        from pathlib import Path
                        if Path(ref).exists():
                            files.append(ref)

                # Get AI response with real implementation
                try:
                    response = await self.get_ai_response(query, files=files, stream=True)
                    # Response is already displayed by streaming, just add some metadata
                    if files:
                        self.console.print(f"[dim]📎 Processed {len(files)} file(s)[/dim]")
                except Exception as e:
                    self.console.print(f"[red]AI Error: {e}[/red]")

            except KeyboardInterrupt:
                break

    async def get_ai_response(self, query: str, files: List[str] = None, stream: bool = True) -> str:
        """Get real AI response from configured provider"""
        if not hasattr(self, 'advanced_cli') or not self.advanced_cli:
            # Initialize advanced CLI core if not already done
            from advanced_cli_core import AdvancedCLICore
            self.advanced_cli = AdvancedCLICore(self.console)

            # Initialize with API keys
            api_keys = self.config.get("api_keys", {})
            if api_keys:
                await self.advanced_cli.initialize_ai_clients(api_keys)
            else:
                return "❌ No API keys configured. Please run settings to configure AI providers."

        # Set current provider and model from project
        if self.current_project:
            self.advanced_cli.current_provider = self.current_project.ai_provider
            self.advanced_cli.current_model = self.current_project.model

        try:
            # Process the query with advanced CLI features
            response = await self.advanced_cli.process_user_input(
                user_input=query,
                files=files,
                stream=stream
            )
            return response
        except Exception as e:
            error_msg = f"AI Provider Error: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")
            return error_msg

    async def configure_settings(self):
        """Configure application settings"""
        self.console.print(Panel("⚙️ Settings Configuration", style="yellow"))

        settings_menu = [
            "1. 🔑 API Keys",
            "2. 🎨 Theme Settings",
            "3. 🤖 Default AI Provider",
            "4. 📁 Workspace Directory",
            "5. 🔧 Feature Toggles",
            "6. 🌐 Network Settings",
            "7. 💾 Auto-save Settings",
            "8. 🔙 Back to Main Menu"
        ]

        for item in settings_menu:
            self.console.print(item)

        choice = Prompt.ask("Select setting to configure", choices=["1", "2", "3", "4", "5", "6", "7", "8"])

        if choice == "1":
            await self.configure_api_keys()
        elif choice == "2":
            await self.configure_theme()
        elif choice == "3":
            await self.configure_default_ai()
        # Add more settings configurations...

    async def configure_api_keys(self):
        """Configure API keys for different providers"""
        self.console.print(Panel("🔑 API Key Configuration", style="green"))

        for provider_name, provider in self.ai_providers.items():
            current_key = self.config["api_keys"].get(provider_name, "Not set")
            masked_key = current_key[:8] + "..." if current_key != "Not set" else "Not set"

            self.console.print(f"\n[cyan]{provider.name}[/cyan] - Current: [yellow]{masked_key}[/yellow]")

            if Confirm.ask(f"Update {provider.name} API key?"):
                new_key = Prompt.ask(f"Enter {provider.name} API key", password=True)
                self.config["api_keys"][provider_name] = new_key
                self.console.print(f"[green]✅ {provider.name} API key updated[/green]")

        self.save_config()

    async def show_help(self):
        """Display help and feature list"""
        self.console.print(Panel("❓ Terminal Coder - Features & Help", style="magenta"))

        features = [
            "🤖 Multi-AI Provider Support (OpenAI, Anthropic, Google, Cohere)",
            "📝 Intelligent Code Generation & Completion",
            "🐛 Advanced Error Analysis & Debugging",
            "🔍 Code Review & Quality Analysis",
            "📚 Automatic Documentation Generation",
            "🧪 Test Case Generation & Management",
            "🔧 Code Refactoring Suggestions",
            "🛡️ Security Vulnerability Scanning",
            "⚡ Performance Optimization Tips",
            "🌐 API Integration Assistant",
            "📊 Project Analytics & Metrics",
            "🚀 Deployment & CI/CD Integration",
            "🎨 Multiple Themes & Customization",
            "💾 Auto-save & Session Management",
            "📁 Smart Project Management",
            "🔄 Version Control Integration",
            "🌍 Multi-language Support",
            "📱 Responsive Terminal UI",
            "🔍 Advanced Search & Navigation",
            "⌨️ Customizable Shortcuts",
            # Add more features...
        ]

        for i, feature in enumerate(features[:25], 1):  # Show first 25 features
            self.console.print(f"{i:2d}. {feature}")

        self.console.print(f"\n[cyan]And {len(features) - 25}+ more features![/cyan]")

        Prompt.ask("\nPress Enter to continue...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Terminal Coder - Advanced AI-Powered Development Terminal")
    parser.add_argument("--version", action="version", version="Terminal Coder v1.0")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--project", help="Open specific project")
    parser.add_argument("--ai-provider", help="Default AI provider", choices=["openai", "anthropic", "google", "cohere"])

    args = parser.parse_args()

    # Create and run the application
    app = TerminalCoder()

    if args.project:
        # Load specific project
        pass

    if args.ai_provider:
        app.config["ai_provider"] = args.ai_provider

    try:
        asyncio.run(app.run_interactive_mode())
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()