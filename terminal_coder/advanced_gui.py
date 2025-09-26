"""
Advanced Terminal GUI Module
Rich, interactive terminal interface with 50+ features
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import json

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.rule import Rule
from rich.status import Status
from rich.spinner import Spinner
from rich.traceback import install as install_traceback
from rich.logging import RichHandler
from rich.filesize import decimal
from rich.highlighter import RegexHighlighter
from rich.theme import Theme

install_traceback(show_locals=True)


class CustomHighlighter(RegexHighlighter):
    """Custom syntax highlighter for terminal output"""
    base_style = "example."
    highlights = [
        r"(?P<brace>[\[\]{}()])",
        r"(?P<tag_start><)(?P<tag_name>[-\w.]+)(?P<tag_contents>.*?)(?P<tag_end>>)",
        r"(?P<attrib_name>[-\w.]+)=(?P<attrib_value>\".*?\")",
        r"(?P<number>-?\d+\.?\d*)",
        r"(?P<bool>True|False|None)",
        r"(?P<string>\".*?\"|'.*?')",
        r"(?P<path>/[^\s]*|[A-Za-z]:\\[^\s]*)",
        r"(?P<url>https?://[^\s]+)",
        r"(?P<email>\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)",
    ]


class AdvancedGUI:
    """Advanced Terminal GUI with rich features"""

    def __init__(self):
        # Custom theme
        custom_theme = Theme({
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
            "success": "green",
            "highlight": "magenta",
            "secondary": "bright_black",
            "primary": "blue",
            "accent": "bright_cyan",
            "example.brace": "bold magenta",
            "example.tag_start": "bold blue",
            "example.tag_name": "bold blue",
            "example.tag_end": "bold blue",
            "example.attrib_name": "cyan",
            "example.attrib_value": "green",
            "example.number": "bold magenta",
            "example.bool": "bold red",
            "example.string": "yellow",
            "example.path": "bold cyan",
            "example.url": "blue underline",
            "example.email": "bright_green",
        })

        self.console = Console(
            theme=custom_theme,
            highlighter=CustomHighlighter(),
            force_terminal=True,
            color_system="truecolor"
        )

        self.layout = Layout()
        self.current_view = "main"
        self.status_messages = []
        self.notifications = []

        self._setup_layouts()
        self._setup_styles()

    def _setup_layouts(self):
        """Setup different layout configurations"""
        self.layouts = {
            "main": self._create_main_layout(),
            "code_editor": self._create_code_editor_layout(),
            "project_browser": self._create_project_browser_layout(),
            "ai_chat": self._create_ai_chat_layout(),
            "settings": self._create_settings_layout(),
            "analytics": self._create_analytics_layout()
        }

    def _setup_styles(self):
        """Setup custom styles and themes"""
        self.styles = {
            "header": "bold white on blue",
            "sidebar": "white on dark_blue",
            "main_content": "white on black",
            "footer": "white on dark_green",
            "border": "bright_blue",
            "accent": "bright_cyan",
            "success": "bright_green",
            "warning": "bright_yellow",
            "error": "bright_red"
        }

    def _create_main_layout(self) -> Layout:
        """Create main application layout"""
        layout = Layout(name="root")

        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="sidebar", minimum_size=30, ratio=1),
            Layout(name="body", ratio=3),
        )

        layout["body"].split(
            Layout(name="content", ratio=1),
            Layout(name="status", size=10),
        )

        return layout

    def _create_code_editor_layout(self) -> Layout:
        """Create code editor layout"""
        layout = Layout(name="editor_root")

        layout.split(
            Layout(name="editor_header", size=3),
            Layout(name="editor_main", ratio=1),
            Layout(name="editor_footer", size=2),
        )

        layout["editor_main"].split_row(
            Layout(name="file_tree", size=25),
            Layout(name="editor_area", ratio=1),
            Layout(name="minimap", size=15),
        )

        layout["editor_area"].split(
            Layout(name="tabs", size=2),
            Layout(name="code", ratio=1),
            Layout(name="terminal", size=8),
        )

        return layout

    def _create_project_browser_layout(self) -> Layout:
        """Create project browser layout"""
        layout = Layout(name="browser_root")

        layout.split_row(
            Layout(name="project_list", ratio=1),
            Layout(name="project_details", ratio=2),
        )

        layout["project_details"].split(
            Layout(name="project_info", size=8),
            Layout(name="project_files", ratio=1),
            Layout(name="project_stats", size=6),
        )

        return layout

    def _create_ai_chat_layout(self) -> Layout:
        """Create AI chat interface layout"""
        layout = Layout(name="chat_root")

        layout.split(
            Layout(name="chat_header", size=3),
            Layout(name="chat_main", ratio=1),
            Layout(name="chat_input", size=4),
        )

        layout["chat_main"].split_row(
            Layout(name="conversation", ratio=3),
            Layout(name="chat_sidebar", ratio=1),
        )

        return layout

    def _create_settings_layout(self) -> Layout:
        """Create settings interface layout"""
        layout = Layout(name="settings_root")

        layout.split_row(
            Layout(name="settings_menu", size=25),
            Layout(name="settings_content", ratio=1),
        )

        return layout

    def _create_analytics_layout(self) -> Layout:
        """Create analytics dashboard layout"""
        layout = Layout(name="analytics_root")

        layout.split(
            Layout(name="analytics_header", size=3),
            Layout(name="analytics_main", ratio=1),
        )

        layout["analytics_main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="charts", ratio=2),
        )

        layout["metrics"].split(
            Layout(name="overview", ratio=1),
            Layout(name="detailed", ratio=1),
        )

        return layout

    def switch_layout(self, layout_name: str):
        """Switch to a different layout"""
        if layout_name in self.layouts:
            self.layout = self.layouts[layout_name]
            self.current_view = layout_name

    def create_header(self, title: str = "Terminal Coder", subtitle: str = None) -> Panel:
        """Create application header"""
        now = datetime.now().strftime("%H:%M:%S")

        header_content = []
        header_content.append(f"🚀 {title}")

        if subtitle:
            header_content.append(f"[dim]{subtitle}[/dim]")

        header_content.append(f"[dim]Time: {now}[/dim]")

        header_text = " | ".join(header_content)

        return Panel(
            Align.center(header_text),
            style=self.styles["header"],
            height=3
        )

    def create_sidebar(self, items: List[Dict]) -> Panel:
        """Create sidebar with navigation items"""
        tree = Tree("📁 Navigation")

        for item in items:
            branch = tree.add(f"{item.get('icon', '•')} {item['name']}")
            if 'children' in item:
                for child in item['children']:
                    branch.add(f"{child.get('icon', '◦')} {child['name']}")

        return Panel(
            tree,
            title="Menu",
            style=self.styles["sidebar"],
            border_style=self.styles["border"]
        )

    def create_main_content(self, content: Any) -> Panel:
        """Create main content panel"""
        return Panel(
            content,
            style=self.styles["main_content"],
            border_style=self.styles["border"],
            padding=(1, 2)
        )

    def create_footer(self, status: str = "Ready", shortcuts: List[str] = None) -> Panel:
        """Create application footer with status and shortcuts"""
        footer_items = [f"Status: {status}"]

        if shortcuts:
            footer_items.extend(shortcuts)

        footer_text = " | ".join(footer_items)

        return Panel(
            footer_text,
            style=self.styles["footer"],
            height=3
        )

    def create_status_panel(self, messages: List[str]) -> Panel:
        """Create status/log panel"""
        content = "\n".join(messages[-8:]) if messages else "No messages"

        return Panel(
            content,
            title="Status",
            style="white on dark_gray",
            height=10
        )

    def display_welcome_screen(self):
        """Display welcome screen with features"""
        welcome_text = """
# 🚀 Welcome to Terminal Coder

## ✨ Features Available:

### 🤖 AI Integration
- Multiple AI providers (OpenAI, Anthropic, Google, Cohere)
- Automatic model detection
- Smart error handling
- Rate limiting & optimization

### 💻 Code Tools
- Syntax highlighting for 100+ languages
- Code completion & suggestions
- Error detection & debugging
- Performance optimization
- Security scanning

### 📁 Project Management
- Smart project templates
- Git integration
- Dependency management
- Build system integration

### 🎨 Interface
- Multiple themes
- Customizable layouts
- Keyboard shortcuts
- Real-time updates

### 🛠️ Advanced Features
- Terminal multiplexing
- Session management
- Plugin system
- API testing tools
- Documentation generation

## 🚀 Get Started:
1. Configure your AI API keys
2. Create or open a project
3. Start coding with AI assistance!
        """

        markdown_content = Markdown(welcome_text)

        self.console.print(Panel(
            markdown_content,
            title="🎉 Terminal Coder",
            border_style="bright_blue",
            padding=(1, 2)
        ))

    def display_feature_grid(self):
        """Display feature grid with icons"""
        features = [
            ("🤖", "AI Assistant", "Multi-provider AI integration"),
            ("💻", "Code Editor", "Advanced syntax highlighting"),
            ("🐛", "Debugger", "Intelligent error detection"),
            ("🧪", "Testing", "Automated test generation"),
            ("📚", "Docs", "Auto documentation"),
            ("🔍", "Search", "Advanced code search"),
            ("🛡️", "Security", "Vulnerability scanning"),
            ("⚡", "Performance", "Code optimization"),
            ("🌐", "API Tools", "REST/GraphQL testing"),
            ("📊", "Analytics", "Code metrics & insights"),
            ("🎨", "Themes", "Customizable interface"),
            ("⌨️", "Shortcuts", "Productivity hotkeys"),
            ("🔧", "Extensions", "Plugin ecosystem"),
            ("📱", "Templates", "Project scaffolding"),
            ("🚀", "Deploy", "CI/CD integration"),
            ("💾", "Backup", "Auto-save & recovery")
        ]

        columns = []
        for icon, title, desc in features:
            feature_panel = Panel(
                f"{icon}\n[bold]{title}[/bold]\n[dim]{desc}[/dim]",
                width=20,
                height=6,
                style="white on dark_blue"
            )
            columns.append(feature_panel)

        grid = Columns(columns, equal=True, expand=True)
        self.console.print(Panel(grid, title="🌟 Features", border_style="cyan"))

    def create_progress_display(self, tasks: List[Dict]) -> Panel:
        """Create progress display for multiple tasks"""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

        for task in tasks:
            progress.add_task(
                task['description'],
                total=task.get('total', 100),
                completed=task.get('completed', 0)
            )

        return Panel(
            progress,
            title="Progress",
            border_style="green"
        )

    def create_code_display(self, code: str, language: str = "python",
                          theme: str = "monokai", line_numbers: bool = True) -> Panel:
        """Display code with syntax highlighting"""
        syntax = Syntax(
            code,
            language,
            theme=theme,
            line_numbers=line_numbers,
            word_wrap=True
        )

        return Panel(
            syntax,
            title=f"Code - {language}",
            border_style="bright_green",
            expand=False
        )

    def create_file_tree(self, root_path: str) -> Tree:
        """Create file tree structure"""
        def add_files(tree_node, path):
            try:
                items = sorted(Path(path).iterdir(),
                             key=lambda x: (x.is_file(), x.name.lower()))

                for item in items:
                    if item.name.startswith('.'):
                        continue

                    if item.is_dir():
                        folder_node = tree_node.add(f"📁 {item.name}")
                        # Recursively add subdirectories (limit depth)
                        if len(str(item).split(os.sep)) < 5:
                            add_files(folder_node, item)
                    else:
                        # Add file with appropriate icon
                        icon = self._get_file_icon(item.suffix)
                        tree_node.add(f"{icon} {item.name}")

            except PermissionError:
                tree_node.add("🚫 Permission denied")

        tree = Tree(f"📁 {Path(root_path).name}")
        add_files(tree, root_path)
        return tree

    def _get_file_icon(self, extension: str) -> str:
        """Get appropriate icon for file type"""
        icons = {
            '.py': '🐍', '.js': '💛', '.ts': '🔷', '.html': '🌐',
            '.css': '🎨', '.json': '📊', '.md': '📝', '.txt': '📄',
            '.yml': '⚙️', '.yaml': '⚙️', '.xml': '📋', '.sql': '🗄️',
            '.sh': '📜', '.bat': '📜', '.exe': '⚙️', '.dll': '🔧',
            '.jpg': '🖼️', '.png': '🖼️', '.gif': '🖼️', '.svg': '🎨',
            '.mp4': '🎥', '.mp3': '🎵', '.pdf': '📕', '.doc': '📘',
            '.zip': '📦', '.tar': '📦', '.gz': '📦', '.log': '📋'
        }
        return icons.get(extension.lower(), '📄')

    def create_table_display(self, title: str, columns: List[str],
                           data: List[List[str]], style: str = "cyan") -> Table:
        """Create rich table display"""
        table = Table(title=title, style=style, show_header=True, header_style="bold magenta")

        for column in columns:
            table.add_column(column)

        for row in data:
            table.add_row(*[str(cell) for cell in row])

        return table

    def create_notification(self, message: str, type: str = "info",
                          duration: int = 3) -> Panel:
        """Create notification panel"""
        styles = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red"
        }

        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }

        return Panel(
            f"{icons.get(type, 'ℹ️')} {message}",
            style=styles.get(type, "blue"),
            width=60
        )

    def create_command_palette(self, commands: List[Dict]) -> Panel:
        """Create command palette interface"""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Shortcut", style="dim")

        for cmd in commands:
            table.add_row(
                cmd['name'],
                cmd.get('description', ''),
                cmd.get('shortcut', '')
            )

        return Panel(
            table,
            title="🔍 Command Palette",
            border_style="bright_magenta"
        )

    def create_metrics_dashboard(self, metrics: Dict) -> Panel:
        """Create metrics dashboard"""
        content = []

        for category, values in metrics.items():
            content.append(f"\n[bold cyan]{category.title()}[/bold cyan]")
            for key, value in values.items():
                content.append(f"  {key}: [yellow]{value}[/yellow]")

        return Panel(
            "\n".join(content),
            title="📊 Metrics",
            border_style="green"
        )

    def create_chat_interface(self, messages: List[Dict]) -> Panel:
        """Create AI chat interface"""
        content = []

        for msg in messages[-10:]:  # Show last 10 messages
            role = msg.get('role', 'user')
            text = msg.get('content', '')
            timestamp = msg.get('timestamp', datetime.now().strftime("%H:%M"))

            if role == 'user':
                content.append(f"[bold blue]👤 You ({timestamp}):[/bold blue]")
                content.append(f"  {text}")
            else:
                content.append(f"[bold green]🤖 AI ({timestamp}):[/bold green]")
                content.append(f"  {text}")
            content.append("")

        return Panel(
            "\n".join(content),
            title="🤖 AI Chat",
            border_style="blue",
            height=20
        )

    def create_settings_form(self, settings: Dict) -> Panel:
        """Create settings configuration form"""
        content = []

        for section, options in settings.items():
            content.append(f"\n[bold yellow]{section.title()}[/bold yellow]")

            for key, value in options.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    content.append(f"  {key}: {status}")
                elif isinstance(value, (int, float)):
                    content.append(f"  {key}: [cyan]{value}[/cyan]")
                else:
                    content.append(f"  {key}: [green]{value}[/green]")

        return Panel(
            "\n".join(content),
            title="⚙️ Settings",
            border_style="yellow"
        )

    async def show_loading_animation(self, message: str, duration: float = 2.0):
        """Show loading animation"""
        with self.console.status(f"[bold blue]{message}...") as status:
            await asyncio.sleep(duration)

    def clear_screen(self):
        """Clear terminal screen"""
        self.console.clear()

    def print_banner(self, text: str, style: str = "bold cyan"):
        """Print banner text"""
        self.console.print(Panel(
            Align.center(text),
            style=style,
            expand=False
        ))

    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"[bold red]❌ Error:[/bold red] {message}")

    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"[bold green]✅ Success:[/bold green] {message}")

    def print_warning(self, message: str):
        """Print warning message"""
        self.console.print(f"[bold yellow]⚠️ Warning:[/bold yellow] {message}")

    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"[bold blue]ℹ️ Info:[/bold blue] {message}")

    def create_keyboard_shortcuts_help(self) -> Panel:
        """Create keyboard shortcuts help panel"""
        shortcuts = [
            ("Ctrl+N", "New Project"),
            ("Ctrl+O", "Open Project"),
            ("Ctrl+S", "Save/Settings"),
            ("Ctrl+A", "AI Assistant"),
            ("Ctrl+T", "Code Tools"),
            ("Ctrl+P", "Project Analytics"),
            ("Ctrl+M", "Model Manager"),
            ("Ctrl+I", "API Manager"),
            ("Ctrl+D", "Documentation"),
            ("Ctrl+E", "Security Scanner"),
            ("Ctrl+Y", "Deploy Assistant"),
            ("F1", "Help"),
            ("Ctrl+Q", "Quit"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Ctrl+F", "Find"),
            ("Ctrl+R", "Replace"),
            ("F5", "Run/Refresh"),
            ("F9", "Debug"),
            ("F11", "Fullscreen")
        ]

        table = Table(title="⌨️ Keyboard Shortcuts", show_header=True)
        table.add_column("Shortcut", style="cyan")
        table.add_column("Action", style="white")

        for shortcut, action in shortcuts:
            table.add_row(shortcut, action)

        return Panel(table, border_style="bright_cyan")

    def create_live_dashboard(self, update_callback: Callable) -> Live:
        """Create live updating dashboard"""
        def generate_content():
            return update_callback()

        return Live(
            generate_content(),
            refresh_per_second=1,
            console=self.console
        )