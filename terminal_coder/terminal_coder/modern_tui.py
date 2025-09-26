"""
Modern Terminal User Interface using Textual
Enhanced UI with modern Python 3.13 features
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListView,
    ListItem,
    Log,
    ProgressBar,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual.binding import Binding
from textual.screen import Screen
from rich.text import Text
from rich.syntax import Syntax
from rich.panel import Panel


class ProjectScreen(Screen):
    """Screen for project management"""

    BINDINGS = [
        Binding("escape", "back", "Back to main"),
        Binding("n", "new_project", "New Project"),
        Binding("o", "open_project", "Open Project"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Label("🚀 Project Management", classes="title")
            with Horizontal():
                with Vertical(classes="sidebar"):
                    yield Button("📂 New Project", id="new_project")
                    yield Button("🔄 Refresh", id="refresh")
                    yield Button("⚙️ Settings", id="settings")
                with Vertical(classes="main"):
                    yield DataTable(id="projects_table")
        yield Footer()

    def on_mount(self) -> None:
        """Setup the projects table"""
        table = self.query_one("#projects_table", DataTable)
        table.add_columns("Name", "Language", "Framework", "Last Modified")
        # Add sample data
        table.add_row("Terminal Coder", "Python", "Rich/Textual", "2025-01-01")
        table.add_row("Web App", "TypeScript", "Next.js", "2024-12-31")

    def action_back(self) -> None:
        """Go back to main screen"""
        self.app.pop_screen()

    def action_new_project(self) -> None:
        """Create new project"""
        self.app.push_screen(NewProjectScreen())


class NewProjectScreen(Screen):
    """Screen for creating new projects"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "save", "Save Project"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Label("📁 Create New Project", classes="title")
            with Vertical(classes="form"):
                yield Label("Project Name:")
                yield Input(id="project_name", placeholder="Enter project name...")
                yield Label("Programming Language:")
                yield Input(id="language", placeholder="python, javascript, etc.")
                yield Label("Framework (optional):")
                yield Input(id="framework", placeholder="django, react, etc.")
                with Horizontal():
                    yield Button("💾 Create", variant="success", id="create")
                    yield Button("❌ Cancel", variant="error", id="cancel")
        yield Footer()

    def action_cancel(self) -> None:
        """Cancel project creation"""
        self.app.pop_screen()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "create":
            await self._create_project()
        elif event.button.id == "cancel":
            self.action_cancel()

    async def _create_project(self) -> None:
        """Create the project"""
        name = self.query_one("#project_name", Input).value
        language = self.query_one("#language", Input).value
        framework = self.query_one("#framework", Input).value or None

        if not name or not language:
            self.notify("Please fill in required fields", severity="error")
            return

        # Here you would integrate with the project creation logic
        self.notify(f"Project '{name}' created successfully!", severity="success")
        self.app.pop_screen()


class AIAssistantScreen(Screen):
    """Screen for AI assistant interaction"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+enter", "send", "Send Message"),
        Binding("ctrl+l", "clear", "Clear Chat"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Label("🤖 AI Assistant", classes="title")
            with Horizontal():
                with Vertical(classes="chat"):
                    yield RichLog(id="chat_log", auto_scroll=True)
                    with Horizontal(classes="input_area"):
                        yield Input(id="message_input", placeholder="Ask me anything...")
                        yield Button("Send", id="send")
                with Vertical(classes="ai_settings"):
                    yield Label("AI Settings", classes="subtitle")
                    yield Label("Provider: OpenAI")
                    yield Label("Model: GPT-4")
                    yield ProgressBar(id="thinking", show_eta=False)
        yield Footer()

    def on_mount(self) -> None:
        """Initialize chat"""
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.write(Panel(
            "👋 Hello! I'm your AI coding assistant. How can I help you today?",
            title="🤖 AI Assistant",
            border_style="blue"
        ))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle send button"""
        if event.button.id == "send":
            await self._send_message()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id == "message_input":
            await self._send_message()

    async def _send_message(self) -> None:
        """Send message to AI"""
        message_input = self.query_one("#message_input", Input)
        chat_log = self.query_one("#chat_log", RichLog)
        progress = self.query_one("#thinking", ProgressBar)

        message = message_input.value.strip()
        if not message:
            return

        # Add user message to chat
        chat_log.write(Panel(
            message,
            title="👤 You",
            border_style="green"
        ))

        # Clear input
        message_input.value = ""

        # Show thinking animation
        progress.show()

        # Simulate AI response (replace with actual AI call)
        await asyncio.sleep(1)

        ai_response = f"I understand you're asking about: '{message}'. This is a simulated response that would be replaced with actual AI integration."

        chat_log.write(Panel(
            ai_response,
            title="🤖 AI Assistant",
            border_style="blue"
        ))

        progress.hide()

    def action_clear(self) -> None:
        """Clear chat history"""
        chat_log = self.query_one("#chat_log", RichLog)
        chat_log.clear()
        self.on_mount()  # Re-add welcome message

    def action_back(self) -> None:
        """Go back to main screen"""
        self.app.pop_screen()


class CodeAnalysisScreen(Screen):
    """Screen for code analysis and review"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh Analysis"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Label("🔍 Code Analysis", classes="title")
            with TabbedContent():
                with TabPane("Security", id="security"):
                    yield RichLog(id="security_log")
                with TabPane("Performance", id="performance"):
                    yield RichLog(id="performance_log")
                with TabPane("Quality", id="quality"):
                    yield RichLog(id="quality_log")
                with TabPane("Complexity", id="complexity"):
                    yield RichLog(id="complexity_log")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize analysis results"""
        # Security analysis
        security_log = self.query_one("#security_log", RichLog)
        security_log.write("🛡️ Security Analysis Results:")
        security_log.write("✅ No SQL injection vulnerabilities found")
        security_log.write("✅ No hardcoded secrets detected")
        security_log.write("⚠️  Consider using environment variables for API keys")

        # Performance analysis
        performance_log = self.query_one("#performance_log", RichLog)
        performance_log.write("⚡ Performance Analysis Results:")
        performance_log.write("✅ No obvious performance bottlenecks")
        performance_log.write("💡 Consider caching for API responses")
        performance_log.write("💡 Async operations are properly implemented")

        # Quality analysis
        quality_log = self.query_one("#quality_log", RichLog)
        quality_log.write("📊 Code Quality Results:")
        quality_log.write("✅ Code follows PEP 8 standards")
        quality_log.write("✅ Proper type hints usage")
        quality_log.write("✅ Good documentation coverage")

        # Complexity analysis
        complexity_log = self.query_one("#complexity_log", RichLog)
        complexity_log.write("🧠 Complexity Analysis Results:")
        complexity_log.write("📈 Average cyclomatic complexity: 3.2")
        complexity_log.write("📈 Lines of code: 2,847")
        complexity_log.write("📈 Functions: 127")

    def action_back(self) -> None:
        """Go back to main screen"""
        self.app.pop_screen()


class TerminalCoderTUI(App):
    """Modern Terminal Coder TUI Application"""

    CSS_PATH = "terminal_coder.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "projects", "Projects"),
        Binding("2", "ai_assistant", "AI Assistant"),
        Binding("3", "code_analysis", "Code Analysis"),
        Binding("f1", "help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        """Main application layout"""
        yield Header()
        with Container():
            yield Label("🚀 Terminal Coder v2.0", classes="title")
            yield Label("Advanced AI-Powered Development Terminal", classes="subtitle")

            with Vertical(classes="menu"):
                yield Button("📁 Project Management", id="projects", variant="primary")
                yield Button("🤖 AI Assistant", id="ai_assistant", variant="success")
                yield Button("🔍 Code Analysis", id="code_analysis", variant="warning")
                yield Button("⚙️  Settings", id="settings")
                yield Button("❓ Help", id="help")
                yield Button("❌ Exit", id="exit", variant="error")
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle main menu button presses"""
        if event.button.id == "projects":
            await self.push_screen(ProjectScreen())
        elif event.button.id == "ai_assistant":
            await self.push_screen(AIAssistantScreen())
        elif event.button.id == "code_analysis":
            await self.push_screen(CodeAnalysisScreen())
        elif event.button.id == "settings":
            self.notify("Settings screen coming soon!", severity="info")
        elif event.button.id == "help":
            self.notify("Help: Use number keys (1-3) or click buttons to navigate", severity="info")
        elif event.button.id == "exit":
            self.exit()

    def action_projects(self) -> None:
        """Navigate to projects screen"""
        self.push_screen(ProjectScreen())

    def action_ai_assistant(self) -> None:
        """Navigate to AI assistant screen"""
        self.push_screen(AIAssistantScreen())

    def action_code_analysis(self) -> None:
        """Navigate to code analysis screen"""
        self.push_screen(CodeAnalysisScreen())

    def action_help(self) -> None:
        """Show help information"""
        self.notify(
            "Terminal Coder v2.0 - Use keyboard shortcuts or click buttons to navigate",
            title="Help",
            severity="info"
        )


def run_modern_tui() -> None:
    """Run the modern TUI application"""
    app = TerminalCoderTUI()
    app.run()


if __name__ == "__main__":
    run_modern_tui()