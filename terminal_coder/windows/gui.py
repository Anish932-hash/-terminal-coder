#!/usr/bin/env python3
"""
Windows GUI Components
Advanced graphical user interface for Terminal Coder on Windows
"""

import asyncio
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import json
import subprocess

# Windows-specific imports
import win32api
import win32con
import win32gui
from PIL import Image, ImageTk

from rich.console import Console

# Import advanced GUI extensions
try:
    from .advanced_gui_extensions import (
        WindowsRealTimeSystemMonitor,
        WindowsAdvancedVisualizationPanel,
        WindowsIntelligentCodeAnalyzer,
        WindowsSmartContainerOrchestrator
    )
    ADVANCED_EXTENSIONS_AVAILABLE = True
except ImportError:
    ADVANCED_EXTENSIONS_AVAILABLE = False


@dataclass
class GUITheme:
    """GUI theme configuration"""
    bg_color: str = "#2b2b2b"
    fg_color: str = "#ffffff"
    accent_color: str = "#0078d4"
    secondary_color: str = "#404040"
    success_color: str = "#107c10"
    warning_color: str = "#ff8c00"
    error_color: str = "#d13438"


class WindowsGUI:
    """Advanced Windows GUI for Terminal Coder"""

    def __init__(self):
        self.root = None
        self.console = Console()
        self.theme = GUITheme()
        self.is_running = False

        # GUI components
        self.main_window = None
        self.notebook = None
        self.status_bar = None

        # Advanced extensions
        self.system_monitor = None
        self.visualization_panel = None
        self.code_analyzer = None
        self.container_orchestrator = None

        # Callbacks
        self.callbacks = {}

    def initialize(self) -> bool:
        """Initialize the GUI"""
        try:
            self.root = tk.Tk()
            self._setup_root_window()
            self._apply_windows_styling()
            self._create_main_interface()
            return True
        except Exception as e:
            self.console.print(f"[red]Error initializing GUI: {e}[/red]")
            return False

    def _setup_root_window(self):
        """Setup the main root window"""
        self.root.title("Terminal Coder - Windows Edition")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Set window icon (if available)
        try:
            # You would set an actual icon file here
            pass
        except Exception:
            pass

        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")

        # Configure close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _apply_windows_styling(self):
        """Apply Windows-specific styling"""
        # Configure ttk styles
        style = ttk.Style()

        # Use Windows 10/11 theme if available
        try:
            style.theme_use('winnative')
        except tk.TclError:
            style.theme_use('clam')

        # Configure custom styles
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Code.TText', font=('Consolas', 10))

        # Configure colors for dark theme
        style.configure('TFrame', background=self.theme.bg_color)
        style.configure('TLabel', background=self.theme.bg_color, foreground=self.theme.fg_color)
        style.configure('TButton', font=('Segoe UI', 9))

    def _create_main_interface(self):
        """Create the main interface"""
        # Create menu bar
        self._create_menu_bar()

        # Create toolbar
        self._create_toolbar()

        # Create main content area with notebook
        self._create_notebook()

        # Create status bar
        self._create_status_bar()

        # Initialize advanced extensions
        self._initialize_advanced_extensions()

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)  # Notebook row

    def _create_menu_bar(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", accelerator="Ctrl+N", command=self._new_project)
        file_menu.add_command(label="Open Project", accelerator="Ctrl+O", command=self._open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Settings", accelerator="Ctrl+,", command=self._show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", accelerator="Alt+F4", command=self._on_closing)

        # AI menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_command(label="AI Assistant", accelerator="Ctrl+A", command=self._open_ai_assistant)
        ai_menu.add_command(label="Configure Providers", command=self._configure_ai_providers)
        ai_menu.add_command(label="Test Connections", command=self._test_ai_connections)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Code Analysis", command=self._code_analysis)
        tools_menu.add_command(label="Security Scanner", command=self._security_scanner)
        tools_menu.add_command(label="Performance Monitor", command=self._performance_monitor)

        # Windows menu
        windows_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Windows", menu=windows_menu)
        windows_menu.add_command(label="System Information", command=self._system_information)
        windows_menu.add_command(label="Services Manager", command=self._services_manager)
        windows_menu.add_command(label="Registry Tools", command=self._registry_tools)
        windows_menu.add_command(label="PowerShell Console", command=self._powershell_console)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_toolbar(self):
        """Create the toolbar"""
        toolbar = ttk.Frame(self.root)
        toolbar.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Toolbar buttons
        ttk.Button(toolbar, text="New", command=self._new_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Open", command=self._open_project).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        ttk.Button(toolbar, text="AI Assistant", command=self._open_ai_assistant).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Analyze", command=self._code_analysis).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        ttk.Button(toolbar, text="Terminal", command=self._open_terminal).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Explorer", command=self._open_explorer).pack(side=tk.LEFT, padx=2)

    def _create_notebook(self):
        """Create the main notebook (tabbed interface)"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Dashboard tab
        self._create_dashboard_tab()

        # Projects tab
        self._create_projects_tab()

        # AI Assistant tab
        self._create_ai_assistant_tab()

        # System Monitor tab
        self._create_system_monitor_tab()

        # Settings tab
        self._create_settings_tab()

    def _create_dashboard_tab(self):
        """Create the dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")

        # Welcome panel
        welcome_frame = ttk.LabelFrame(dashboard_frame, text="Welcome to Terminal Coder", padding=10)
        welcome_frame.pack(fill=tk.X, padx=10, pady=10)

        welcome_text = """
Terminal Coder - Windows Edition
Advanced AI-Powered Development Environment

🚀 Create new projects with intelligent templates
🤖 AI-powered code assistance and analysis
🖥️ Windows system integration and tools
📊 Real-time performance monitoring
🛡️ Security scanning and compliance
        """

        ttk.Label(welcome_frame, text=welcome_text, font=('Segoe UI', 10)).pack()

        # Quick actions panel
        actions_frame = ttk.LabelFrame(dashboard_frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill=tk.X, padx=10, pady=10)

        actions_inner = ttk.Frame(actions_frame)
        actions_inner.pack()

        # Action buttons in grid
        ttk.Button(actions_inner, text="🆕 New Project",
                  command=self._new_project, width=20).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(actions_inner, text="📂 Open Project",
                  command=self._open_project, width=20).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(actions_inner, text="🤖 AI Assistant",
                  command=self._open_ai_assistant, width=20).grid(row=0, column=2, padx=5, pady=5)

        ttk.Button(actions_inner, text="🔍 Code Analysis",
                  command=self._code_analysis, width=20).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(actions_inner, text="⚙️ System Tools",
                  command=self._system_information, width=20).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(actions_inner, text="🛡️ Security Scan",
                  command=self._security_scanner, width=20).grid(row=1, column=2, padx=5, pady=5)

        # Recent projects panel
        recent_frame = ttk.LabelFrame(dashboard_frame, text="Recent Projects", padding=10)
        recent_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Recent projects list
        self.recent_projects_tree = ttk.Treeview(recent_frame, columns=('Language', 'Modified'), show='tree headings')
        self.recent_projects_tree.heading('#0', text='Project')
        self.recent_projects_tree.heading('Language', text='Language')
        self.recent_projects_tree.heading('Modified', text='Last Modified')
        self.recent_projects_tree.pack(fill=tk.BOTH, expand=True)

        # Bind double-click to open project
        self.recent_projects_tree.bind('<Double-1>', self._on_recent_project_double_click)

    def _create_projects_tab(self):
        """Create the projects management tab"""
        projects_frame = ttk.Frame(self.notebook)
        self.notebook.add(projects_frame, text="Projects")

        # Projects toolbar
        projects_toolbar = ttk.Frame(projects_frame)
        projects_toolbar.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(projects_toolbar, text="New Project",
                  command=self._new_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(projects_toolbar, text="Open Project",
                  command=self._open_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(projects_toolbar, text="Delete Project",
                  command=self._delete_project).pack(side=tk.LEFT, padx=2)

        ttk.Separator(projects_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        ttk.Button(projects_toolbar, text="Refresh",
                  command=self._refresh_projects).pack(side=tk.LEFT, padx=2)

        # Projects list
        projects_list_frame = ttk.Frame(projects_frame)
        projects_list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.projects_tree = ttk.Treeview(projects_list_frame,
                                         columns=('Language', 'Framework', 'Created', 'Path'),
                                         show='tree headings')

        self.projects_tree.heading('#0', text='Name')
        self.projects_tree.heading('Language', text='Language')
        self.projects_tree.heading('Framework', text='Framework')
        self.projects_tree.heading('Created', text='Created')
        self.projects_tree.heading('Path', text='Path')

        # Configure column widths
        self.projects_tree.column('#0', width=200)
        self.projects_tree.column('Language', width=100)
        self.projects_tree.column('Framework', width=120)
        self.projects_tree.column('Created', width=120)
        self.projects_tree.column('Path', width=300)

        # Scrollbars
        projects_scrolly = ttk.Scrollbar(projects_list_frame, orient=tk.VERTICAL,
                                        command=self.projects_tree.yview)
        projects_scrollx = ttk.Scrollbar(projects_list_frame, orient=tk.HORIZONTAL,
                                        command=self.projects_tree.xview)

        self.projects_tree.configure(yscrollcommand=projects_scrolly.set,
                                    xscrollcommand=projects_scrollx.set)

        # Pack treeview and scrollbars
        self.projects_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        projects_scrolly.grid(row=0, column=1, sticky=(tk.N, tk.S))
        projects_scrollx.grid(row=1, column=0, sticky=(tk.W, tk.E))

        projects_list_frame.columnconfigure(0, weight=1)
        projects_list_frame.rowconfigure(0, weight=1)

        # Context menu
        self._create_projects_context_menu()

    def _create_ai_assistant_tab(self):
        """Create the AI assistant tab"""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="AI Assistant")

        # AI provider selection
        provider_frame = ttk.LabelFrame(ai_frame, text="AI Provider", padding=5)
        provider_frame.pack(fill=tk.X, padx=5, pady=5)

        self.ai_provider_var = tk.StringVar(value="openai")
        self.ai_model_var = tk.StringVar(value="gpt-4")

        ttk.Label(provider_frame, text="Provider:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.provider_combo = ttk.Combobox(provider_frame, textvariable=self.ai_provider_var,
                                          values=["openai", "anthropic", "google", "cohere"])
        self.provider_combo.grid(row=0, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))

        ttk.Label(provider_frame, text="Model:").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        self.model_combo = ttk.Combobox(provider_frame, textvariable=self.ai_model_var)
        self.model_combo.grid(row=0, column=3, padx=5, pady=2, sticky=(tk.W, tk.E))

        ttk.Button(provider_frame, text="Configure",
                  command=self._configure_ai_providers).grid(row=0, column=4, padx=5, pady=2)

        provider_frame.columnconfigure(1, weight=1)
        provider_frame.columnconfigure(3, weight=1)

        # Conversation area
        conversation_frame = ttk.LabelFrame(ai_frame, text="Conversation", padding=5)
        conversation_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(conversation_frame,
                                                     height=20,
                                                     state=tk.DISABLED,
                                                     wrap=tk.WORD,
                                                     font=('Segoe UI', 10))
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Input area
        input_frame = ttk.Frame(conversation_frame)
        input_frame.pack(fill=tk.X)

        self.user_input = tk.Text(input_frame, height=3, wrap=tk.WORD, font=('Segoe UI', 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Send button
        send_button = ttk.Button(input_frame, text="Send", command=self._send_ai_message)
        send_button.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind Enter key
        self.user_input.bind('<Control-Return>', lambda e: self._send_ai_message())

    def _create_system_monitor_tab(self):
        """Create the system monitoring tab"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="System Monitor")

        # System info panel
        info_frame = ttk.LabelFrame(monitor_frame, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.system_info_text = tk.Text(info_frame, height=8, state=tk.DISABLED,
                                       font=('Consolas', 9))
        self.system_info_text.pack(fill=tk.X)

        # Performance monitoring
        perf_frame = ttk.LabelFrame(monitor_frame, text="Performance", padding=10)
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # CPU and Memory usage (simplified - would use actual monitoring)
        ttk.Label(perf_frame, text="CPU Usage:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.cpu_progress = ttk.Progressbar(perf_frame, length=200)
        self.cpu_progress.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.cpu_label = ttk.Label(perf_frame, text="0%")
        self.cpu_label.grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(perf_frame, text="Memory Usage:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.memory_progress = ttk.Progressbar(perf_frame, length=200)
        self.memory_progress.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.memory_label = ttk.Label(perf_frame, text="0%")
        self.memory_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        perf_frame.columnconfigure(1, weight=1)

        # Processes list
        processes_frame = ttk.LabelFrame(perf_frame, text="Top Processes", padding=5)
        processes_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.processes_tree = ttk.Treeview(processes_frame,
                                          columns=('PID', 'CPU', 'Memory'),
                                          show='tree headings', height=10)

        self.processes_tree.heading('#0', text='Name')
        self.processes_tree.heading('PID', text='PID')
        self.processes_tree.heading('CPU', text='CPU %')
        self.processes_tree.heading('Memory', text='Memory (MB)')

        self.processes_tree.pack(fill=tk.BOTH, expand=True)

        perf_frame.rowconfigure(2, weight=1)

    def _create_settings_tab(self):
        """Create the settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")

        # Create notebook for settings categories
        settings_notebook = ttk.Notebook(settings_frame)
        settings_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # General settings
        self._create_general_settings_tab(settings_notebook)

        # AI settings
        self._create_ai_settings_tab(settings_notebook)

        # Windows settings
        self._create_windows_settings_tab(settings_notebook)

    def _create_general_settings_tab(self, parent):
        """Create general settings tab"""
        general_frame = ttk.Frame(parent)
        parent.add(general_frame, text="General")

        # Theme settings
        theme_frame = ttk.LabelFrame(general_frame, text="Appearance", padding=10)
        theme_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(theme_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.theme_var = tk.StringVar(value="dark")
        theme_combo = ttk.Combobox(theme_frame, textvariable=self.theme_var,
                                  values=["dark", "light", "auto"])
        theme_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

        theme_frame.columnconfigure(1, weight=1)

    def _create_ai_settings_tab(self, parent):
        """Create AI settings tab"""
        ai_frame = ttk.Frame(parent)
        parent.add(ai_frame, text="AI")

        # API Keys section
        keys_frame = ttk.LabelFrame(ai_frame, text="API Keys", padding=10)
        keys_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(keys_frame, text="Configure API Keys",
                  command=self._configure_ai_providers).pack()

    def _create_windows_settings_tab(self, parent):
        """Create Windows-specific settings tab"""
        windows_frame = ttk.Frame(parent)
        parent.add(windows_frame, text="Windows")

        # Windows integration settings
        integration_frame = ttk.LabelFrame(windows_frame, text="Windows Integration", padding=10)
        integration_frame.pack(fill=tk.X, padx=5, pady=5)

        self.powershell_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(integration_frame, text="Use PowerShell integration",
                       variable=self.powershell_var).pack(anchor=tk.W, pady=2)

        self.wsl_var = tk.BooleanVar()
        ttk.Checkbutton(integration_frame, text="Enable WSL support",
                       variable=self.wsl_var).pack(anchor=tk.W, pady=2)

        self.registry_var = tk.BooleanVar()
        ttk.Checkbutton(integration_frame, text="Allow registry access",
                       variable=self.registry_var).pack(anchor=tk.W, pady=2)

    def _create_projects_context_menu(self):
        """Create context menu for projects tree"""
        self.projects_context_menu = tk.Menu(self.root, tearoff=0)
        self.projects_context_menu.add_command(label="Open", command=self._open_selected_project)
        self.projects_context_menu.add_command(label="Open in Explorer", command=self._open_project_in_explorer)
        self.projects_context_menu.add_separator()
        self.projects_context_menu.add_command(label="Delete", command=self._delete_selected_project)

        self.projects_tree.bind("<Button-3>", self._show_projects_context_menu)

    def _create_status_bar(self):
        """Create the status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)

        # Status label
        self.status_label = ttk.Label(self.status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT)

        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(self.status_bar, length=200)

        # AI status
        self.ai_status_label = ttk.Label(self.status_bar, text="AI: Not configured")
        self.ai_status_label.pack(side=tk.RIGHT, padx=5)

    # Event handlers and callbacks
    def _on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit Terminal Coder?"):
            self.is_running = False
            if self.root:
                self.root.quit()
                self.root.destroy()

    def _new_project(self):
        """Handle new project creation"""
        if 'create_project' in self.callbacks:
            threading.Thread(target=self.callbacks['create_project'], daemon=True).start()

    def _open_project(self):
        """Handle project opening"""
        if 'open_project' in self.callbacks:
            threading.Thread(target=self.callbacks['open_project'], daemon=True).start()

    def _open_ai_assistant(self):
        """Handle AI assistant opening"""
        if 'ai_assistant' in self.callbacks:
            threading.Thread(target=self.callbacks['ai_assistant'], daemon=True).start()

    def _configure_ai_providers(self):
        """Configure AI providers"""
        if 'configure_ai' in self.callbacks:
            threading.Thread(target=self.callbacks['configure_ai'], daemon=True).start()

    def _test_ai_connections(self):
        """Test AI connections"""
        if 'test_ai' in self.callbacks:
            threading.Thread(target=self.callbacks['test_ai'], daemon=True).start()

    def _code_analysis(self):
        """Perform code analysis"""
        if 'code_analysis' in self.callbacks:
            threading.Thread(target=self.callbacks['code_analysis'], daemon=True).start()

    def _security_scanner(self):
        """Run security scanner"""
        if 'security_scanner' in self.callbacks:
            threading.Thread(target=self.callbacks['security_scanner'], daemon=True).start()

    def _performance_monitor(self):
        """Open performance monitor"""
        pass  # Already in the GUI

    def _system_information(self):
        """Show system information"""
        if 'system_info' in self.callbacks:
            threading.Thread(target=self.callbacks['system_info'], daemon=True).start()

    def _services_manager(self):
        """Open services manager"""
        if 'services_manager' in self.callbacks:
            threading.Thread(target=self.callbacks['services_manager'], daemon=True).start()

    def _registry_tools(self):
        """Open registry tools"""
        if 'registry_tools' in self.callbacks:
            threading.Thread(target=self.callbacks['registry_tools'], daemon=True).start()

    def _powershell_console(self):
        """Open PowerShell console"""
        try:
            subprocess.Popen(['powershell'], shell=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open PowerShell: {e}")

    def _open_terminal(self):
        """Open terminal"""
        try:
            subprocess.Popen(['cmd'], shell=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open terminal: {e}")

    def _open_explorer(self):
        """Open Windows Explorer"""
        try:
            subprocess.Popen(['explorer'], shell=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open Explorer: {e}")

    def _show_settings(self):
        """Show settings (switch to settings tab)"""
        self.notebook.select(4)  # Settings tab index

    def _show_documentation(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", "Documentation will be available in future versions.")

    def _show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
Keyboard Shortcuts:

Ctrl+N - New Project
Ctrl+O - Open Project
Ctrl+A - AI Assistant
Ctrl+, - Settings
Ctrl+Enter - Send AI message

Alt+F4 - Exit
F1 - Help
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def _show_about(self):
        """Show about dialog"""
        about_text = """
Terminal Coder - Windows Edition
Version 2.0.0

Advanced AI-Powered Development Environment
Optimized for Windows

© 2024 Terminal Coder Team
        """
        messagebox.showinfo("About Terminal Coder", about_text)

    def _send_ai_message(self):
        """Send message to AI"""
        message = self.user_input.get(1.0, tk.END).strip()
        if not message:
            return

        # Clear input
        self.user_input.delete(1.0, tk.END)

        # Add message to chat display
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"You: {message}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

        # Send to AI (via callback)
        if 'send_ai_message' in self.callbacks:
            threading.Thread(
                target=self.callbacks['send_ai_message'],
                args=(message, self._on_ai_response),
                daemon=True
            ).start()

    def _on_ai_response(self, response: str):
        """Handle AI response"""
        self.root.after(0, lambda: self._display_ai_response(response))

    def _display_ai_response(self, response: str):
        """Display AI response in chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"AI: {response}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _on_recent_project_double_click(self, event):
        """Handle double-click on recent project"""
        selection = self.recent_projects_tree.selection()
        if selection and 'open_project_by_name' in self.callbacks:
            item = self.recent_projects_tree.item(selection[0])
            project_name = item['text']
            threading.Thread(
                target=self.callbacks['open_project_by_name'],
                args=(project_name,),
                daemon=True
            ).start()

    def _refresh_projects(self):
        """Refresh projects list"""
        if 'refresh_projects' in self.callbacks:
            self.callbacks['refresh_projects']()

    def _delete_project(self):
        """Delete selected project"""
        selection = self.projects_tree.selection()
        if selection:
            item = self.projects_tree.item(selection[0])
            project_name = item['text']

            if messagebox.askyesno("Confirm Delete",
                                 f"Are you sure you want to delete project '{project_name}'?"):
                if 'delete_project' in self.callbacks:
                    threading.Thread(
                        target=self.callbacks['delete_project'],
                        args=(project_name,),
                        daemon=True
                    ).start()

    def _show_projects_context_menu(self, event):
        """Show projects context menu"""
        try:
            self.projects_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.projects_context_menu.grab_release()

    def _open_selected_project(self):
        """Open selected project"""
        selection = self.projects_tree.selection()
        if selection and 'open_project_by_name' in self.callbacks:
            item = self.projects_tree.item(selection[0])
            project_name = item['text']
            threading.Thread(
                target=self.callbacks['open_project_by_name'],
                args=(project_name,),
                daemon=True
            ).start()

    def _open_project_in_explorer(self):
        """Open project in Explorer"""
        selection = self.projects_tree.selection()
        if selection:
            item = self.projects_tree.item(selection[0])
            project_path = item['values'][3]  # Path column
            try:
                subprocess.Popen(['explorer', project_path])
            except Exception as e:
                messagebox.showerror("Error", f"Could not open Explorer: {e}")

    def _delete_selected_project(self):
        """Delete selected project (same as _delete_project)"""
        self._delete_project()

    # Public methods for external use
    def run(self):
        """Run the GUI main loop"""
        if not self.root:
            if not self.initialize():
                return False

        self.is_running = True
        self.root.mainloop()
        return True

    def set_callback(self, event: str, callback: Callable):
        """Set callback function for events"""
        self.callbacks[event] = callback

    def update_status(self, message: str):
        """Update status bar message"""
        if self.status_label:
            self.root.after(0, lambda: self.status_label.config(text=message))

    def show_progress(self, show: bool = True):
        """Show/hide progress bar"""
        if show:
            self.progress_bar.pack(side=tk.RIGHT, padx=5)
        else:
            self.progress_bar.pack_forget()

    def update_progress(self, value: int):
        """Update progress bar value (0-100)"""
        if self.progress_bar:
            self.root.after(0, lambda: self.progress_bar.config(value=value))

    def update_ai_status(self, status: str):
        """Update AI status in status bar"""
        if self.ai_status_label:
            self.root.after(0, lambda: self.ai_status_label.config(text=f"AI: {status}"))

    def update_projects_list(self, projects: list):
        """Update projects list in GUI"""
        def _update():
            # Clear existing items
            for item in self.projects_tree.get_children():
                self.projects_tree.delete(item)

            # Add projects
            for project in projects:
                self.projects_tree.insert('', tk.END, text=project['name'],
                                        values=(project['language'],
                                               project.get('framework', 'None'),
                                               project['created_at'][:10],  # Date only
                                               project['path']))

        self.root.after(0, _update)

    def update_recent_projects(self, projects: list):
        """Update recent projects list"""
        def _update():
            # Clear existing items
            for item in self.recent_projects_tree.get_children():
                self.recent_projects_tree.delete(item)

            # Add recent projects (limit to 10)
            for project in projects[:10]:
                self.recent_projects_tree.insert('', tk.END, text=project['name'],
                                                values=(project['language'],
                                                       project['last_modified'][:16]))  # Date + time

        self.root.after(0, _update)

    def show_message(self, title: str, message: str, msg_type: str = "info"):
        """Show message dialog"""
        def _show():
            if msg_type == "error":
                messagebox.showerror(title, message)
            elif msg_type == "warning":
                messagebox.showwarning(title, message)
            else:
                messagebox.showinfo(title, message)

        self.root.after(0, _show)

    def update_system_info(self, info: str):
        """Update system information display"""
        def _update():
            self.system_info_text.config(state=tk.NORMAL)
            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(1.0, info)
            self.system_info_text.config(state=tk.DISABLED)

        self.root.after(0, _update)

    def update_performance_data(self, cpu_percent: float, memory_percent: float):
        """Update performance monitoring data"""
        def _update():
            self.cpu_progress['value'] = cpu_percent
            self.cpu_label.config(text=f"{cpu_percent:.1f}%")

            self.memory_progress['value'] = memory_percent
            self.memory_label.config(text=f"{memory_percent:.1f}%")

        self.root.after(0, _update)

    def _initialize_advanced_extensions(self):
        """Initialize advanced GUI extensions"""
        if not ADVANCED_EXTENSIONS_AVAILABLE:
            self.console.print("[yellow]Advanced GUI extensions not available[/yellow]")
            return

        try:
            # Initialize system monitor
            self.system_monitor = WindowsRealTimeSystemMonitor(self)
            self.console.print("[green]Windows System Monitor initialized[/green]")

            # Initialize code analyzer
            self.code_analyzer = WindowsIntelligentCodeAnalyzer()
            self.console.print("[green]Windows Code Analyzer initialized[/green]")

            # Initialize container orchestrator
            self.container_orchestrator = WindowsSmartContainerOrchestrator()
            self.console.print("[green]Windows Container Orchestrator initialized[/green]")

            # Add advanced visualization tabs if system monitor tab exists
            self._add_advanced_visualization_tabs()

        except Exception as e:
            self.console.print(f"[red]Error initializing advanced extensions: {e}[/red]")

    def _add_advanced_visualization_tabs(self):
        """Add advanced visualization tabs to the notebook"""
        if not ADVANCED_EXTENSIONS_AVAILABLE or not hasattr(self, 'notebook'):
            return

        try:
            # Create advanced system monitoring tab
            self.advanced_monitor_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.advanced_monitor_frame, text="📊 Advanced Monitor")

            # Initialize visualization panel for advanced monitoring
            if self.system_monitor:
                self.visualization_panel = WindowsAdvancedVisualizationPanel(self.advanced_monitor_frame)

            # Create advanced code analysis tab
            self.advanced_analysis_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.advanced_analysis_frame, text="🤖 Code Intelligence")

            # Create container management tab
            self.container_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.container_frame, text="🐳 Containers")

            self.console.print("[green]Advanced visualization tabs added[/green]")

        except Exception as e:
            self.console.print(f"[red]Error adding visualization tabs: {e}[/red]")

    def start_advanced_monitoring(self):
        """Start advanced system monitoring"""
        if self.system_monitor and self.visualization_panel:
            def monitoring_update_callback(metrics, anomalies):
                """Callback for monitoring updates"""
                try:
                    # Update visualization panel with new data
                    self.visualization_panel.update_realtime_charts(metrics, anomalies)

                    # Show notifications for critical anomalies
                    for anomaly in anomalies:
                        if anomaly.get('severity') == 'critical':
                            self._show_windows_notification(
                                "System Alert",
                                anomaly.get('message', 'Critical system anomaly detected')
                            )

                except Exception as e:
                    self.console.print(f"[red]Monitoring update error: {e}[/red]")

            # Start monitoring in separate thread
            asyncio.run_coroutine_threadsafe(
                self.system_monitor.start_monitoring(monitoring_update_callback),
                asyncio.new_event_loop()
            )

    def stop_advanced_monitoring(self):
        """Stop advanced system monitoring"""
        if self.system_monitor:
            self.system_monitor.stop_monitoring()

    def _show_windows_notification(self, title: str, message: str):
        """Show Windows native notification"""
        try:
            import win32api
            import win32con

            # Show balloon tooltip notification
            win32api.MessageBox(0, message, title, win32con.MB_ICONINFORMATION)

        except Exception as e:
            # Fallback to tkinter messagebox
            self.show_message(title, message, "info")

    def analyze_project_with_ai(self, project_path: str):
        """Analyze project using advanced AI-powered code analyzer"""
        if not self.code_analyzer:
            self.show_message("Code Analysis", "Advanced code analyzer not available", "error")
            return

        def analyze_thread():
            try:
                self.update_status("Analyzing project with AI...")

                # Perform comprehensive Windows project analysis
                analysis_results = asyncio.run(
                    self.code_analyzer.analyze_windows_project(project_path)
                )

                # Display results in a new window
                self._show_analysis_results(analysis_results)

                self.update_status("Ready")

            except Exception as e:
                self.show_message("Analysis Error", f"Failed to analyze project: {e}", "error")
                self.update_status("Ready")

        threading.Thread(target=analyze_thread, daemon=True).start()

    def _show_analysis_results(self, results: Dict[str, Any]):
        """Show code analysis results in a dedicated window"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Windows Project Analysis Results")
        results_window.geometry("1000x800")
        results_window.resizable(True, True)

        # Create notebook for different result categories
        results_notebook = ttk.Notebook(results_window)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Summary tab
        self._create_summary_tab(results_notebook, results.get('summary', {}))

        # Windows-specific tab
        self._create_windows_analysis_tab(results_notebook, results.get('windows_specific', {}))

        # Security tab
        self._create_security_tab(results_notebook, results.get('security', {}))

        # Recommendations tab
        self._create_recommendations_tab(results_notebook, results.get('recommendations', []))

    def _create_summary_tab(self, parent, summary_data):
        """Create summary tab for analysis results"""
        summary_frame = ttk.Frame(parent)
        parent.add(summary_frame, text="📊 Summary")

        summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD, font=('Consolas', 10))
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Format summary data
        summary_content = f"""
PROJECT ANALYSIS SUMMARY
========================

Total Files: {summary_data.get('total_files', 0)}
Total Lines: {summary_data.get('total_lines', 0)}
Windows-Specific Files: {summary_data.get('windows_specific_files', 0)}

Languages Distribution:
{self._format_dict(summary_data.get('languages', {}))}

Largest Files:
{self._format_largest_files(summary_data.get('largest_files', []))}
        """

        summary_text.insert(tk.END, summary_content)
        summary_text.config(state=tk.DISABLED)

    def _create_windows_analysis_tab(self, parent, windows_data):
        """Create Windows-specific analysis tab"""
        windows_frame = ttk.Frame(parent)
        parent.add(windows_frame, text="🪟 Windows Analysis")

        windows_text = scrolledtext.ScrolledText(windows_frame, wrap=tk.WORD, font=('Consolas', 10))
        windows_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        windows_content = f"""
WINDOWS-SPECIFIC ANALYSIS
=========================

Win32 API Usage: {len(windows_data.get('win32_api_usage', []))} occurrences
Registry Operations: {len(windows_data.get('registry_operations', []))} occurrences
COM Objects: {len(windows_data.get('com_objects', []))} occurrences
Service Interactions: {len(windows_data.get('service_interactions', []))} occurrences
PowerShell Usage: {len(windows_data.get('powershell_usage', []))} occurrences

Compatibility Issues:
{self._format_compatibility_issues(windows_data.get('compatibility_issues', []))}

Windows Paths Found:
{self._format_windows_paths(windows_data.get('windows_paths', []))}
        """

        windows_text.insert(tk.END, windows_content)
        windows_text.config(state=tk.DISABLED)

    def _create_security_tab(self, parent, security_data):
        """Create security analysis tab"""
        security_frame = ttk.Frame(parent)
        parent.add(security_frame, text="🛡️ Security")

        security_text = scrolledtext.ScrolledText(security_frame, wrap=tk.WORD, font=('Consolas', 10))
        security_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        security_content = f"""
SECURITY ANALYSIS
=================

Security Score: {security_data.get('security_score', 0)}/100

Potential Issues: {len(security_data.get('potential_issues', []))}
Windows Security Issues: {len(security_data.get('windows_security_issues', []))}
Sensitive Files: {len(security_data.get('sensitive_files', []))}

Windows-Specific Security Issues:
{self._format_security_issues(security_data.get('windows_security_issues', []))}

General Security Issues:
{self._format_security_issues(security_data.get('potential_issues', []))}
        """

        security_text.insert(tk.END, security_content)
        security_text.config(state=tk.DISABLED)

    def _create_recommendations_tab(self, parent, recommendations):
        """Create recommendations tab"""
        rec_frame = ttk.Frame(parent)
        parent.add(rec_frame, text="💡 Recommendations")

        rec_text = scrolledtext.ScrolledText(rec_frame, wrap=tk.WORD, font=('Consolas', 10))
        rec_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        rec_content = "RECOMMENDATIONS\n===============\n\n"

        for i, rec in enumerate(recommendations, 1):
            priority_emoji = "🔴" if rec.get('priority') == 'critical' else "🟡" if rec.get('priority') == 'high' else "🟢"
            rec_content += f"{i}. {priority_emoji} {rec.get('title', 'Unknown')}\n"
            rec_content += f"   Type: {rec.get('type', 'Unknown')}\n"
            rec_content += f"   Priority: {rec.get('priority', 'Unknown')}\n"
            rec_content += f"   Description: {rec.get('description', 'No description')}\n"
            rec_content += f"   Action: {rec.get('action', 'No action specified')}\n\n"

        rec_text.insert(tk.END, rec_content)
        rec_text.config(state=tk.DISABLED)

    def _format_dict(self, data_dict):
        """Format dictionary for display"""
        return '\n'.join([f"  {k}: {v}" for k, v in data_dict.items()])

    def _format_largest_files(self, files_list):
        """Format largest files list for display"""
        return '\n'.join([f"  {Path(f[0]).name}: {f[1]} lines, {f[2]/1024:.1f} KB" for f in files_list[:5]])

    def _format_compatibility_issues(self, issues_list):
        """Format compatibility issues for display"""
        return '\n'.join([f"  {Path(issue['file']).name}:{issue['line']} - {issue['issue']}" for issue in issues_list[:10]])

    def _format_windows_paths(self, paths_list):
        """Format Windows paths for display"""
        return '\n'.join([f"  {Path(path['file']).name}:{path['line']} - {path['path']}" for path in paths_list[:10]])

    def _format_security_issues(self, issues_list):
        """Format security issues for display"""
        return '\n'.join([f"  {Path(issue['file']).name}:{issue['line']} - {issue['issue']}" for issue in issues_list[:10]])

    def manage_containers(self):
        """Open container management interface"""
        if not self.container_orchestrator:
            self.show_message("Container Management", "Container orchestrator not available", "error")
            return

        def container_analysis_thread():
            try:
                self.update_status("Analyzing container infrastructure...")

                # Perform container analysis
                container_analysis = asyncio.run(
                    self.container_orchestrator.analyze_windows_container_infrastructure()
                )

                # Display results
                self._show_container_results(container_analysis)

                self.update_status("Ready")

            except Exception as e:
                self.show_message("Container Analysis Error", f"Failed to analyze containers: {e}", "error")
                self.update_status("Ready")

        threading.Thread(target=container_analysis_thread, daemon=True).start()

    def _show_container_results(self, results: Dict[str, Any]):
        """Show container analysis results"""
        container_window = tk.Toplevel(self.root)
        container_window.title("Windows Container Analysis")
        container_window.geometry("1200x800")

        container_notebook = ttk.Notebook(container_window)
        container_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Docker tab
        docker_frame = ttk.Frame(container_notebook)
        container_notebook.add(docker_frame, text="🐳 Docker")

        docker_text = scrolledtext.ScrolledText(docker_frame, wrap=tk.WORD, font=('Consolas', 10))
        docker_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        docker_data = results.get('docker', {})
        system_info = docker_data.get('system_info', {})

        docker_content = f"""
DOCKER ON WINDOWS ANALYSIS
===========================

System Information:
  Docker Version: {system_info.get('server_version', 'Unknown')}
  Operating System: {system_info.get('operating_system', 'Unknown')}
  OS Type: {system_info.get('os_type', 'Unknown')}
  Architecture: {system_info.get('architecture', 'Unknown')}
  Windows Version: {system_info.get('windows_version', 'Unknown')}
  Memory Limit: {system_info.get('memory_limit', 0) / 1024 / 1024 / 1024:.1f} GB
  CPU Count: {system_info.get('cpu_count', 0)}

Containers:
  Running: {system_info.get('containers_running', 0)}
  Stopped: {system_info.get('containers_stopped', 0)}
  Total Images: {system_info.get('images', 0)}

Container Details:
{self._format_container_list(docker_data.get('containers', []))}

Windows Container Information:
{self._format_windows_containers(results.get('windows_containers', {}))}
        """

        docker_text.insert(tk.END, docker_content)
        docker_text.config(state=tk.DISABLED)

        # Recommendations tab
        rec_frame = ttk.Frame(container_notebook)
        container_notebook.add(rec_frame, text="💡 Recommendations")

        rec_text = scrolledtext.ScrolledText(rec_frame, wrap=tk.WORD, font=('Consolas', 10))
        rec_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        recommendations = results.get('recommendations', [])
        rec_content = "CONTAINER RECOMMENDATIONS\n========================\n\n"

        for i, rec in enumerate(recommendations, 1):
            priority_emoji = "🔴" if rec.get('priority') == 'critical' else "🟡" if rec.get('priority') == 'high' else "🟢"
            rec_content += f"{i}. {priority_emoji} {rec.get('title', 'Unknown')}\n"
            rec_content += f"   Description: {rec.get('description', 'No description')}\n"
            rec_content += f"   Action: {rec.get('action', 'No action specified')}\n\n"

        rec_text.insert(tk.END, rec_content)
        rec_text.config(state=tk.DISABLED)

    def _format_container_list(self, containers_list):
        """Format container list for display"""
        if not containers_list:
            return "  No containers found"

        result = ""
        for container in containers_list[:10]:  # Show first 10
            result += f"  {container.get('name', 'Unknown')}: {container.get('status', 'Unknown')} "
            result += f"({container.get('image', 'Unknown')})\n"
            if container.get('isolation'):
                result += f"    Isolation: {container['isolation']}\n"
            if container.get('cpu_percent'):
                result += f"    CPU: {container['cpu_percent']}%, Memory: {container.get('memory_usage_mb', 0):.1f} MB\n"
        return result

    def _format_windows_containers(self, windows_data):
        """Format Windows containers data for display"""
        if not windows_data:
            return "  No Windows-specific data available"

        result = f"Isolation Modes:\n"
        for mode, count in windows_data.get('isolation_modes', {}).items():
            result += f"  {mode}: {count}\n"

        result += f"\nBase Images:\n"
        for image, count in windows_data.get('base_images', {}).items():
            result += f"  {image}: {count}\n"

        return result

    def setup_ai_callbacks(self, advanced_ai):
        """Setup AI system callbacks for Windows GUI integration"""
        self.advanced_ai = advanced_ai

        # Set up callback functions for the Windows GUI to use
        self.set_callback('create_project', self._create_project_with_ai)
        self.set_callback('open_project', self._open_project_with_ai)
        self.set_callback('ai_assistant', self._ai_assistant_with_integration)
        self.set_callback('configure_ai', self._configure_ai_providers_integrated)
        self.set_callback('test_ai', self._test_ai_connections_integrated)
        self.set_callback('send_ai_message', self._send_ai_message_integrated)
        self.set_callback('code_analysis', self._code_analysis_with_ai)
        self.set_callback('security_scanner', self._security_scanner_with_ai)

    def _create_project_with_ai(self):
        """Create project with AI assistance"""
        if hasattr(self, 'advanced_ai'):
            def ai_task():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(self.advanced_ai.process_user_input_windows(
                        "Help me create a new Windows project. What information do you need?"
                    ))
                    loop.close()
                    self.root.after(0, lambda: self.show_message("AI Project Assistant", response))
                except Exception as e:
                    self.root.after(0, lambda: self.show_message("AI Error", f"Failed to get AI assistance: {e}", "error"))

            threading.Thread(target=ai_task, daemon=True).start()
        else:
            self.show_message("Create Project", "AI assistance not available. Basic project creation would be implemented here.")

    def _open_project_with_ai(self):
        """Open project with AI context"""
        if hasattr(self, 'advanced_ai'):
            folder = filedialog.askdirectory(title="Select Project Directory")
            if folder:
                def ai_analysis_task():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        response = loop.run_until_complete(self.advanced_ai.process_user_input_windows(
                            f"I'm opening a Windows project at {folder}. Can you analyze its structure and provide insights?",
                            files=[folder]
                        ))
                        loop.close()
                        self.root.after(0, lambda: self.show_message("AI Project Analysis", response))
                    except Exception as e:
                        self.root.after(0, lambda: self.show_message("AI Analysis Error", f"Failed to analyze project: {e}", "error"))

                self.update_status("Analyzing project...")
                threading.Thread(target=ai_analysis_task, daemon=True).start()
        else:
            folder = filedialog.askdirectory(title="Select Project Directory")
            if folder:
                self.show_message("Open Project", f"Opened project: {folder}")

    def _ai_assistant_with_integration(self):
        """Launch AI assistant with full Windows integration"""
        if hasattr(self, 'advanced_ai'):
            # Switch to AI assistant tab
            for i in range(self.notebook.index("end")):
                if "AI Assistant" in self.notebook.tab(i, "text"):
                    self.notebook.select(i)
                    break
            self.show_message("AI Assistant", "Windows AI Assistant ready! Type your questions below.")
        else:
            self.show_message("AI Assistant", "AI integration not available.")

    def _configure_ai_providers_integrated(self):
        """Configure AI providers with Windows credential management"""
        if hasattr(self, 'advanced_ai'):
            self.show_message("AI Configuration", "AI provider configuration dialog would be shown here with Windows Credential Manager integration.")
        else:
            self.show_message("AI Configuration", "AI system not available.")

    def _test_ai_connections_integrated(self):
        """Test AI connections with real providers"""
        if hasattr(self, 'advanced_ai'):
            def test_connections():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    test_results = []

                    for provider in ['openai', 'anthropic', 'google', 'cohere']:
                        try:
                            if hasattr(self.advanced_ai, 'ai_clients') and provider in self.advanced_ai.ai_clients:
                                # Test with a simple query
                                result = loop.run_until_complete(asyncio.wait_for(
                                    self.advanced_ai.process_user_input_windows(
                                        f"Test connection for {provider}",
                                        stream=False
                                    ),
                                    timeout=10.0  # 10 second timeout per provider
                                ))
                                test_results.append(f"{provider}: ✅ Connected")
                            else:
                                test_results.append(f"{provider}: ❌ Not configured")
                        except asyncio.TimeoutError:
                            test_results.append(f"{provider}: ⏱️ Connection timeout")
                        except Exception as e:
                            test_results.append(f"{provider}: ❌ Error: {str(e)[:50]}")

                    loop.close()
                    self.root.after(0, lambda: self.show_message("AI Connection Test", "\n".join(test_results)))
                except Exception as e:
                    self.root.after(0, lambda: self.show_message("AI Connection Test", f"Test failed: {e}", "error"))

            self.update_status("Testing AI connections...")
            threading.Thread(target=test_connections, daemon=True).start()
        else:
            self.show_message("AI Connection Test", "AI system not available.", "error")

    def _send_ai_message_integrated(self, message: str, callback):
        """Send message to AI with streaming response"""
        if hasattr(self, 'advanced_ai'):
            def send_message():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(self.advanced_ai.process_user_input_windows(
                        message,
                        stream=True,
                        use_windows_integration=True
                    ))
                    loop.close()
                    # Use root.after for thread-safe GUI updates
                    self.root.after(0, lambda: callback(response))
                except Exception as e:
                    self.root.after(0, lambda: callback(f"AI Error: {e}"))

            threading.Thread(target=send_message, daemon=True).start()
        else:
            callback("AI integration not configured. Please configure your AI providers in settings.")

    def _code_analysis_with_ai(self):
        """Perform AI-powered code analysis"""
        if hasattr(self, 'advanced_ai'):
            folder = filedialog.askdirectory(title="Select Code Directory for Analysis")
            if folder:
                # Use the advanced code analyzer
                self.analyze_project_with_ai(folder)
        else:
            self.show_message("Code Analysis", "AI-powered code analysis requires AI integration.")

    def _security_scanner_with_ai(self):
        """Perform AI-powered security scanning"""
        if hasattr(self, 'advanced_ai'):
            folder = filedialog.askdirectory(title="Select Directory for Security Scan")
            if folder:
                def scan_security():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        response = loop.run_until_complete(asyncio.wait_for(
                            self.advanced_ai.process_user_input_windows(
                                f"Perform a comprehensive Windows security scan of {folder}. Look for potential vulnerabilities, Windows-specific security issues, and provide recommendations.",
                                files=[folder]
                            ),
                            timeout=120.0  # 2 minute timeout for security scan
                        ))
                        loop.close()
                        self.root.after(0, lambda: self.show_message("AI Security Scan", response))
                        self.root.after(0, lambda: self.update_status("Ready"))
                    except asyncio.TimeoutError:
                        self.root.after(0, lambda: self.show_message("AI Security Scan", "Security scan timed out. Try scanning a smaller directory.", "error"))
                        self.root.after(0, lambda: self.update_status("Ready"))
                    except Exception as e:
                        self.root.after(0, lambda: self.show_message("AI Security Scan", f"Scan failed: {e}", "error"))
                        self.root.after(0, lambda: self.update_status("Ready"))

                self.update_status("Performing AI security scan...")
                threading.Thread(target=scan_security, daemon=True).start()
        else:
            self.show_message("Security Scanner", "AI-powered security scanning requires AI integration.")