#!/usr/bin/env python3
"""
Linux GUI Components
Advanced graphical user interface for Terminal Coder on Linux
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext, simpledialog
import threading
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import json
import subprocess
import os
import sys

# Linux-specific imports
try:
    import dbus
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

from rich.console import Console


@dataclass
class LinuxGUITheme:
    """Linux GUI theme configuration"""
    bg_color: str = "#2d3142"
    fg_color: str = "#ffffff"
    accent_color: str = "#4f5d75"
    secondary_color: str = "#bfc0c0"
    success_color: str = "#2a9d8f"
    warning_color: str = "#e9c46a"
    error_color: str = "#e76f51"


class LinuxGUI:
    """Advanced Linux GUI for Terminal Coder"""

    def __init__(self):
        self.root = None
        self.console = Console()
        self.theme = LinuxGUITheme()
        self.is_running = False

        # Desktop environment detection
        self.desktop_env = self._detect_desktop_environment()

        # D-Bus integration
        self.dbus_session = None
        if DBUS_AVAILABLE:
            self._init_dbus()

        # GUI components
        self.main_window = None
        self.notebook = None
        self.status_bar = None

        # Callbacks
        self.callbacks = {}

    def _detect_desktop_environment(self) -> str:
        """Detect the desktop environment"""
        desktop_env = (
            os.environ.get('XDG_CURRENT_DESKTOP') or
            os.environ.get('DESKTOP_SESSION') or
            os.environ.get('GDMSESSION')
        )

        if desktop_env:
            return desktop_env.lower()

        # Fallback detection
        if os.environ.get('KDE_FULL_SESSION'):
            return 'kde'
        elif os.environ.get('GNOME_DESKTOP_SESSION_ID'):
            return 'gnome'
        elif os.environ.get('XFCE4_SESSION'):
            return 'xfce'

        return 'unknown'

    def _init_dbus(self):
        """Initialize D-Bus for desktop integration"""
        try:
            self.dbus_session = dbus.SessionBus()
        except Exception as e:
            self.console.print(f"[yellow]D-Bus not available: {e}[/yellow]")

    def initialize(self) -> bool:
        """Initialize the GUI"""
        try:
            self.root = tk.Tk()
            self._setup_root_window()
            self._apply_linux_styling()
            self._create_main_interface()
            return True
        except Exception as e:
            self.console.print(f"[red]Error initializing GUI: {e}[/red]")
            return False

    def _setup_root_window(self):
        """Setup the main root window"""
        self.root.title("Terminal Coder - Linux Edition")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Set window icon (Linux desktop icon)
        try:
            # Try to set a standard icon
            self.root.wm_iconname("terminal-coder")
        except Exception:
            pass

        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")

        # Configure close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Linux-specific window properties
        if self.desktop_env in ['gnome', 'unity']:
            # GNOME-specific settings
            try:
                self.root.tk.call('wm', 'attributes', '.', '-type', 'dialog')
            except tk.TclError:
                pass

    def _apply_linux_styling(self):
        """Apply Linux-specific styling"""
        # Configure ttk styles for Linux
        style = ttk.Style()

        # Use appropriate theme based on desktop environment
        available_themes = style.theme_names()

        if self.desktop_env == 'kde' and 'plastik' in available_themes:
            style.theme_use('plastik')
        elif self.desktop_env in ['gnome', 'unity'] and 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        else:
            style.theme_use('default')

        # Configure custom styles
        style.configure('Title.TLabel', font=('Liberation Sans', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Liberation Sans', 12, 'bold'))
        style.configure('Code.TText', font=('Liberation Mono', 10))

        # Configure colors for dark theme (respecting system theme)
        try:
            style.configure('TFrame', background=self.theme.bg_color)
            style.configure('TLabel', background=self.theme.bg_color, foreground=self.theme.fg_color)
            style.configure('TButton', font=('Liberation Sans', 9))
        except tk.TclError:
            # Fall back to default styling if theme doesn't support it
            pass

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
        file_menu.add_command(label="Preferences", accelerator="Ctrl+,", command=self._show_preferences)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", accelerator="Ctrl+Q", command=self._on_closing)

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
        tools_menu.add_command(label="Package Manager", command=self._package_manager)
        tools_menu.add_command(label="Container Manager", command=self._container_manager)

        # Linux menu
        linux_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Linux", menu=linux_menu)
        linux_menu.add_command(label="System Information", command=self._system_information)
        linux_menu.add_command(label="Process Manager", command=self._process_manager)
        linux_menu.add_command(label="Service Manager", command=self._service_manager)
        linux_menu.add_command(label="Terminal", command=self._open_terminal)
        linux_menu.add_command(label="File Manager", command=self._open_file_manager)

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

        # Toolbar buttons with Linux-appropriate icons
        ttk.Button(toolbar, text="📄 New", command=self._new_project).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📂 Open", command=self._open_project).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        ttk.Button(toolbar, text="🤖 AI", command=self._open_ai_assistant).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔍 Analyze", command=self._code_analysis).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)

        ttk.Button(toolbar, text="🐧 Terminal", command=self._open_terminal).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📁 Files", command=self._open_file_manager).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📦 Packages", command=self._package_manager).pack(side=tk.LEFT, padx=2)

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

        # Linux Tools tab
        self._create_linux_tools_tab()

        # Settings tab
        self._create_settings_tab()

    def _create_dashboard_tab(self):
        """Create the dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="🏠 Dashboard")

        # Welcome panel
        welcome_frame = ttk.LabelFrame(dashboard_frame, text="Welcome to Terminal Coder Linux", padding=10)
        welcome_frame.pack(fill=tk.X, padx=10, pady=10)

        welcome_text = f"""
Terminal Coder - Linux Edition
Advanced AI-Powered Development Environment

🐧 Native Linux integration with {self.desktop_env.title()} desktop
🤖 AI-powered code assistance and analysis
📦 Package manager integration
🐳 Container development support
🔧 systemd service management
🛡️ Security scanning and compliance
        """

        ttk.Label(welcome_frame, text=welcome_text, font=('Liberation Sans', 10)).pack()

        # System info panel
        system_frame = ttk.LabelFrame(dashboard_frame, text="System Information", padding=10)
        system_frame.pack(fill=tk.X, padx=10, pady=10)

        system_info = self._get_system_info()
        system_text = f"""Distribution: {system_info['distribution']}
Kernel: {system_info['kernel']}
Desktop: {self.desktop_env.title()}
Python: {sys.version.split()[0]}
        """

        ttk.Label(system_frame, text=system_info, font=('Liberation Mono', 9)).pack(anchor=tk.W)

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

        ttk.Button(actions_inner, text="📦 Packages",
                  command=self._package_manager, width=20).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(actions_inner, text="🐳 Containers",
                  command=self._container_manager, width=20).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(actions_inner, text="🐧 Terminal",
                  command=self._open_terminal, width=20).grid(row=1, column=2, padx=5, pady=5)

    def _create_projects_tab(self):
        """Create the projects management tab"""
        projects_frame = ttk.Frame(self.notebook)
        self.notebook.add(projects_frame, text="📂 Projects")

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

        ttk.Button(projects_toolbar, text="Open in Terminal",
                  command=self._open_project_terminal).pack(side=tk.LEFT, padx=2)
        ttk.Button(projects_toolbar, text="Open in File Manager",
                  command=self._open_project_files).pack(side=tk.LEFT, padx=2)

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
        self.notebook.add(ai_frame, text="🤖 AI Assistant")

        # AI provider selection
        provider_frame = ttk.LabelFrame(ai_frame, text="AI Configuration", padding=5)
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
        ttk.Button(provider_frame, text="Test",
                  command=self._test_ai_connections).grid(row=0, column=5, padx=5, pady=2)

        provider_frame.columnconfigure(1, weight=1)
        provider_frame.columnconfigure(3, weight=1)

        # Conversation area
        conversation_frame = ttk.LabelFrame(ai_frame, text="AI Conversation", padding=5)
        conversation_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(conversation_frame,
                                                     height=20,
                                                     state=tk.DISABLED,
                                                     wrap=tk.WORD,
                                                     font=('Liberation Mono', 10))
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Input area
        input_frame = ttk.Frame(conversation_frame)
        input_frame.pack(fill=tk.X)

        self.user_input = tk.Text(input_frame, height=3, wrap=tk.WORD, font=('Liberation Sans', 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Send button
        send_button = ttk.Button(input_frame, text="Send", command=self._send_ai_message)
        send_button.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind Enter key
        self.user_input.bind('<Control-Return>', lambda e: self._send_ai_message())

    def _create_system_monitor_tab(self):
        """Create the system monitoring tab"""
        monitor_frame = ttk.Frame(self.notebook)
        self.notebook.add(monitor_frame, text="📊 System")

        # System info panel
        info_frame = ttk.LabelFrame(monitor_frame, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.system_info_text = tk.Text(info_frame, height=8, state=tk.DISABLED,
                                       font=('Liberation Mono', 9))
        self.system_info_text.pack(fill=tk.X)

        # Performance monitoring
        perf_frame = ttk.LabelFrame(monitor_frame, text="Performance", padding=10)
        perf_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # CPU and Memory usage
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

        ttk.Label(perf_frame, text="Load Average:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.load_label = ttk.Label(perf_frame, text="0.00, 0.00, 0.00")
        self.load_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        perf_frame.columnconfigure(1, weight=1)

        # Processes list
        processes_frame = ttk.LabelFrame(perf_frame, text="Top Processes", padding=5)
        processes_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.processes_tree = ttk.Treeview(processes_frame,
                                          columns=('PID', 'User', 'CPU', 'Memory'),
                                          show='tree headings', height=10)

        self.processes_tree.heading('#0', text='Name')
        self.processes_tree.heading('PID', text='PID')
        self.processes_tree.heading('User', text='User')
        self.processes_tree.heading('CPU', text='CPU %')
        self.processes_tree.heading('Memory', text='Memory %')

        self.processes_tree.pack(fill=tk.BOTH, expand=True)

        perf_frame.rowconfigure(3, weight=1)

    def _create_linux_tools_tab(self):
        """Create Linux-specific tools tab"""
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(tools_frame, text="🐧 Linux Tools")

        # Package management
        pkg_frame = ttk.LabelFrame(tools_frame, text="Package Management", padding=10)
        pkg_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(pkg_frame, text="Search Packages",
                  command=self._search_packages).pack(side=tk.LEFT, padx=5)
        ttk.Button(pkg_frame, text="Update System",
                  command=self._update_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(pkg_frame, text="Package Manager",
                  command=self._package_manager).pack(side=tk.LEFT, padx=5)

        # Services management
        services_frame = ttk.LabelFrame(tools_frame, text="Services Management", padding=10)
        services_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(services_frame, text="List Services",
                  command=self._list_services).pack(side=tk.LEFT, padx=5)
        ttk.Button(services_frame, text="Service Manager",
                  command=self._service_manager).pack(side=tk.LEFT, padx=5)

        # Container tools
        containers_frame = ttk.LabelFrame(tools_frame, text="Container Tools", padding=10)
        containers_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(containers_frame, text="Docker Status",
                  command=self._docker_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(containers_frame, text="Container Manager",
                  command=self._container_manager).pack(side=tk.LEFT, padx=5)

        # Network tools
        network_frame = ttk.LabelFrame(tools_frame, text="Network Tools", padding=10)
        network_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(network_frame, text="Network Info",
                  command=self._network_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(network_frame, text="Port Scanner",
                  command=self._port_scanner).pack(side=tk.LEFT, padx=5)

    def _create_settings_tab(self):
        """Create the settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")

        # Create notebook for settings categories
        settings_notebook = ttk.Notebook(settings_frame)
        settings_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # General settings
        self._create_general_settings_tab(settings_notebook)

        # AI settings
        self._create_ai_settings_tab(settings_notebook)

        # Linux settings
        self._create_linux_settings_tab(settings_notebook)

    def _create_general_settings_tab(self, parent):
        """Create general settings tab"""
        general_frame = ttk.Frame(parent)
        parent.add(general_frame, text="General")

        # Theme settings
        theme_frame = ttk.LabelFrame(general_frame, text="Appearance", padding=10)
        theme_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(theme_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.theme_var = tk.StringVar(value="auto")
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

    def _create_linux_settings_tab(self, parent):
        """Create Linux-specific settings tab"""
        linux_frame = ttk.Frame(parent)
        parent.add(linux_frame, text="Linux")

        # Linux integration settings
        integration_frame = ttk.LabelFrame(linux_frame, text="Linux Integration", padding=10)
        integration_frame.pack(fill=tk.X, padx=5, pady=5)

        self.dbus_var = tk.BooleanVar(value=DBUS_AVAILABLE)
        ttk.Checkbutton(integration_frame, text="Enable D-Bus integration",
                       variable=self.dbus_var, state=tk.NORMAL if DBUS_AVAILABLE else tk.DISABLED).pack(anchor=tk.W, pady=2)

        self.notifications_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(integration_frame, text="Desktop notifications",
                       variable=self.notifications_var).pack(anchor=tk.W, pady=2)

        self.systemd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(integration_frame, text="systemd integration",
                       variable=self.systemd_var).pack(anchor=tk.W, pady=2)

        # Desktop environment info
        de_frame = ttk.LabelFrame(linux_frame, text="Desktop Environment", padding=10)
        de_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(de_frame, text=f"Detected: {self.desktop_env.title()}").pack(anchor=tk.W)

    def _create_projects_context_menu(self):
        """Create context menu for projects tree"""
        self.projects_context_menu = tk.Menu(self.root, tearoff=0)
        self.projects_context_menu.add_command(label="Open", command=self._open_selected_project)
        self.projects_context_menu.add_command(label="Open in Terminal", command=self._open_project_terminal)
        self.projects_context_menu.add_command(label="Open in File Manager", command=self._open_project_files)
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

        # Linux system info
        system_info = self._get_system_info()
        self.system_status_label = ttk.Label(self.status_bar, text=f"Linux | {self.desktop_env.title()}")
        self.system_status_label.pack(side=tk.RIGHT, padx=5)

        # AI status
        self.ai_status_label = ttk.Label(self.status_bar, text="AI: Not configured")
        self.ai_status_label.pack(side=tk.RIGHT, padx=5)

    def _get_system_info(self) -> str:
        """Get Linux system information"""
        try:
            import distro
            import os

            distribution = distro.name()
            kernel = os.uname().release

            return f"{distribution} | Kernel {kernel}"
        except Exception:
            return "Linux"

    # Event handlers
    def _on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit Terminal Coder?"):
            self.is_running = False
            if self.root:
                self.root.quit()
                self.root.destroy()

    def _send_notification(self, title: str, message: str):
        """Send desktop notification via D-Bus"""
        if not DBUS_AVAILABLE or not self.dbus_session:
            return

        try:
            notify_service = self.dbus_session.get_object(
                'org.freedesktop.Notifications',
                '/org/freedesktop/Notifications'
            )

            notify_interface = dbus.Interface(notify_service, 'org.freedesktop.Notifications')

            notify_interface.Notify(
                "Terminal Coder",
                0,
                "terminal",
                title,
                message,
                [],
                {"urgency": 1},
                5000
            )
        except Exception as e:
            print(f"Notification failed: {e}")

    # Placeholder methods for button callbacks
    def _new_project(self):
        if 'create_project' in self.callbacks:
            threading.Thread(target=self.callbacks['create_project'], daemon=True).start()

    def _open_project(self):
        if 'open_project' in self.callbacks:
            threading.Thread(target=self.callbacks['open_project'], daemon=True).start()

    def _open_ai_assistant(self):
        if 'ai_assistant' in self.callbacks:
            threading.Thread(target=self.callbacks['ai_assistant'], daemon=True).start()

    def _configure_ai_providers(self):
        if 'configure_ai' in self.callbacks:
            threading.Thread(target=self.callbacks['configure_ai'], daemon=True).start()

    def _test_ai_connections(self):
        if 'test_ai' in self.callbacks:
            threading.Thread(target=self.callbacks['test_ai'], daemon=True).start()

    def _open_terminal(self):
        """Open Linux terminal"""
        terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm', 'terminator']

        for terminal in terminals:
            try:
                subprocess.Popen([terminal])
                break
            except FileNotFoundError:
                continue
        else:
            messagebox.showerror("Error", "No terminal emulator found")

    def _open_file_manager(self):
        """Open Linux file manager"""
        file_managers = ['nautilus', 'dolphin', 'thunar', 'pcmanfm', 'nemo']

        for fm in file_managers:
            try:
                subprocess.Popen([fm])
                break
            except FileNotFoundError:
                continue
        else:
            messagebox.showerror("Error", "No file manager found")

    # Real implementations for Linux GUI functionality
    def _show_preferences(self):
        """Show preferences (switch to settings tab)"""
        self.notebook.select(5)  # Settings tab index

    def _code_analysis(self):
        """Perform code analysis"""
        if 'code_analysis' in self.callbacks:
            threading.Thread(target=self.callbacks['code_analysis'], daemon=True).start()
        else:
            self.show_message("Code Analysis", "Code analysis feature will be available with AI integration.")

    def _security_scanner(self):
        """Run security scanner"""
        if 'security_scanner' in self.callbacks:
            threading.Thread(target=self.callbacks['security_scanner'], daemon=True).start()
        else:
            self.show_message("Security Scanner", "Security scanner feature will be available with AI integration.")

    def _package_manager(self):
        """Open package manager dialog"""
        def run_package_manager():
            dialog = PackageManagerDialog(self.root, self.desktop_env)
            dialog.show()

        threading.Thread(target=run_package_manager, daemon=True).start()

    def _container_manager(self):
        """Open container manager"""
        def run_container_manager():
            dialog = ContainerManagerDialog(self.root)
            dialog.show()

        threading.Thread(target=run_container_manager, daemon=True).start()

    def _system_information(self):
        """Show system information"""
        def show_system_info():
            try:
                # Gather comprehensive system information
                import platform
                import psutil
                import distro

                info = []
                info.append("=== Linux System Information ===\n")
                info.append(f"Distribution: {distro.name()} {distro.version()}")
                info.append(f"Kernel: {platform.release()}")
                info.append(f"Architecture: {platform.machine()}")
                info.append(f"Desktop Environment: {self.desktop_env.title()}")
                info.append(f"Hostname: {platform.node()}")
                info.append("")

                # CPU Information
                info.append("=== CPU Information ===")
                info.append(f"Processor: {platform.processor()}")
                info.append(f"CPU Count: {psutil.cpu_count()} cores")
                info.append(f"CPU Frequency: {psutil.cpu_freq().current:.0f} MHz")
                info.append("")

                # Memory Information
                memory = psutil.virtual_memory()
                info.append("=== Memory Information ===")
                info.append(f"Total RAM: {memory.total // (1024**3)} GB")
                info.append(f"Available RAM: {memory.available // (1024**3)} GB")
                info.append(f"Used RAM: {memory.used // (1024**3)} GB ({memory.percent:.1f}%)")
                info.append("")

                # Disk Information
                info.append("=== Disk Information ===")
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        info.append(f"Device: {partition.device}")
                        info.append(f"  Mountpoint: {partition.mountpoint}")
                        info.append(f"  File system: {partition.fstype}")
                        info.append(f"  Total: {usage.total // (1024**3)} GB")
                        info.append(f"  Used: {usage.used // (1024**3)} GB ({usage.percent:.1f}%)")
                        info.append(f"  Free: {usage.free // (1024**3)} GB")
                        info.append("")
                    except PermissionError:
                        continue

                system_info_text = "\n".join(info)

                # Update the system info display
                self.update_system_info(system_info_text)

                # Switch to system monitor tab
                self.notebook.select(3)

                # Show notification
                self._send_notification("System Information", "System information updated successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Could not gather system information: {e}")

        threading.Thread(target=show_system_info, daemon=True).start()

    def _process_manager(self):
        """Open process manager"""
        def show_processes():
            try:
                import psutil

                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
                    try:
                        processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Sort by CPU usage
                processes.sort(key=lambda x: x.get('cpu_percent', 0) or 0, reverse=True)

                # Update processes tree
                self.update_processes_list(processes[:50])  # Top 50 processes

                # Switch to system monitor tab
                self.notebook.select(3)

            except Exception as e:
                messagebox.showerror("Error", f"Could not get process information: {e}")

        threading.Thread(target=show_processes, daemon=True).start()

    def _service_manager(self):
        """Open service manager"""
        def show_services():
            try:
                # Check if systemd is available
                result = subprocess.run(['systemctl', '--version'], capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    # Get systemd services
                    services_result = subprocess.run(
                        ['systemctl', 'list-units', '--type=service', '--no-pager'],
                        capture_output=True, text=True, timeout=10
                    )

                    if services_result.returncode == 0:
                        dialog = ServicesManagerDialog(self.root, services_result.stdout)
                        dialog.show()
                    else:
                        messagebox.showerror("Error", "Could not retrieve systemd services")
                else:
                    messagebox.showinfo("Service Manager", "systemd not available on this system")

            except Exception as e:
                messagebox.showerror("Error", f"Could not access service manager: {e}")

        threading.Thread(target=show_services, daemon=True).start()

    def _show_documentation(self):
        """Show documentation"""
        doc_text = """
Terminal Coder - Linux Edition Documentation

== Getting Started ==
1. Create a new project or open an existing one
2. Configure your AI providers in the AI settings
3. Use the AI Assistant tab for intelligent code assistance
4. Access Linux tools for system management

== AI Features ==
- Multi-provider support (OpenAI, Anthropic, Google, Cohere)
- Real-time streaming responses
- Code analysis and suggestions
- Project-specific AI contexts

== Linux Integration ==
- systemd service management
- Package manager integration
- Container support (Docker/Podman)
- Desktop environment integration
- D-Bus notifications

== Keyboard Shortcuts ==
Ctrl+N - New Project
Ctrl+O - Open Project
Ctrl+A - AI Assistant
Ctrl+, - Preferences
Ctrl+Q - Quit
Ctrl+Return - Send AI message

== System Tools ==
- Process manager
- System information
- Network tools
- Container management
- Package management
        """
        messagebox.showinfo("Documentation", doc_text)

    def _show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
Terminal Coder - Linux Edition Keyboard Shortcuts:

=== General ===
Ctrl+N - New Project
Ctrl+O - Open Project
Ctrl+Q - Quit Application
Ctrl+, - Open Preferences

=== AI Assistant ===
Ctrl+A - Open AI Assistant
Ctrl+Return - Send AI message
Esc - Clear input

=== Linux Tools ===
Ctrl+T - Open Terminal
Ctrl+Shift+P - Package Manager
Ctrl+Shift+S - Service Manager
Ctrl+Shift+I - System Information

=== Project Management ===
F2 - Rename Project
Delete - Delete Selected Project
Ctrl+R - Refresh Project List

=== Navigation ===
Ctrl+Tab - Next Tab
Ctrl+Shift+Tab - Previous Tab
F1 - Help Documentation
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def _show_about(self):
        """Show about dialog"""
        about_text = f"""
Terminal Coder - Linux Edition
Version 2.0.0

Advanced AI-Powered Development Environment
Optimized for Linux

Desktop Environment: {self.desktop_env.title()}
D-Bus Integration: {'✓' if DBUS_AVAILABLE else '✗'}

Features:
• Multi-AI provider support
• Native Linux integration
• systemd service management
• Package manager integration
• Container development tools
• Desktop notifications
• XDG Base Directory compliance

© 2024 Terminal Coder Team
Open Source Project
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
        else:
            # Fallback response
            self.root.after(2000, lambda: self._display_ai_response("AI integration not configured. Please configure your AI providers in settings."))

    def _on_ai_response(self, response: str):
        """Handle AI response"""
        self.root.after(0, lambda: self._display_ai_response(response))

    def _display_ai_response(self, response: str):
        """Display AI response in chat"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"AI: {response}\n\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

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

    def _open_project_terminal(self):
        """Open project in terminal"""
        selection = self.projects_tree.selection()
        if selection:
            item = self.projects_tree.item(selection[0])
            project_path = item['values'][3]  # Path column

            terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']

            for terminal in terminals:
                try:
                    if terminal in ['gnome-terminal', 'xfce4-terminal']:
                        subprocess.Popen([terminal, '--working-directory', project_path])
                    elif terminal == 'konsole':
                        subprocess.Popen([terminal, '--workdir', project_path])
                    else:
                        subprocess.Popen([terminal], cwd=project_path)
                    break
                except FileNotFoundError:
                    continue
            else:
                messagebox.showerror("Error", "No terminal emulator found")

    def _open_project_files(self):
        """Open project in file manager"""
        selection = self.projects_tree.selection()
        if selection:
            item = self.projects_tree.item(selection[0])
            project_path = item['values'][3]  # Path column

            file_managers = ['nautilus', 'dolphin', 'thunar', 'pcmanfm', 'nemo']

            for fm in file_managers:
                try:
                    subprocess.Popen([fm, project_path])
                    break
                except FileNotFoundError:
                    continue
            else:
                messagebox.showerror("Error", "No file manager found")

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

    def _delete_selected_project(self):
        """Delete selected project (same as _delete_project)"""
        self._delete_project()

    def _search_packages(self):
        """Search for packages"""
        package_name = tk.simpledialog.askstring("Search Packages", "Enter package name to search:")
        if package_name:
            def search():
                try:
                    # Detect package manager
                    package_managers = {
                        'apt': ['apt', 'search', package_name],
                        'dnf': ['dnf', 'search', package_name],
                        'pacman': ['pacman', '-Ss', package_name],
                        'zypper': ['zypper', 'search', package_name],
                        'apk': ['apk', 'search', package_name]
                    }

                    for pm, cmd in package_managers.items():
                        if subprocess.run(['which', pm], capture_output=True).returncode == 0:
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                            if result.returncode == 0:
                                dialog = SearchResultsDialog(self.root, f"Package Search Results ({pm})", result.stdout)
                                dialog.show()
                            else:
                                messagebox.showerror("Error", f"Search failed: {result.stderr}")
                            break
                    else:
                        messagebox.showerror("Error", "No supported package manager found")

                except Exception as e:
                    messagebox.showerror("Error", f"Search failed: {e}")

            threading.Thread(target=search, daemon=True).start()

    def _update_system(self):
        """Update system packages"""
        if messagebox.askyesno("System Update", "Update all system packages? This may take some time."):
            def update():
                try:
                    # Detect package manager and run update
                    if subprocess.run(['which', 'apt'], capture_output=True).returncode == 0:
                        cmd = ['sudo', 'apt', 'update', '&&', 'sudo', 'apt', 'upgrade', '-y']
                    elif subprocess.run(['which', 'dnf'], capture_output=True).returncode == 0:
                        cmd = ['sudo', 'dnf', 'update', '-y']
                    elif subprocess.run(['which', 'pacman'], capture_output=True).returncode == 0:
                        cmd = ['sudo', 'pacman', '-Syu', '--noconfirm']
                    else:
                        messagebox.showerror("Error", "No supported package manager found")
                        return

                    self.update_status("Updating system packages...")

                    # Open terminal for the update process
                    terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']

                    for terminal in terminals:
                        try:
                            if terminal == 'gnome-terminal':
                                subprocess.Popen([terminal, '--', 'bash', '-c', f"{' '.join(cmd)}; read -p 'Press Enter to continue...'"])
                            else:
                                subprocess.Popen([terminal, '-e', 'bash', '-c', f"{' '.join(cmd)}; read -p 'Press Enter to continue...'"])
                            break
                        except FileNotFoundError:
                            continue
                    else:
                        messagebox.showerror("Error", "No terminal emulator found")

                    self.update_status("Ready")

                except Exception as e:
                    messagebox.showerror("Error", f"Update failed: {e}")
                    self.update_status("Ready")

            threading.Thread(target=update, daemon=True).start()

    def _list_services(self):
        """List systemd services"""
        def show_services():
            try:
                result = subprocess.run(
                    ['systemctl', 'list-units', '--type=service', '--no-pager'],
                    capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0:
                    dialog = SearchResultsDialog(self.root, "systemd Services", result.stdout)
                    dialog.show()
                else:
                    messagebox.showerror("Error", "Could not list systemd services")

            except Exception as e:
                messagebox.showerror("Error", f"Could not list services: {e}")

        threading.Thread(target=show_services, daemon=True).start()

    def _docker_status(self):
        """Show Docker status"""
        def check_docker():
            try:
                # Check if Docker is running
                result = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    dialog = SearchResultsDialog(self.root, "Docker Status", result.stdout)
                    dialog.show()
                else:
                    messagebox.showwarning("Docker", "Docker is not running or not installed")

            except FileNotFoundError:
                messagebox.showerror("Error", "Docker not found. Is Docker installed?")
            except Exception as e:
                messagebox.showerror("Error", f"Could not check Docker status: {e}")

        threading.Thread(target=check_docker, daemon=True).start()

    def _network_info(self):
        """Show network information"""
        def show_network():
            try:
                import psutil

                info = []
                info.append("=== Network Interface Information ===\n")

                # Get network interfaces
                interfaces = psutil.net_if_addrs()
                stats = psutil.net_if_stats()

                for interface_name, addresses in interfaces.items():
                    info.append(f"Interface: {interface_name}")

                    if interface_name in stats:
                        stat = stats[interface_name]
                        info.append(f"  Status: {'Up' if stat.isup else 'Down'}")
                        info.append(f"  Speed: {stat.speed} Mbps" if stat.speed > 0 else "  Speed: Unknown")

                    for addr in addresses:
                        if addr.family.name == 'AF_INET':  # IPv4
                            info.append(f"  IPv4: {addr.address}")
                            info.append(f"  Netmask: {addr.netmask}")
                        elif addr.family.name == 'AF_INET6':  # IPv6
                            info.append(f"  IPv6: {addr.address}")

                    info.append("")

                # Network IO statistics
                net_io = psutil.net_io_counters()
                info.append("=== Network I/O Statistics ===")
                info.append(f"Bytes Sent: {net_io.bytes_sent:,}")
                info.append(f"Bytes Received: {net_io.bytes_recv:,}")
                info.append(f"Packets Sent: {net_io.packets_sent:,}")
                info.append(f"Packets Received: {net_io.packets_recv:,}")

                network_info_text = "\n".join(info)
                dialog = SearchResultsDialog(self.root, "Network Information", network_info_text)
                dialog.show()

            except Exception as e:
                messagebox.showerror("Error", f"Could not get network information: {e}")

        threading.Thread(target=show_network, daemon=True).start()

    def _port_scanner(self):
        """Simple port scanner"""
        host = tk.simpledialog.askstring("Port Scanner", "Enter host to scan:", initialvalue="localhost")
        if host:
            port_range = tk.simpledialog.askstring("Port Scanner", "Enter port range (e.g., 1-1000):", initialvalue="1-100")
            if port_range:
                def scan_ports():
                    try:
                        import socket

                        start_port, end_port = map(int, port_range.split('-'))
                        open_ports = []

                        self.update_status(f"Scanning {host}...")

                        for port in range(start_port, min(end_port + 1, 65536)):
                            try:
                                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                sock.settimeout(0.1)
                                result = sock.connect_ex((host, port))
                                if result == 0:
                                    open_ports.append(port)
                                sock.close()
                            except socket.error:
                                pass

                        if open_ports:
                            result_text = f"Open ports on {host}:\n" + "\n".join(map(str, open_ports))
                        else:
                            result_text = f"No open ports found on {host} in range {port_range}"

                        dialog = SearchResultsDialog(self.root, "Port Scan Results", result_text)
                        dialog.show()

                        self.update_status("Ready")

                    except Exception as e:
                        messagebox.showerror("Error", f"Port scan failed: {e}")
                        self.update_status("Ready")

                threading.Thread(target=scan_ports, daemon=True).start()

    # Public methods
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

    def show_message(self, title: str, message: str, msg_type: str = "info"):
        """Show message dialog with optional desktop notification"""
        def _show():
            if msg_type == "error":
                messagebox.showerror(title, message)
            elif msg_type == "warning":
                messagebox.showwarning(title, message)
            else:
                messagebox.showinfo(title, message)

            # Also send desktop notification
            self._send_notification(title, message)

        self.root.after(0, _show)

    def update_system_info(self, info_text: str):
        """Update system info display"""
        if hasattr(self, 'system_info_text') and self.system_info_text:
            self.system_info_text.config(state=tk.NORMAL)
            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(1.0, info_text)
            self.system_info_text.config(state=tk.DISABLED)

    def update_processes_list(self, processes: list):
        """Update processes tree with process data"""
        if hasattr(self, 'processes_tree') and self.processes_tree:
            # Clear existing items
            for item in self.processes_tree.get_children():
                self.processes_tree.delete(item)

            # Add new process data
            for proc in processes:
                try:
                    self.processes_tree.insert('', tk.END, text=proc.get('name', 'Unknown'),
                                             values=(
                                                 str(proc.get('pid', 0)),
                                                 proc.get('username', 'Unknown'),
                                                 f"{proc.get('cpu_percent', 0):.1f}",
                                                 f"{proc.get('memory_percent', 0):.1f}"
                                             ))
                except Exception:
                    continue

    def setup_ai_callbacks(self, advanced_ai):
        """Setup AI system callbacks for GUI integration"""
        self.advanced_ai = advanced_ai

        # Set up callback functions for the GUI to use
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
            # AI can help suggest project structure, dependencies, etc.
            response = asyncio.run(self.advanced_ai.process_user_input(
                "Help me create a new project. What information do you need?"
            ))
            self.show_message("AI Project Assistant", response)
        else:
            self.show_message("Create Project", "AI assistance not available. Basic project creation would be implemented here.")

    def _open_project_with_ai(self):
        """Open project with AI context"""
        if hasattr(self, 'advanced_ai'):
            # AI can analyze project structure and provide insights
            folder = filedialog.askdirectory(title="Select Project Directory")
            if folder:
                response = asyncio.run(self.advanced_ai.process_user_input(
                    f"I'm opening a project at {folder}. Can you analyze its structure and provide insights?",
                    files=[folder]
                ))
                self.show_message("AI Project Analysis", response)
        else:
            folder = filedialog.askdirectory(title="Select Project Directory")
            if folder:
                self.show_message("Open Project", f"Opened project: {folder}")

    def _ai_assistant_with_integration(self):
        """Launch AI assistant with full integration"""
        if hasattr(self, 'advanced_ai'):
            # Switch to AI assistant tab and enable streaming chat
            self.notebook.select(2)  # AI Assistant tab
            self.show_message("AI Assistant", "AI Assistant ready! Type your questions below.")
        else:
            self.show_message("AI Assistant", "AI integration not available.")

    def _configure_ai_providers_integrated(self):
        """Configure AI providers with credential management"""
        if hasattr(self, 'advanced_ai'):
            # Show advanced configuration dialog
            self.show_message("AI Configuration", "AI provider configuration dialog would be shown here with keyring integration.")
        else:
            self.show_message("AI Configuration", "AI system not available.")

    def _test_ai_connections_integrated(self):
        """Test AI connections with real providers"""
        if hasattr(self, 'advanced_ai'):
            def test_connections():
                try:
                    # Test each configured provider
                    test_results = []
                    for provider in ['openai', 'anthropic', 'google', 'cohere']:
                        if hasattr(self.advanced_ai, 'ai_clients') and provider in self.advanced_ai.ai_clients:
                            result = asyncio.run(self.advanced_ai.process_user_input(
                                f"Test connection for {provider}",
                                stream=False
                            ))
                            test_results.append(f"{provider}: ✅ Connected")
                        else:
                            test_results.append(f"{provider}: ❌ Not configured")

                    self.show_message("AI Connection Test", "\n".join(test_results))
                except Exception as e:
                    self.show_message("AI Connection Test", f"Test failed: {e}", "error")

            import threading
            threading.Thread(target=test_connections, daemon=True).start()
        else:
            self.show_message("AI Connection Test", "AI system not available.", "error")

    def _send_ai_message_integrated(self, message: str, callback):
        """Send message to AI with streaming response"""
        if hasattr(self, 'advanced_ai'):
            def send_message():
                try:
                    response = asyncio.run(self.advanced_ai.process_user_input_linux(
                        message,
                        stream=True,
                        use_systemd=True
                    ))
                    callback(response)
                except Exception as e:
                    callback(f"AI Error: {e}")

            import threading
            threading.Thread(target=send_message, daemon=True).start()
        else:
            callback("AI integration not configured. Please configure your AI providers in settings.")

    def _code_analysis_with_ai(self):
        """Perform AI-powered code analysis"""
        if hasattr(self, 'advanced_ai'):
            folder = filedialog.askdirectory(title="Select Code Directory for Analysis")
            if folder:
                def analyze_code():
                    try:
                        response = asyncio.run(self.advanced_ai.process_user_input(
                            f"Analyze the code in {folder}. Provide insights on code quality, potential issues, and improvements.",
                            files=[folder]
                        ))
                        self.show_message("AI Code Analysis", response)
                    except Exception as e:
                        self.show_message("AI Code Analysis", f"Analysis failed: {e}", "error")

                import threading
                threading.Thread(target=analyze_code, daemon=True).start()
        else:
            self.show_message("Code Analysis", "AI-powered code analysis requires AI integration.")

    def _security_scanner_with_ai(self):
        """Perform AI-powered security scanning"""
        if hasattr(self, 'advanced_ai'):
            folder = filedialog.askdirectory(title="Select Directory for Security Scan")
            if folder:
                def scan_security():
                    try:
                        response = asyncio.run(self.advanced_ai.process_user_input(
                            f"Perform a security scan of {folder}. Look for potential vulnerabilities, security issues, and provide recommendations.",
                            files=[folder]
                        ))
                        self.show_message("AI Security Scan", response)
                    except Exception as e:
                        self.show_message("AI Security Scan", f"Scan failed: {e}", "error")

                import threading
                threading.Thread(target=scan_security, daemon=True).start()
        else:
            self.show_message("Security Scanner", "AI-powered security scanning requires AI integration.")


# Missing Dialog Classes Implementation

class PackageManagerDialog:
    """Linux Package Manager Dialog"""

    def __init__(self, parent, desktop_env):
        self.parent = parent
        self.desktop_env = desktop_env
        self.dialog = None
        self.package_manager = self._detect_package_manager()

    def _detect_package_manager(self) -> str:
        """Detect available package manager"""
        managers = [
            ("apt", "apt"),
            ("dnf", "dnf"),
            ("yum", "yum"),
            ("pacman", "pacman"),
            ("zypper", "zypper"),
            ("apk", "apk")
        ]

        for name, command in managers:
            try:
                result = subprocess.run([command, "--version"], capture_output=True, timeout=5)
                if result.returncode == 0:
                    return name
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return "unknown"

    def show(self):
        """Show package manager dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(f"Package Manager - {self.package_manager.upper()}")
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)

        # Center the dialog
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Search frame
        search_frame = ttk.LabelFrame(main_frame, text="Search Packages", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))

        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        ttk.Button(search_frame, text="Search", command=self._search_packages).pack(side=tk.RIGHT)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Results treeview
        self.results_tree = ttk.Treeview(results_frame, columns=('Description',), show='tree headings')
        self.results_tree.heading('#0', text='Package Name')
        self.results_tree.heading('Description', text='Description')

        # Scrollbars for results
        results_scrolly = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        results_scrollx = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)

        self.results_tree.configure(yscrollcommand=results_scrolly.set, xscrollcommand=results_scrollx.set)

        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrolly.grid(row=0, column=1, sticky=(tk.N, tk.S))
        results_scrollx.grid(row=1, column=0, sticky=(tk.W, tk.E))

        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X)

        ttk.Button(buttons_frame, text="Install Selected", command=self._install_selected).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Refresh", command=self._refresh_packages).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Update System", command=self._update_system).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Close", command=self.dialog.destroy).pack(side=tk.RIGHT)

        # Bind Enter key to search
        search_entry.bind('<Return>', lambda e: self._search_packages())

        # Show initial status
        if self.package_manager == "unknown":
            self.results_tree.insert('', tk.END, text="No supported package manager found", values=("",))
        else:
            self.results_tree.insert('', tk.END, text=f"Package manager: {self.package_manager}", values=("Ready to search",))

    def _search_packages(self):
        """Search for packages"""
        query = self.search_var.get().strip()
        if not query:
            return

        def search_thread():
            try:
                # Clear results
                for item in self.results_tree.get_children():
                    self.results_tree.delete(item)

                self.results_tree.insert('', tk.END, text="Searching...", values=("Please wait",))

                # Build search command
                if self.package_manager == "apt":
                    cmd = ["apt", "search", query]
                elif self.package_manager in ["dnf", "yum"]:
                    cmd = [self.package_manager, "search", query]
                elif self.package_manager == "pacman":
                    cmd = ["pacman", "-Ss", query]
                elif self.package_manager == "zypper":
                    cmd = ["zypper", "search", query]
                elif self.package_manager == "apk":
                    cmd = ["apk", "search", query]
                else:
                    return

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                # Clear searching message
                for item in self.results_tree.get_children():
                    self.results_tree.delete(item)

                if result.returncode == 0:
                    self._parse_search_results(result.stdout)
                else:
                    self.results_tree.insert('', tk.END, text="Search failed", values=(result.stderr[:100],))

            except Exception as e:
                self.results_tree.insert('', tk.END, text="Error", values=(str(e),))

        threading.Thread(target=search_thread, daemon=True).start()

    def _parse_search_results(self, output: str):
        """Parse package search results"""
        lines = output.split('\n')
        count = 0

        for line in lines:
            if count >= 50:  # Limit results
                break

            if line.strip():
                if self.package_manager == "apt":
                    if line.startswith('WARNING') or line.startswith('NOTE'):
                        continue
                    parts = line.split(' - ')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        description = parts[1].strip()
                        self.results_tree.insert('', tk.END, text=name, values=(description,))
                        count += 1

                elif self.package_manager in ["dnf", "yum"]:
                    if ':' in line and not line.startswith('='):
                        parts = line.split(' : ')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            description = parts[1].strip()
                            self.results_tree.insert('', tk.END, text=name, values=(description,))
                            count += 1

                elif self.package_manager == "pacman":
                    if '/' in line:
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            name = parts[1]
                            description = ' '.join(parts[2:]) if len(parts) > 2 else ''
                            self.results_tree.insert('', tk.END, text=name, values=(description,))
                            count += 1

        if count == 0:
            self.results_tree.insert('', tk.END, text="No packages found", values=("",))

    def _install_selected(self):
        """Install selected package"""
        selection = self.results_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a package to install.")
            return

        item = self.results_tree.item(selection[0])
        package_name = item['text']

        if messagebox.askyesno("Confirm Install", f"Install package '{package_name}'?"):
            def install_thread():
                try:
                    # Build install command
                    if self.package_manager == "apt":
                        cmd = ["sudo", "apt", "install", "-y", package_name]
                    elif self.package_manager == "dnf":
                        cmd = ["sudo", "dnf", "install", "-y", package_name]
                    elif self.package_manager == "yum":
                        cmd = ["sudo", "yum", "install", "-y", package_name]
                    elif self.package_manager == "pacman":
                        cmd = ["sudo", "pacman", "-S", "--noconfirm", package_name]
                    elif self.package_manager == "zypper":
                        cmd = ["sudo", "zypper", "install", "-y", package_name]
                    elif self.package_manager == "apk":
                        cmd = ["sudo", "apk", "add", package_name]
                    else:
                        return

                    # Open terminal for installation
                    terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
                    for terminal in terminals:
                        try:
                            if terminal == 'gnome-terminal':
                                subprocess.Popen([terminal, '--', 'bash', '-c', f"{' '.join(cmd)}; read -p 'Press Enter to continue...'"])
                            else:
                                subprocess.Popen([terminal, '-e', 'bash', '-c', f"{' '.join(cmd)}; read -p 'Press Enter to continue...'"])
                            break
                        except FileNotFoundError:
                            continue

                except Exception as e:
                    messagebox.showerror("Installation Error", f"Failed to install package: {e}")

            threading.Thread(target=install_thread, daemon=True).start()

    def _refresh_packages(self):
        """Refresh package lists"""
        def refresh_thread():
            try:
                if self.package_manager == "apt":
                    cmd = ["sudo", "apt", "update"]
                elif self.package_manager in ["dnf", "yum"]:
                    cmd = ["sudo", self.package_manager, "check-update"]
                elif self.package_manager == "pacman":
                    cmd = ["sudo", "pacman", "-Sy"]
                elif self.package_manager == "zypper":
                    cmd = ["sudo", "zypper", "refresh"]
                elif self.package_manager == "apk":
                    cmd = ["sudo", "apk", "update"]
                else:
                    return

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    messagebox.showinfo("Refresh Complete", "Package lists refreshed successfully.")
                else:
                    messagebox.showerror("Refresh Error", f"Failed to refresh: {result.stderr}")

            except Exception as e:
                messagebox.showerror("Refresh Error", f"Failed to refresh: {e}")

        if messagebox.askyesno("Refresh Packages", "Refresh package lists? This may take a moment."):
            threading.Thread(target=refresh_thread, daemon=True).start()

    def _update_system(self):
        """Update system packages"""
        if messagebox.askyesno("System Update", "Update all system packages? This may take some time."):
            def update_thread():
                try:
                    # Build update command
                    if self.package_manager == "apt":
                        cmd = "sudo apt update && sudo apt upgrade -y"
                    elif self.package_manager == "dnf":
                        cmd = "sudo dnf update -y"
                    elif self.package_manager == "yum":
                        cmd = "sudo yum update -y"
                    elif self.package_manager == "pacman":
                        cmd = "sudo pacman -Syu --noconfirm"
                    elif self.package_manager == "zypper":
                        cmd = "sudo zypper update -y"
                    elif self.package_manager == "apk":
                        cmd = "sudo apk update && sudo apk upgrade"
                    else:
                        return

                    # Open terminal for update
                    terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
                    for terminal in terminals:
                        try:
                            if terminal == 'gnome-terminal':
                                subprocess.Popen([terminal, '--', 'bash', '-c', f"{cmd}; read -p 'Press Enter to continue...'"])
                            else:
                                subprocess.Popen([terminal, '-e', 'bash', '-c', f"{cmd}; read -p 'Press Enter to continue...'"])
                            break
                        except FileNotFoundError:
                            continue

                except Exception as e:
                    messagebox.showerror("Update Error", f"Failed to start update: {e}")

            threading.Thread(target=update_thread, daemon=True).start()


class ContainerManagerDialog:
    """Linux Container Manager Dialog"""

    def __init__(self, parent):
        self.parent = parent
        self.dialog = None
        self.container_type = "docker"  # or "podman"
        self.has_docker = self._check_docker()
        self.has_podman = self._check_podman()

    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_podman(self) -> bool:
        """Check if Podman is available"""
        try:
            result = subprocess.run(["podman", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def show(self):
        """Show container manager dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Container Manager")
        self.dialog.geometry("900x700")
        self.dialog.resizable(True, True)

        # Center the dialog
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Container type selection
        type_frame = ttk.LabelFrame(main_frame, text="Container Runtime", padding=10)
        type_frame.pack(fill=tk.X, pady=(0, 10))

        self.container_type_var = tk.StringVar()

        if self.has_docker:
            ttk.Radiobutton(type_frame, text="Docker", variable=self.container_type_var,
                          value="docker", command=self._refresh_containers).pack(side=tk.LEFT, padx=(0, 20))
            self.container_type_var.set("docker")

        if self.has_podman:
            ttk.Radiobutton(type_frame, text="Podman", variable=self.container_type_var,
                          value="podman", command=self._refresh_containers).pack(side=tk.LEFT, padx=(0, 20))
            if not self.has_docker:
                self.container_type_var.set("podman")

        if not self.has_docker and not self.has_podman:
            ttk.Label(type_frame, text="No container runtime found (Docker or Podman required)",
                     foreground="red").pack()
            return

        # Containers list
        containers_frame = ttk.LabelFrame(main_frame, text="Containers", padding=10)
        containers_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Containers treeview
        self.containers_tree = ttk.Treeview(containers_frame,
                                          columns=('Status', 'Image', 'Ports'),
                                          show='tree headings')
        self.containers_tree.heading('#0', text='Container Name')
        self.containers_tree.heading('Status', text='Status')
        self.containers_tree.heading('Image', text='Image')
        self.containers_tree.heading('Ports', text='Ports')

        self.containers_tree.column('#0', width=200)
        self.containers_tree.column('Status', width=120)
        self.containers_tree.column('Image', width=200)
        self.containers_tree.column('Ports', width=150)

        # Scrollbars
        containers_scrolly = ttk.Scrollbar(containers_frame, orient=tk.VERTICAL,
                                         command=self.containers_tree.yview)
        containers_scrollx = ttk.Scrollbar(containers_frame, orient=tk.HORIZONTAL,
                                         command=self.containers_tree.xview)

        self.containers_tree.configure(yscrollcommand=containers_scrolly.set,
                                     xscrollcommand=containers_scrollx.set)

        self.containers_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        containers_scrolly.grid(row=0, column=1, sticky=(tk.N, tk.S))
        containers_scrollx.grid(row=1, column=0, sticky=(tk.W, tk.E))

        containers_frame.columnconfigure(0, weight=1)
        containers_frame.rowconfigure(0, weight=1)

        # Control buttons
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(controls_frame, text="Start", command=self._start_container).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Stop", command=self._stop_container).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Restart", command=self._restart_container).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Remove", command=self._remove_container).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Logs", command=self._show_logs).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(controls_frame, text="Refresh", command=self._refresh_containers).pack(side=tk.LEFT, padx=(0, 5))

        # Image management
        images_frame = ttk.LabelFrame(main_frame, text="Images", padding=10)
        images_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(images_frame, text="List Images", command=self._list_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(images_frame, text="Pull Image", command=self._pull_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(images_frame, text="Build Image", command=self._build_image).pack(side=tk.LEFT, padx=(0, 5))

        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X)

        ttk.Button(bottom_frame, text="Open Terminal", command=self._open_terminal).pack(side=tk.LEFT)
        ttk.Button(bottom_frame, text="Close", command=self.dialog.destroy).pack(side=tk.RIGHT)

        # Load initial data
        self._refresh_containers()

    def _refresh_containers(self):
        """Refresh container list"""
        container_type = self.container_type_var.get() or "docker"

        def refresh_thread():
            try:
                # Clear current items
                for item in self.containers_tree.get_children():
                    self.containers_tree.delete(item)

                cmd = [container_type, "ps", "-a", "--format",
                      "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                name = parts[0]
                                status = parts[1]
                                image = parts[2]
                                ports = parts[3] if len(parts) > 3 else ""

                                self.containers_tree.insert('', tk.END, text=name,
                                                          values=(status, image, ports))
                else:
                    self.containers_tree.insert('', tk.END, text="Error loading containers",
                                              values=(result.stderr[:50], "", ""))

            except Exception as e:
                self.containers_tree.insert('', tk.END, text="Error", values=(str(e), "", ""))

        threading.Thread(target=refresh_thread, daemon=True).start()

    def _start_container(self):
        """Start selected container"""
        self._container_action("start")

    def _stop_container(self):
        """Stop selected container"""
        self._container_action("stop")

    def _restart_container(self):
        """Restart selected container"""
        self._container_action("restart")

    def _remove_container(self):
        """Remove selected container"""
        selection = self.containers_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a container.")
            return

        container_name = self.containers_tree.item(selection[0])['text']

        if messagebox.askyesno("Confirm Remove", f"Remove container '{container_name}'?"):
            self._container_action("rm")

    def _container_action(self, action: str):
        """Perform container action"""
        selection = self.containers_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a container.")
            return

        container_name = self.containers_tree.item(selection[0])['text']
        container_type = self.container_type_var.get()

        def action_thread():
            try:
                cmd = [container_type, action, container_name]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    messagebox.showinfo("Success", f"Container {action} completed successfully.")
                    self._refresh_containers()
                else:
                    messagebox.showerror("Error", f"Failed to {action} container: {result.stderr}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to {action} container: {e}")

        threading.Thread(target=action_thread, daemon=True).start()

    def _show_logs(self):
        """Show container logs"""
        selection = self.containers_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a container.")
            return

        container_name = self.containers_tree.item(selection[0])['text']
        container_type = self.container_type_var.get()

        # Open terminal with logs
        terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
        for terminal in terminals:
            try:
                if terminal == 'gnome-terminal':
                    subprocess.Popen([terminal, '--title', f'{container_name} Logs', '--',
                                    container_type, 'logs', '-f', container_name])
                else:
                    subprocess.Popen([terminal, '-T', f'{container_name} Logs', '-e',
                                    container_type, 'logs', '-f', container_name])
                break
            except FileNotFoundError:
                continue

    def _list_images(self):
        """List container images"""
        def list_thread():
            try:
                container_type = self.container_type_var.get()
                cmd = [container_type, "images", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    messagebox.showinfo("Container Images", result.stdout)
                else:
                    messagebox.showerror("Error", f"Failed to list images: {result.stderr}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to list images: {e}")

        threading.Thread(target=list_thread, daemon=True).start()

    def _pull_image(self):
        """Pull container image"""
        image_name = simpledialog.askstring("Pull Image", "Enter image name (e.g., nginx:latest):")
        if image_name:
            def pull_thread():
                container_type = self.container_type_var.get()

                # Open terminal for pull
                terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
                for terminal in terminals:
                    try:
                        if terminal == 'gnome-terminal':
                            subprocess.Popen([terminal, '--title', f'Pulling {image_name}', '--',
                                            container_type, 'pull', image_name])
                        else:
                            subprocess.Popen([terminal, '-T', f'Pulling {image_name}', '-e',
                                            container_type, 'pull', image_name])
                        break
                    except FileNotFoundError:
                        continue

            threading.Thread(target=pull_thread, daemon=True).start()

    def _build_image(self):
        """Build container image"""
        dockerfile_dir = filedialog.askdirectory(title="Select directory containing Dockerfile")
        if dockerfile_dir:
            image_name = simpledialog.askstring("Build Image", "Enter image name:")
            if image_name:
                def build_thread():
                    container_type = self.container_type_var.get()

                    # Open terminal for build
                    terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
                    for terminal in terminals:
                        try:
                            if terminal == 'gnome-terminal':
                                subprocess.Popen([terminal, '--title', f'Building {image_name}',
                                                '--working-directory', dockerfile_dir, '--',
                                                container_type, 'build', '-t', image_name, '.'])
                            else:
                                subprocess.Popen([terminal, '-T', f'Building {image_name}', '-e',
                                                'bash', '-c',
                                                f'cd "{dockerfile_dir}" && {container_type} build -t {image_name} .'])
                            break
                        except FileNotFoundError:
                            continue

                threading.Thread(target=build_thread, daemon=True).start()

    def _open_terminal(self):
        """Open terminal in container context"""
        selection = self.containers_tree.selection()
        if selection:
            container_name = self.containers_tree.item(selection[0])['text']
            container_type = self.container_type_var.get()

            terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
            for terminal in terminals:
                try:
                    if terminal == 'gnome-terminal':
                        subprocess.Popen([terminal, '--title', f'{container_name} Shell', '--',
                                        container_type, 'exec', '-it', container_name, '/bin/bash'])
                    else:
                        subprocess.Popen([terminal, '-T', f'{container_name} Shell', '-e',
                                        container_type, 'exec', '-it', container_name, '/bin/bash'])
                    break
                except FileNotFoundError:
                    continue
        else:
            # Just open regular terminal
            terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
            for terminal in terminals:
                try:
                    subprocess.Popen([terminal])
                    break
                except FileNotFoundError:
                    continue


class ServicesManagerDialog:
    """Linux Services Manager Dialog"""

    def __init__(self, parent, services_data):
        self.parent = parent
        self.services_data = services_data
        self.dialog = None
        self.has_systemd = self._check_systemd()

    def _check_systemd(self) -> bool:
        """Check if systemd is available"""
        try:
            result = subprocess.run(["systemctl", "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def show(self):
        """Show services manager dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Linux Services Manager")
        self.dialog.geometry("1000x700")
        self.dialog.resizable(True, True)

        # Center the dialog
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        if not self.has_systemd:
            ttk.Label(self.dialog, text="systemd not available on this system",
                     foreground="red", font=("Arial", 12)).pack(pady=50)
            ttk.Button(self.dialog, text="Close", command=self.dialog.destroy).pack(pady=20)
            return

        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Filter frame
        filter_frame = ttk.LabelFrame(main_frame, text="Filter Services", padding=10)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=30)
        filter_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(filter_frame, text="Filter", command=self._filter_services).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(filter_frame, text="Show All", command=self._show_all_services).pack(side=tk.LEFT, padx=(0, 10))

        # Service type filters
        self.service_type_var = tk.StringVar(value="all")
        ttk.Radiobutton(filter_frame, text="All", variable=self.service_type_var,
                       value="all", command=self._refresh_services).pack(side=tk.LEFT, padx=(20, 5))
        ttk.Radiobutton(filter_frame, text="Active", variable=self.service_type_var,
                       value="active", command=self._refresh_services).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(filter_frame, text="Failed", variable=self.service_type_var,
                       value="failed", command=self._refresh_services).pack(side=tk.LEFT, padx=(0, 5))

        # Services list
        services_frame = ttk.LabelFrame(main_frame, text="Services", padding=10)
        services_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Services treeview
        self.services_tree = ttk.Treeview(services_frame,
                                        columns=('Load', 'Active', 'Sub', 'Description'),
                                        show='tree headings')
        self.services_tree.heading('#0', text='Service Name')
        self.services_tree.heading('Load', text='Load')
        self.services_tree.heading('Active', text='Active')
        self.services_tree.heading('Sub', text='Sub')
        self.services_tree.heading('Description', text='Description')

        self.services_tree.column('#0', width=250)
        self.services_tree.column('Load', width=80)
        self.services_tree.column('Active', width=80)
        self.services_tree.column('Sub', width=100)
        self.services_tree.column('Description', width=300)

        # Scrollbars
        services_scrolly = ttk.Scrollbar(services_frame, orient=tk.VERTICAL,
                                       command=self.services_tree.yview)
        services_scrollx = ttk.Scrollbar(services_frame, orient=tk.HORIZONTAL,
                                       command=self.services_tree.xview)

        self.services_tree.configure(yscrollcommand=services_scrolly.set,
                                   xscrollcommand=services_scrollx.set)

        self.services_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        services_scrolly.grid(row=0, column=1, sticky=(tk.N, tk.S))
        services_scrollx.grid(row=1, column=0, sticky=(tk.W, tk.E))

        services_frame.columnconfigure(0, weight=1)
        services_frame.rowconfigure(0, weight=1)

        # Control buttons
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(controls_frame, text="Start", command=self._start_service).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Stop", command=self._stop_service).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Restart", command=self._restart_service).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Enable", command=self._enable_service).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Disable", command=self._disable_service).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(controls_frame, text="Status", command=self._show_status).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls_frame, text="Logs", command=self._show_logs).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(controls_frame, text="Refresh", command=self._refresh_services).pack(side=tk.LEFT, padx=(0, 5))

        # Bottom buttons
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X)

        ttk.Button(bottom_frame, text="System Status", command=self._show_system_status).pack(side=tk.LEFT)
        ttk.Button(bottom_frame, text="Close", command=self.dialog.destroy).pack(side=tk.RIGHT)

        # Load services data
        self._load_services()

        # Bind Enter key to filter
        filter_entry.bind('<Return>', lambda e: self._filter_services())

    def _load_services(self):
        """Load services data into treeview"""
        # Clear current items
        for item in self.services_tree.get_children():
            self.services_tree.delete(item)

        # Parse services data
        lines = self.services_data.split('\n')
        for line in lines:
            if '.service' in line:
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0].replace('.service', '')
                    load = parts[1]
                    active = parts[2]
                    sub = parts[3]
                    description = ' '.join(parts[4:]) if len(parts) > 4 else ''

                    # Color coding based on status
                    tags = []
                    if active == 'active':
                        tags.append('active')
                    elif active == 'failed':
                        tags.append('failed')
                    elif active == 'inactive':
                        tags.append('inactive')

                    self.services_tree.insert('', tk.END, text=name,
                                            values=(load, active, sub, description),
                                            tags=tags)

        # Configure tags for color coding
        self.services_tree.tag_configure('active', foreground='green')
        self.services_tree.tag_configure('failed', foreground='red')
        self.services_tree.tag_configure('inactive', foreground='gray')

    def _filter_services(self):
        """Filter services by name"""
        filter_text = self.filter_var.get().lower()
        if not filter_text:
            self._show_all_services()
            return

        # Hide/show items based on filter
        for item in self.services_tree.get_children():
            service_name = self.services_tree.item(item)['text'].lower()
            if filter_text in service_name:
                self.services_tree.item(item, tags=self.services_tree.item(item)['tags'])
            else:
                # This is a simple way to hide items (move them to end and make invisible)
                pass  # Tkinter treeview doesn't have direct hide functionality

    def _show_all_services(self):
        """Show all services"""
        self.filter_var.set("")
        self._load_services()

    def _refresh_services(self):
        """Refresh services list"""
        service_type = self.service_type_var.get()

        def refresh_thread():
            try:
                cmd = ["systemctl", "list-units", "--type=service", "--no-pager"]

                if service_type == "active":
                    cmd.extend(["--state=active"])
                elif service_type == "failed":
                    cmd.extend(["--state=failed"])

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    self.services_data = result.stdout
                    self._load_services()
                else:
                    messagebox.showerror("Error", f"Failed to refresh services: {result.stderr}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to refresh services: {e}")

        threading.Thread(target=refresh_thread, daemon=True).start()

    def _service_action(self, action: str, confirm_message: str = None):
        """Perform service action"""
        selection = self.services_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a service.")
            return

        service_name = self.services_tree.item(selection[0])['text']

        if confirm_message and not messagebox.askyesno("Confirm Action", confirm_message.format(service_name)):
            return

        def action_thread():
            try:
                cmd = ["sudo", "systemctl", action, f"{service_name}.service"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    messagebox.showinfo("Success", f"Service {action} completed successfully.")
                    self._refresh_services()
                else:
                    messagebox.showerror("Error", f"Failed to {action} service: {result.stderr}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to {action} service: {e}")

        threading.Thread(target=action_thread, daemon=True).start()

    def _start_service(self):
        """Start selected service"""
        self._service_action("start", "Start service '{}'?")

    def _stop_service(self):
        """Stop selected service"""
        self._service_action("stop", "Stop service '{}'?")

    def _restart_service(self):
        """Restart selected service"""
        self._service_action("restart", "Restart service '{}'?")

    def _enable_service(self):
        """Enable selected service"""
        self._service_action("enable", "Enable service '{}' to start on boot?")

    def _disable_service(self):
        """Disable selected service"""
        self._service_action("disable", "Disable service '{}' from starting on boot?")

    def _show_status(self):
        """Show service status"""
        selection = self.services_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a service.")
            return

        service_name = self.services_tree.item(selection[0])['text']

        def status_thread():
            try:
                cmd = ["systemctl", "status", f"{service_name}.service", "--no-pager"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                # Create status window
                status_window = tk.Toplevel(self.dialog)
                status_window.title(f"Service Status: {service_name}")
                status_window.geometry("800x600")

                text_widget = scrolledtext.ScrolledText(status_window, wrap=tk.WORD,
                                                      font=('Courier', 10))
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                text_widget.insert(tk.END, result.stdout)
                text_widget.configure(state=tk.DISABLED)

                ttk.Button(status_window, text="Close",
                          command=status_window.destroy).pack(pady=5)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to get service status: {e}")

        threading.Thread(target=status_thread, daemon=True).start()

    def _show_logs(self):
        """Show service logs"""
        selection = self.services_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a service.")
            return

        service_name = self.services_tree.item(selection[0])['text']

        # Open terminal with logs
        terminals = ['gnome-terminal', 'konsole', 'xfce4-terminal', 'xterm']
        for terminal in terminals:
            try:
                if terminal == 'gnome-terminal':
                    subprocess.Popen([terminal, '--title', f'{service_name} Logs', '--',
                                    'journalctl', '-u', f'{service_name}.service', '-f'])
                else:
                    subprocess.Popen([terminal, '-T', f'{service_name} Logs', '-e',
                                    'journalctl', '-u', f'{service_name}.service', '-f'])
                break
            except FileNotFoundError:
                continue

    def _show_system_status(self):
        """Show overall system status"""
        def status_thread():
            try:
                cmd = ["systemctl", "status", "--no-pager"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                # Create status window
                status_window = tk.Toplevel(self.dialog)
                status_window.title("System Status")
                status_window.geometry("900x700")

                text_widget = scrolledtext.ScrolledText(status_window, wrap=tk.WORD,
                                                      font=('Courier', 10))
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                text_widget.insert(tk.END, result.stdout)
                text_widget.configure(state=tk.DISABLED)

                ttk.Button(status_window, text="Close",
                          command=status_window.destroy).pack(pady=5)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to get system status: {e}")

        threading.Thread(target=status_thread, daemon=True).start()


class SearchResultsDialog:
    """Generic search results dialog"""

    def __init__(self, parent, title, content):
        self.parent = parent
        self.title = title
        self.content = content
        self.dialog = None

    def show(self):
        """Show search results dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(self.title)
        self.dialog.geometry("900x600")
        self.dialog.resizable(True, True)

        # Center the dialog
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Content text widget
        self.text_widget = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD,
                                                   font=('Courier', 10))
        self.text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Insert content
        self.text_widget.insert(tk.END, self.content)
        self.text_widget.configure(state=tk.DISABLED)

        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X)

        ttk.Button(buttons_frame, text="Copy to Clipboard",
                  command=self._copy_to_clipboard).pack(side=tk.LEFT)
        ttk.Button(buttons_frame, text="Save to File",
                  command=self._save_to_file).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(buttons_frame, text="Close",
                  command=self.dialog.destroy).pack(side=tk.RIGHT)

    def _copy_to_clipboard(self):
        """Copy content to clipboard"""
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(self.content)
        messagebox.showinfo("Copied", "Content copied to clipboard.")

    def _save_to_file(self):
        """Save content to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.content)
                messagebox.showinfo("Saved", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")