"""
Terminal Coder - Windows Implementation
Windows-specific modules and functionality
"""

from .main import TerminalCoderApp
from .system_manager import WindowsSystemManager
from .ai_integration import WindowsAIIntegration
from .project_manager import WindowsProjectManager
from .gui import WindowsGUI

__all__ = [
    'TerminalCoderApp',
    'WindowsSystemManager',
    'WindowsAIIntegration',
    'WindowsProjectManager',
    'WindowsGUI'
]