# Terminal Coder - Cross-Platform Architecture

## Overview

Terminal Coder has been redesigned as a truly cross-platform AI-powered development terminal that supports both Windows and Linux operating systems. The architecture uses OS detection and intelligent routing to provide native experiences on each platform while maintaining identical functionality.

## Architecture Components

### 1. OS Detection System (`os_detector.py`)
- **Purpose**: Automatically detects the operating system and available platform features
- **Features**:
  - Detects Windows, Linux, and macOS
  - Identifies architecture (x64, ARM, etc.)
  - Checks for platform-specific features and dependencies
  - Provides unified interface for OS information

### 2. Cross-Platform Launcher (`main_cross_platform.py`)
- **Purpose**: Universal entry point that routes to appropriate OS implementation
- **Features**:
  - Displays unified startup banner with OS information
  - Dependency checking and installation suggestions
  - Error handling with platform-specific guidance
  - Command-line argument parsing

### 3. Platform-Specific Implementations

#### Windows Implementation (`windows/`)
- **main.py**: Windows-optimized main application with Windows Terminal integration
- **system_manager.py**: Windows services, registry, WMI integration
- **ai_integration.py**: Windows credential store, encrypted API keys
- **project_manager.py**: PowerShell scripts, batch files, Windows dev templates
- **gui.py**: Windows-native GUI with taskbar integration

#### Linux Implementation (`linux/`)
- **main.py**: Linux-optimized main application with systemd integration
- **system_manager.py**: systemd services, package managers, D-Bus
- **ai_integration.py**: Linux keyring, XDG Base Directory compliance
- **project_manager.py**: systemd services, .desktop files, man pages
- **gui.py**: Linux desktop environment integration with D-Bus notifications

## Key Features by Platform

### Windows-Specific Features
- **Windows Terminal Integration**: Profile management and custom themes
- **PowerShell Automation**: Advanced scripting and system management
- **Registry Access**: Configuration and system settings management
- **WMI Integration**: Comprehensive system information and monitoring
- **Windows Services**: Service creation, management, and monitoring
- **Windows Credential Store**: Secure API key storage
- **Taskbar Integration**: System tray notifications and shortcuts
- **Windows-Specific Project Templates**: .NET, PowerShell, Windows services

### Linux-Specific Features
- **systemd Integration**: Service creation, management, and monitoring
- **D-Bus Communication**: Desktop notifications and system integration
- **Package Manager Support**: apt, yum, pacman, zypper integration
- **XDG Base Directory**: Standards-compliant configuration storage
- **Desktop Environment Detection**: GNOME, KDE, XFCE optimization
- **Linux Keyring**: Secure credential storage with libsecret
- **Manual Pages**: Automatic man page generation for projects
- **Container Support**: Docker and Podman integration

### Shared Features
- **AI Integration**: OpenAI, Anthropic, Google, Cohere support
- **Project Management**: Template-based project creation and management
- **Code Analysis**: Multi-provider AI code analysis and suggestions
- **Rich Terminal UI**: Consistent user interface across platforms
- **Encrypted Storage**: Secure API key and configuration storage
- **Version Control**: Git integration with platform-specific optimizations
- **Multi-Language Support**: Python, JavaScript, TypeScript, Rust, Go, C++, and more

## Installation and Usage

### Prerequisites
- Python 3.10+ (recommended: Python 3.12+)
- Platform-specific dependencies (automatically detected and installed)

### Installation
```bash
# Clone or download the project
cd terminal_coder

# Install dependencies (automatically detects platform)
pip install -r requirements.txt

# Run the cross-platform launcher
python main_cross_platform.py
```

### Platform Detection
The system automatically detects your platform and loads the appropriate implementation:
- **Windows**: Loads `windows.main.TerminalCoderApp`
- **Linux**: Loads `linux.main.TerminalCoderApp`
- **macOS**: Uses Linux implementation with macOS-specific adaptations

## Development Guidelines

### Adding New Features
1. Implement the feature in both `windows/` and `linux/` folders
2. Ensure identical API signatures and functionality
3. Use platform-specific optimizations where beneficial
4. Update tests for both platforms
5. Document platform-specific considerations

### Code Organization
```
terminal_coder/
├── main_cross_platform.py    # Universal launcher
├── os_detector.py            # OS detection system
├── requirements.txt          # All dependencies with platform markers
├── windows/                  # Windows-specific implementation
│   ├── main.py              # Windows main application
│   ├── system_manager.py    # Windows system integration
│   ├── ai_integration.py    # Windows AI integration
│   ├── project_manager.py   # Windows project management
│   └── gui.py               # Windows GUI
├── linux/                   # Linux-specific implementation
│   ├── main.py              # Linux main application
│   ├── system_manager.py    # Linux system integration
│   ├── ai_integration.py    # Linux AI integration
│   ├── project_manager.py   # Linux project management
│   └── gui.py               # Linux GUI
└── tests/                   # Cross-platform tests
    ├── test_windows/        # Windows-specific tests
    └── test_linux/          # Linux-specific tests
```

## Testing Strategy

### Automated Testing
- **Unit Tests**: Test individual components on each platform
- **Integration Tests**: Test cross-platform functionality
- **Platform Tests**: Test platform-specific features

### Manual Testing
- **Windows**: Test on Windows 10, 11 with PowerShell and cmd
- **Linux**: Test on Ubuntu, Fedora, Arch with different desktop environments
- **Cross-Platform**: Verify identical functionality across platforms

## Deployment

### Windows Deployment
- **Executable**: Create standalone .exe with PyInstaller
- **MSI Installer**: Windows Installer package
- **Store**: Microsoft Store submission
- **PowerShell Module**: PowerShell Gallery publication

### Linux Deployment
- **AppImage**: Portable Linux application
- **Package Managers**: Create .deb, .rpm, and AUR packages
- **Snap/Flatpak**: Universal Linux packages
- **Docker**: Container-based deployment

## Configuration

### Cross-Platform Configuration
- **Windows**: `%APPDATA%\terminal-coder\`
- **Linux**: `$XDG_CONFIG_HOME/terminal-coder/` or `~/.config/terminal-coder/`
- **Shared**: JSON configuration files with platform-specific overrides

### Platform-Specific Settings
Each platform maintains its own configuration for:
- System integration preferences
- Default applications and tools
- Platform-specific feature toggles
- Native UI preferences

## Security Considerations

### Credential Storage
- **Windows**: Windows Credential Manager integration
- **Linux**: libsecret/keyring integration with encryption fallback
- **Encryption**: AES-256 encryption for sensitive data
- **Permissions**: Platform-appropriate file permissions (0600 on Unix)

### Code Execution
- **Sandboxing**: Platform-appropriate sandboxing for executed code
- **Validation**: Input validation and sanitization
- **Permissions**: Minimal required permissions for operation

## Performance Optimization

### Startup Time
- **Lazy Loading**: Import modules only when needed
- **Caching**: Cache OS detection and feature detection results
- **Async Operations**: Non-blocking initialization

### Memory Usage
- **Efficient Data Structures**: Use appropriate data structures for each platform
- **Memory Pools**: Reuse objects where possible
- **Garbage Collection**: Optimize garbage collection for long-running processes

## Future Enhancements

### Planned Features
- **macOS Native Support**: Full macOS-specific implementation
- **Remote Development**: SSH and remote container development
- **Plugin System**: Third-party plugin architecture
- **Web Interface**: Browser-based interface option
- **Mobile Companion**: Mobile app for monitoring and basic operations

### Integration Opportunities
- **IDE Integration**: VSCode, PyCharm, IntelliJ plugins
- **CI/CD Integration**: GitHub Actions, GitLab CI, Jenkins plugins
- **Cloud Services**: AWS, Azure, GCP integration
- **Container Orchestration**: Kubernetes, Docker Swarm support

## Contributing

### Development Setup
1. Clone the repository
2. Set up virtual environment
3. Install development dependencies
4. Run platform-specific tests
5. Follow code style guidelines

### Code Style
- **Python**: PEP 8 compliance with Black formatting
- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings
- **Testing**: Minimum 80% code coverage

This architecture ensures that Terminal Coder provides a native, optimized experience on each platform while maintaining consistent functionality and a unified user experience across all supported operating systems.