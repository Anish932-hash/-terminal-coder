#!/bin/bash

# Terminal Coder Linux Installation Script
# Advanced AI-Powered Development Terminal for Linux

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
MIN_PYTHON_VERSION="3.8"
REQUIRED_SPACE_MB=500
INSTALL_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/terminal-coder"
DATA_DIR="$HOME/.local/share/terminal-coder"

# Print formatted messages
print_header() {
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} $1"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Linux
check_linux() {
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "This installer is designed for Linux systems only."
        print_info "Detected OS: $OSTYPE"
        exit 1
    fi
    print_success "Linux system detected"
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
        print_info "Detected distribution: $PRETTY_NAME"
    else
        print_warning "Cannot detect Linux distribution"
        DISTRO="unknown"
        DISTRO_VERSION="unknown"
    fi
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."

    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python $PYTHON_VERSION found (>= $MIN_PYTHON_VERSION)"
        else
            print_error "Python $PYTHON_VERSION is too old. Required: >= $MIN_PYTHON_VERSION"
            install_python
        fi
    else
        print_error "Python 3 not found"
        install_python
    fi

    # Check pip
    if command -v pip3 &> /dev/null || python3 -m pip --version &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip not found"
        install_pip
    fi

    # Check available disk space
    AVAILABLE_SPACE=$(df "$HOME" | tail -1 | awk '{print $4}')
    if [ "$AVAILABLE_SPACE" -gt $((REQUIRED_SPACE_MB * 1024)) ]; then
        print_success "Sufficient disk space available"
    else
        print_warning "Low disk space. Required: ${REQUIRED_SPACE_MB}MB"
    fi

    # Check for optional dependencies
    check_optional_dependencies
}

# Install Python based on distribution
install_python() {
    print_info "Installing Python 3..."

    case $DISTRO in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv python3-dev
            ;;
        fedora|rhel|centos)
            if command -v dnf &> /dev/null; then
                sudo dnf install -y python3 python3-pip python3-devel
            else
                sudo yum install -y python3 python3-pip python3-devel
            fi
            ;;
        opensuse*|suse)
            sudo zypper install -y python3 python3-pip python3-devel
            ;;
        arch|manjaro)
            sudo pacman -S --noconfirm python python-pip
            ;;
        alpine)
            sudo apk add python3 python3-dev py3-pip
            ;;
        *)
            print_error "Unsupported distribution for automatic Python installation: $DISTRO"
            print_info "Please install Python 3.8+ manually and run this script again"
            exit 1
            ;;
    esac

    print_success "Python installed"
}

# Install pip
install_pip() {
    print_info "Installing pip..."

    if command -v python3 &> /dev/null; then
        python3 -m ensurepip --upgrade
    else
        case $DISTRO in
            ubuntu|debian)
                sudo apt install -y python3-pip
                ;;
            fedora|rhel|centos)
                if command -v dnf &> /dev/null; then
                    sudo dnf install -y python3-pip
                else
                    sudo yum install -y python3-pip
                fi
                ;;
            *)
                curl -sS https://bootstrap.pypa.io/get-pip.py | python3
                ;;
        esac
    fi

    print_success "pip installed"
}

# Check for optional dependencies
check_optional_dependencies() {
    print_info "Checking optional dependencies..."

    # Git
    if command -v git &> /dev/null; then
        print_success "Git found: $(git --version)"
    else
        print_warning "Git not found - version control features will be limited"
        install_optional_dependency "git"
    fi

    # Docker
    if command -v docker &> /dev/null; then
        print_success "Docker found: $(docker --version)"
    else
        print_info "Docker not found - containerization features will be limited"
    fi

    # Node.js (for JavaScript/TypeScript projects)
    if command -v node &> /dev/null; then
        print_success "Node.js found: $(node --version)"
    else
        print_info "Node.js not found - JavaScript/TypeScript templates will have limited functionality"
    fi

    # Rust (for Rust projects)
    if command -v rustc &> /dev/null; then
        print_success "Rust found: $(rustc --version)"
    else
        print_info "Rust not found - Rust templates will not be available"
    fi

    # Go (for Go projects)
    if command -v go &> /dev/null; then
        print_success "Go found: $(go version)"
    else
        print_info "Go not found - Go templates will not be available"
    fi
}

# Install optional dependencies
install_optional_dependency() {
    local dep=$1

    read -p "Do you want to install $dep? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installing $dep..."

        case $DISTRO in
            ubuntu|debian)
                sudo apt install -y "$dep"
                ;;
            fedora|rhel|centos)
                if command -v dnf &> /dev/null; then
                    sudo dnf install -y "$dep"
                else
                    sudo yum install -y "$dep"
                fi
                ;;
            opensuse*|suse)
                sudo zypper install -y "$dep"
                ;;
            arch|manjaro)
                sudo pacman -S --noconfirm "$dep"
                ;;
            alpine)
                sudo apk add "$dep"
                ;;
        esac

        print_success "$dep installed"
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating directories..."

    mkdir -p "$INSTALL_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$HOME/.cache/terminal-coder"

    # Set secure permissions
    chmod 700 "$CONFIG_DIR"
    chmod 755 "$DATA_DIR"
    chmod 755 "$HOME/.cache/terminal-coder"

    print_success "Directories created"
}

# Install Terminal Coder
install_terminal_coder() {
    print_info "Installing Terminal Coder..."

    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    # Install from current directory or download
    if [ -f "../setup.py" ]; then
        print_info "Installing from local source..."
        cd ..
        python3 -m pip install --user -e .
    else
        print_info "Downloading and installing from repository..."
        git clone https://github.com/terminalcoder/terminal-coder-linux.git
        cd terminal-coder-linux
        python3 -m pip install --user .
    fi

    # Cleanup
    cd "$HOME"
    rm -rf "$TEMP_DIR"

    print_success "Terminal Coder installed"
}

# Create desktop entry
create_desktop_entry() {
    print_info "Creating desktop entry..."

    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"

    cat > "$DESKTOP_DIR/terminal-coder.desktop" << EOF
[Desktop Entry]
Name=Terminal Coder
Comment=Advanced AI-Powered Development Terminal
Exec=$INSTALL_DIR/terminal-coder
Icon=terminal
Terminal=true
Type=Application
Categories=Development;IDE;TextEditor;Programming;
Keywords=ai;coding;development;terminal;programming;python;rust;go;javascript;
StartupNotify=true
EOF

    chmod +x "$DESKTOP_DIR/terminal-coder.desktop"

    print_success "Desktop entry created"
}

# Setup shell completion
setup_shell_completion() {
    print_info "Setting up shell completion..."

    # Bash completion
    if [ -n "$BASH_VERSION" ]; then
        BASH_COMPLETION_DIR="$HOME/.bash_completion.d"
        mkdir -p "$BASH_COMPLETION_DIR"

        cat > "$BASH_COMPLETION_DIR/terminal-coder" << 'EOF'
_terminal_coder_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="--help --version --config-dir --data-dir --project --ai-provider --debug --verbose"

    if [[ ${cur} == -* ]]; then
        COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
        return 0
    fi

    case "${prev}" in
        --ai-provider)
            COMPREPLY=($(compgen -W "openai anthropic google cohere" -- ${cur}))
            return 0
            ;;
    esac
}

complete -F _terminal_coder_completion terminal-coder
complete -F _terminal_coder_completion tcoder
complete -F _terminal_coder_completion tc
EOF

        # Add to bashrc if not already present
        if ! grep -q "bash_completion.d/terminal-coder" "$HOME/.bashrc" 2>/dev/null; then
            echo "[ -f ~/.bash_completion.d/terminal-coder ] && source ~/.bash_completion.d/terminal-coder" >> "$HOME/.bashrc"
        fi

        print_success "Bash completion configured"
    fi

    # Zsh completion
    if [ -n "$ZSH_VERSION" ]; then
        ZSH_COMPLETION_DIR="$HOME/.oh-my-zsh/completions"
        if [ ! -d "$ZSH_COMPLETION_DIR" ]; then
            ZSH_COMPLETION_DIR="$HOME/.zfunc"
            mkdir -p "$ZSH_COMPLETION_DIR"
        fi

        cat > "$ZSH_COMPLETION_DIR/_terminal-coder" << 'EOF'
#compdef terminal-coder tcoder tc

_terminal_coder() {
    local state line

    _arguments \
        '--help[Show help message]' \
        '--version[Show version information]' \
        '--config-dir[Configuration directory]:directory:_directories' \
        '--data-dir[Data directory]:directory:_directories' \
        '--project[Open specific project]:project:' \
        '--ai-provider[AI provider]:provider:(openai anthropic google cohere)' \
        '--debug[Enable debug mode]' \
        '--verbose[Verbose output]'
}

_terminal_coder "$@"
EOF

        print_success "Zsh completion configured"
    fi
}

# Setup systemd user service
setup_systemd_service() {
    print_info "Setting up systemd user service..."

    if command -v systemctl &> /dev/null && [ -d "$HOME/.config/systemd/user" ] || mkdir -p "$HOME/.config/systemd/user"; then
        cat > "$HOME/.config/systemd/user/terminal-coder.service" << EOF
[Unit]
Description=Terminal Coder Background Service
After=graphical-session.target

[Service]
Type=simple
ExecStart=$INSTALL_DIR/terminal-coder --daemon
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

        print_success "Systemd service created"
        print_info "To enable the service, run: systemctl --user enable terminal-coder.service"
    else
        print_warning "Systemd not available, skipping service setup"
    fi
}

# Configure environment
configure_environment() {
    print_info "Configuring environment..."

    # Add to PATH if not already present
    if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
        # Add to shell rc files
        for rcfile in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
            if [ -f "$rcfile" ] && ! grep -q "$INSTALL_DIR" "$rcfile"; then
                echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$rcfile"
                print_info "Added to PATH in $rcfile"
            fi
        done
    fi

    print_success "Environment configured"
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."

    if command -v terminal-coder &> /dev/null; then
        VERSION=$(terminal-coder --version 2>&1 | head -1 || echo "Unknown version")
        print_success "Installation verified: $VERSION"
        return 0
    else
        print_error "Installation verification failed"
        print_info "Try running: export PATH=\"$INSTALL_DIR:\$PATH\""
        return 1
    fi
}

# Display post-installation information
show_post_install_info() {
    print_header "🎉 Terminal Coder Installation Complete!"

    echo -e "${GREEN}Terminal Coder has been successfully installed!${NC}\n"

    echo -e "${CYAN}📍 Installation Details:${NC}"
    echo -e "   • Executable: ${YELLOW}$INSTALL_DIR/terminal-coder${NC}"
    echo -e "   • Configuration: ${YELLOW}$CONFIG_DIR${NC}"
    echo -e "   • Data: ${YELLOW}$DATA_DIR${NC}"
    echo -e "   • Desktop Entry: Created"
    echo -e "   • Shell Completion: Configured"

    echo -e "\n${CYAN}🚀 Getting Started:${NC}"
    echo -e "   1. Open a new terminal or run: ${YELLOW}source ~/.bashrc${NC}"
    echo -e "   2. Start Terminal Coder: ${YELLOW}terminal-coder${NC}"
    echo -e "   3. Configure your AI API keys in the settings"
    echo -e "   4. Create your first project!"

    echo -e "\n${CYAN}📚 Available Commands:${NC}"
    echo -e "   • ${YELLOW}terminal-coder${NC} - Full interface"
    echo -e "   • ${YELLOW}tcoder${NC} - Short alias"
    echo -e "   • ${YELLOW}tc${NC} - Shortest alias"

    echo -e "\n${CYAN}🔧 Configuration:${NC}"
    echo -e "   • Run ${YELLOW}terminal-coder --help${NC} for options"
    echo -e "   • Edit ${YELLOW}$CONFIG_DIR/config.json${NC} for advanced settings"

    echo -e "\n${CYAN}🆘 Support:${NC}"
    echo -e "   • Documentation: ${BLUE}https://docs.terminalcoder.linux${NC}"
    echo -e "   • Issues: ${BLUE}https://github.com/terminalcoder/terminal-coder-linux/issues${NC}"

    echo -e "\n${GREEN}Happy coding with Terminal Coder! 🐧${NC}"
}

# Handle errors
error_handler() {
    local line_no=$1
    print_error "Installation failed at line $line_no"
    print_info "Check the error messages above for details"
    print_info "You can run this script again after resolving the issues"
    exit 1
}

# Cleanup on exit
cleanup() {
    # Remove any temporary files if they exist
    if [ -n "${TEMP_DIR:-}" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Main installation function
main() {
    # Set up error handling
    trap 'error_handler ${LINENO}' ERR
    trap cleanup EXIT

    print_header "🐧 Terminal Coder Linux Installation"

    print_info "This installer will set up Terminal Coder on your Linux system"
    print_info "Installation directory: $INSTALL_DIR"
    print_info "Configuration directory: $CONFIG_DIR"

    # Confirm installation
    read -p "Do you want to continue? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi

    # Run installation steps
    check_linux
    detect_distro
    check_requirements
    create_directories
    install_terminal_coder
    create_desktop_entry
    setup_shell_completion
    setup_systemd_service
    configure_environment

    if verify_installation; then
        show_post_install_info
    else
        print_error "Installation completed but verification failed"
        print_info "You may need to restart your terminal or run: source ~/.bashrc"
        exit 1
    fi
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this installer as root"
    print_info "Terminal Coder should be installed in user space"
    exit 1
fi

# Run main installation
main "$@"