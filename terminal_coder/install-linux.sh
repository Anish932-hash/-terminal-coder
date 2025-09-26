#!/bin/bash
# Terminal Coder Linux Installation Script
# Optimized installer for Linux systems with distribution detection

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
TERMINAL_CODER_DIR="$HOME/.terminal_coder"
VENV_DIR="$TERMINAL_CODER_DIR/venv"
REPO_URL="https://github.com/terminalcoder/terminal_coder.git"
MIN_PYTHON_VERSION="3.10"

# Logging
LOG_FILE="$HOME/terminal-coder-install.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

print_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🐧 TERMINAL CODER LINUX INSTALLER                         ║
║                Advanced AI-Powered Development Terminal                       ║
║                      Optimized for Linux Systems                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

detect_distribution() {
    local distro=""

    if command -v lsb_release &> /dev/null; then
        distro=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
    elif [[ -f /etc/os-release ]]; then
        distro=$(grep '^ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"' | tr '[:upper:]' '[:lower:]')
    elif [[ -f /etc/debian_version ]]; then
        distro="debian"
    elif [[ -f /etc/redhat-release ]]; then
        distro="rhel"
    elif [[ -f /etc/arch-release ]]; then
        distro="arch"
    else
        distro="unknown"
    fi

    echo "$distro"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        echo -e "${RED}❌ Please don't run this script as root${NC}"
        echo -e "${YELLOW}   Run as regular user - sudo will be used when needed${NC}"
        exit 1
    fi
}

check_python_version() {
    local python_cmd=""

    # Try different Python commands
    for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            local version=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ $(echo "$version >= $MIN_PYTHON_VERSION" | bc -l) -eq 1 ]]; then
                python_cmd="$cmd"
                break
            fi
        fi
    done

    if [[ -z "$python_cmd" ]]; then
        echo -e "${RED}❌ Python $MIN_PYTHON_VERSION or higher is required${NC}"
        echo -e "${YELLOW}   Please install Python $MIN_PYTHON_VERSION+ and try again${NC}"
        exit 1
    fi

    echo "$python_cmd"
}

install_system_dependencies() {
    local distro=$1
    echo -e "${BLUE}📦 Installing system dependencies for $distro...${NC}"

    case "$distro" in
        ubuntu|debian)
            log "Installing dependencies for Debian/Ubuntu"
            sudo apt update
            sudo apt install -y \
                python3-dev python3-pip python3-venv \
                git curl wget unzip \
                build-essential libffi-dev libssl-dev \
                libdbus-1-dev libdbus-glib-1-dev \
                python3-tk \
                htop tree jq tmux vim \
                bc
            ;;
        fedora)
            log "Installing dependencies for Fedora"
            sudo dnf update -y
            sudo dnf install -y \
                python3-devel python3-pip python3-virtualenv \
                git curl wget unzip \
                gcc gcc-c++ make libffi-devel openssl-devel \
                dbus-devel dbus-glib-devel \
                python3-tkinter \
                htop tree jq tmux vim \
                bc
            ;;
        centos|rhel)
            log "Installing dependencies for CentOS/RHEL"
            sudo yum update -y
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                python3-devel python3-pip \
                git curl wget unzip \
                libffi-devel openssl-devel \
                dbus-devel dbus-glib-devel \
                htop tree jq tmux vim \
                bc
            ;;
        arch|manjaro)
            log "Installing dependencies for Arch/Manjaro"
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm \
                python python-pip python-virtualenv \
                git curl wget unzip \
                base-devel libffi openssl \
                dbus dbus-glib \
                tk \
                htop tree jq tmux vim \
                bc
            ;;
        opensuse*)
            log "Installing dependencies for openSUSE"
            sudo zypper refresh
            sudo zypper install -y \
                python3-devel python3-pip python3-virtualenv \
                git curl wget unzip \
                gcc gcc-c++ make libffi-devel libopenssl-devel \
                dbus-1-devel dbus-1-glib-devel \
                python3-tk \
                htop tree jq tmux vim \
                bc
            ;;
        alpine)
            log "Installing dependencies for Alpine"
            sudo apk update
            sudo apk add \
                python3 python3-dev py3-pip py3-virtualenv \
                git curl wget unzip \
                build-base libffi-dev openssl-dev \
                dbus-dev dbus-glib-dev \
                py3-tkinter \
                htop tree jq tmux vim \
                bc
            ;;
        *)
            echo -e "${YELLOW}⚠️  Unknown distribution: $distro${NC}"
            echo -e "${YELLOW}   Attempting generic installation...${NC}"
            # Try generic commands
            if command -v apt &> /dev/null; then
                sudo apt update && sudo apt install -y python3-dev python3-pip git curl wget build-essential
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y python3-devel python3-pip git curl wget gcc
            elif command -v yum &> /dev/null; then
                sudo yum install -y python3-devel python3-pip git curl wget gcc
            else
                echo -e "${RED}❌ Could not determine package manager${NC}"
                exit 1
            fi
            ;;
    esac

    log "System dependencies installed successfully"
}

create_virtual_environment() {
    local python_cmd=$1

    echo -e "${BLUE}🐍 Creating Python virtual environment...${NC}"

    # Create Terminal Coder directory
    mkdir -p "$TERMINAL_CODER_DIR"

    # Create virtual environment
    "$python_cmd" -m venv "$VENV_DIR"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip setuptools wheel

    log "Virtual environment created successfully"
}

install_terminal_coder() {
    echo -e "${BLUE}⬇️  Installing Terminal Coder...${NC}"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Install from PyPI (when available) or from source
    if pip show terminal-coder-linux &> /dev/null; then
        echo -e "${GREEN}Installing from PyPI...${NC}"
        pip install --upgrade terminal-coder-linux[all]
    else
        echo -e "${YELLOW}PyPI package not available, installing from source...${NC}"

        # Clone repository
        local temp_dir=$(mktemp -d)
        git clone "$REPO_URL" "$temp_dir"
        cd "$temp_dir"

        # Install in development mode
        pip install -e .[dev,quality,typing]

        # Clean up
        cd "$HOME"
        rm -rf "$temp_dir"
    fi

    log "Terminal Coder installed successfully"
}

setup_shell_integration() {
    echo -e "${BLUE}🐚 Setting up shell integration...${NC}"

    local shell_name=$(basename "$SHELL")
    local completion_installed=false

    case "$shell_name" in
        bash)
            if [[ -f "$HOME/.bashrc" ]]; then
                # Add to .bashrc if not already present
                local completion_line='eval "$(_TERMINAL_CODER_COMPLETE=bash_source terminal-coder)"'
                if ! grep -q "terminal-coder" "$HOME/.bashrc"; then
                    echo "" >> "$HOME/.bashrc"
                    echo "# Terminal Coder completion and aliases" >> "$HOME/.bashrc"
                    echo "$completion_line" >> "$HOME/.bashrc"
                    echo "alias tcoder='terminal-coder'" >> "$HOME/.bashrc"
                    completion_installed=true
                fi
            fi
            ;;
        zsh)
            if [[ -f "$HOME/.zshrc" ]]; then
                local completion_line='eval "$(_TERMINAL_CODER_COMPLETE=zsh_source terminal-coder)"'
                if ! grep -q "terminal-coder" "$HOME/.zshrc"; then
                    echo "" >> "$HOME/.zshrc"
                    echo "# Terminal Coder completion and aliases" >> "$HOME/.zshrc"
                    echo "$completion_line" >> "$HOME/.zshrc"
                    echo "alias tcoder='terminal-coder'" >> "$HOME/.zshrc"
                    completion_installed=true
                fi
            fi
            ;;
        fish)
            local fish_config_dir="$HOME/.config/fish"
            local completions_dir="$fish_config_dir/completions"
            mkdir -p "$completions_dir"

            # Create fish completion file
            cat > "$completions_dir/terminal-coder.fish" << 'EOF'
complete -c terminal-coder -f
complete -c terminal-coder -s h -l help -d "Show help"
complete -c terminal-coder -l version -d "Show version"
complete -c terminal-coder -l tui -d "Launch TUI mode"
complete -c terminal-coder -s p -l project -d "Project path"
EOF

            # Add alias to fish config
            if [[ -f "$fish_config_dir/config.fish" ]]; then
                if ! grep -q "terminal-coder" "$fish_config_dir/config.fish"; then
                    echo "" >> "$fish_config_dir/config.fish"
                    echo "# Terminal Coder alias" >> "$fish_config_dir/config.fish"
                    echo "alias tcoder='terminal-coder'" >> "$fish_config_dir/config.fish"
                fi
            fi
            completion_installed=true
            ;;
    esac

    if [[ "$completion_installed" == true ]]; then
        log "Shell integration setup completed for $shell_name"
    else
        log "Shell integration setup skipped - unsupported shell or missing config files"
    fi
}

create_desktop_entry() {
    echo -e "${BLUE}🖥️  Creating desktop entry...${NC}"

    local desktop_dir="$HOME/.local/share/applications"
    mkdir -p "$desktop_dir"

    cat > "$desktop_dir/terminal-coder.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Terminal Coder
Comment=Advanced AI-Powered Development Terminal
Exec=$VENV_DIR/bin/terminal-coder --tui
Icon=utilities-terminal
Terminal=true
Categories=Development;IDE;
Keywords=terminal;coding;ai;development;
StartupNotify=true
EOF

    # Make executable
    chmod +x "$desktop_dir/terminal-coder.desktop"

    log "Desktop entry created"
}

create_bin_symlink() {
    echo -e "${BLUE}🔗 Creating system-wide symlink...${NC}"

    local bin_dir="$HOME/.local/bin"
    mkdir -p "$bin_dir"

    # Create symlink to terminal-coder executable
    ln -sf "$VENV_DIR/bin/terminal-coder" "$bin_dir/terminal-coder"
    ln -sf "$VENV_DIR/bin/terminal-coder" "$bin_dir/tcoder"

    # Add to PATH if not already there
    local shell_name=$(basename "$SHELL")
    case "$shell_name" in
        bash)
            if [[ -f "$HOME/.bashrc" ]] && ! grep -q ".local/bin" "$HOME/.bashrc"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            fi
            ;;
        zsh)
            if [[ -f "$HOME/.zshrc" ]] && ! grep -q ".local/bin" "$HOME/.zshrc"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
            fi
            ;;
    esac

    log "Binary symlinks created"
}

run_initial_setup() {
    echo -e "${BLUE}⚙️  Running initial setup...${NC}"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Run doctor command to check installation
    terminal-coder doctor || true

    # Create initial configuration
    terminal-coder config --show > /dev/null 2>&1 || true

    log "Initial setup completed"
}

cleanup_on_error() {
    echo -e "${RED}❌ Installation failed. Cleaning up...${NC}"
    if [[ -d "$TERMINAL_CODER_DIR" ]]; then
        read -p "Remove partially installed files? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$TERMINAL_CODER_DIR"
            echo -e "${YELLOW}Cleanup completed${NC}"
        fi
    fi
}

print_success_message() {
    echo -e "${GREEN}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🎉 INSTALLATION SUCCESSFUL! 🎉                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo -e "${GREEN}✅ Terminal Coder has been successfully installed!${NC}"
    echo ""
    echo -e "${CYAN}🚀 Quick Start:${NC}"
    echo -e "   ${YELLOW}terminal-coder --tui${NC}     # Launch modern TUI interface"
    echo -e "   ${YELLOW}terminal-coder${NC}           # Launch interactive mode"
    echo -e "   ${YELLOW}tcoder${NC}                   # Short alias"
    echo ""
    echo -e "${CYAN}📚 Next Steps:${NC}"
    echo -e "   1. Restart your shell or run: ${YELLOW}source ~/.bashrc${NC}"
    echo -e "   2. Configure API keys: ${YELLOW}terminal-coder config${NC}"
    echo -e "   3. Create your first project: ${YELLOW}terminal-coder project create${NC}"
    echo -e "   4. Get help: ${YELLOW}terminal-coder --help${NC}"
    echo ""
    echo -e "${CYAN}📍 Installation Details:${NC}"
    echo -e "   • Installed in: ${YELLOW}$TERMINAL_CODER_DIR${NC}"
    echo -e "   • Virtual environment: ${YELLOW}$VENV_DIR${NC}"
    echo -e "   • Log file: ${YELLOW}$LOG_FILE${NC}"
    echo ""
    echo -e "${BLUE}Happy coding with Terminal Coder! 🐧💻${NC}"
}

# Main installation function
main() {
    print_banner

    # Trap cleanup on error
    trap cleanup_on_error ERR

    log "Starting Terminal Coder Linux installation"

    # Pre-installation checks
    check_root

    # Detect system
    local distro=$(detect_distribution)
    echo -e "${CYAN}🔍 Detected Linux distribution: $distro${NC}"
    log "Detected distribution: $distro"

    # Check Python
    local python_cmd=$(check_python_version)
    echo -e "${GREEN}✅ Found compatible Python: $python_cmd${NC}"
    log "Using Python command: $python_cmd"

    # Installation steps
    install_system_dependencies "$distro"
    create_virtual_environment "$python_cmd"
    install_terminal_coder
    setup_shell_integration
    create_desktop_entry
    create_bin_symlink
    run_initial_setup

    print_success_message

    log "Installation completed successfully"
}

# Command line options
case "${1:-}" in
    --help|-h)
        echo "Terminal Coder Linux Installer"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --uninstall    Uninstall Terminal Coder"
        echo "  --update       Update existing installation"
        exit 0
        ;;
    --uninstall)
        echo -e "${YELLOW}🗑️  Uninstalling Terminal Coder...${NC}"
        if [[ -d "$TERMINAL_CODER_DIR" ]]; then
            rm -rf "$TERMINAL_CODER_DIR"
            # Remove desktop entry
            rm -f "$HOME/.local/share/applications/terminal-coder.desktop"
            # Remove symlinks
            rm -f "$HOME/.local/bin/terminal-coder"
            rm -f "$HOME/.local/bin/tcoder"
            echo -e "${GREEN}✅ Terminal Coder uninstalled successfully${NC}"
        else
            echo -e "${YELLOW}⚠️  Terminal Coder is not installed${NC}"
        fi
        exit 0
        ;;
    --update)
        echo -e "${BLUE}🔄 Updating Terminal Coder...${NC}"
        if [[ -d "$VENV_DIR" ]]; then
            source "$VENV_DIR/bin/activate"
            pip install --upgrade terminal-coder-linux[all]
            echo -e "${GREEN}✅ Terminal Coder updated successfully${NC}"
        else
            echo -e "${RED}❌ Terminal Coder is not installed. Run without --update to install.${NC}"
            exit 1
        fi
        exit 0
        ;;
    "")
        # Run main installation
        main
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        echo "Use --help for usage information"
        exit 1
        ;;
esac