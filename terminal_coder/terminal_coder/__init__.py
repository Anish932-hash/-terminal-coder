"""
Terminal Coder Linux v2.0 - Ultra-Advanced Development Terminal
Enterprise-grade AI-powered development terminal with deep Linux integration

Features:
🐧 Native Linux integration (systemd, D-Bus, inotify)
🤖 Ultra-advanced AI management with multiple providers
🔒 Enterprise-grade security scanning and compliance
📊 Advanced code analysis with AI insights
🚀 Professional project templates with Linux optimization
📈 Real-time system monitoring and optimization
🛡️ Built-in vulnerability scanning and auto-remediation
⚡ Linux kernel-level optimizations

Compatible with Python 3.10+ and modern Python features
"""

__version__ = "2.0.0-ultra"
__author__ = "Terminal Coder Linux Team"
__email__ = "linux@terminalcoder.dev"
__license__ = "MIT"
__description__ = "Ultra-Advanced AI-Powered Linux Development Terminal with Enterprise Features"
__url__ = "https://github.com/terminalcoder/terminal_coder"

# Feature flags
ULTRA_FEATURES_ENABLED = True
LINUX_NATIVE_OPTIMIZATIONS = True
ENTERPRISE_SECURITY = True
ADVANCED_AI_MANAGEMENT = True

# Version information tuple for programmatic access
VERSION_INFO = (2, 0, 0, 'ultra')
LINUX_VERSION_INFO = {
    'version': '2.0.0-ultra',
    'features': ['ultra-ai', 'enterprise-security', 'linux-native', 'advanced-monitoring'],
    'linux_optimized': True,
    'enterprise_ready': True
}

# Import ultra-advanced components with error handling
try:
    from .main import TerminalCoder, main
except ImportError:
    TerminalCoder = None
    main = None

# Ultra-advanced AI management
try:
    from .advanced_ai_manager import AdvancedAIManager
    from .modern_ai_integration import ModernAIIntegration, AIProviderType, AIRequest
    from .quantum_ai_integration import (
        QuantumAIManager, get_quantum_ai_manager, initialize_quantum_ai,
        QuantumTask, QuantumResult, quantum_optimize_code, quantum_analyze_code
    )
    from .neural_acceleration_engine import (
        get_neural_engine, initialize_neural_acceleration, accelerated_compute,
        NeuralComputeEngine, AccelerationType, PrecisionType
    )
    from .advanced_debugging_profiler import (
        get_advanced_profiler, initialize_advanced_debugging, profile_method,
        AdvancedProfiler, DebugLevel, ProfilerType, PerformanceIssue
    )
except ImportError:
    AdvancedAIManager = None
    ModernAIIntegration = None
    AIProviderType = None
    AIRequest = None
    # Quantum AI
    QuantumAIManager = None
    get_quantum_ai_manager = None
    initialize_quantum_ai = None
    QuantumTask = None
    QuantumResult = None
    quantum_optimize_code = None
    quantum_analyze_code = None
    # Neural Acceleration
    get_neural_engine = None
    initialize_neural_acceleration = None
    accelerated_compute = None
    NeuralComputeEngine = None
    AccelerationType = None
    PrecisionType = None
    # Advanced Debugging
    get_advanced_profiler = None
    initialize_advanced_debugging = None
    profile_method = None
    AdvancedProfiler = None
    DebugLevel = None
    ProfilerType = None
    PerformanceIssue = None

# Enterprise security features
try:
    from .enterprise_security_manager import EnterpriseSecurityManager, SecurityFramework
except ImportError:
    EnterpriseSecurityManager = None
    SecurityFramework = None

# Ultra Linux system management
try:
    from .ultra_linux_manager import UltraLinuxManager, SystemOptimizationLevel
except ImportError:
    UltraLinuxManager = None
    SystemOptimizationLevel = None

# Advanced code analysis
try:
    from .advanced_code_analyzer import AdvancedCodeAnalyzer
except ImportError:
    AdvancedCodeAnalyzer = None

# Enterprise project templates
try:
    from .enterprise_project_templates import EnterpriseProjectTemplates, ProjectType as EnterpriseProjectType
except ImportError:
    EnterpriseProjectTemplates = None
    EnterpriseProjectType = None

# Legacy components for compatibility
try:
    from .project_manager import AdvancedProjectManager, ProjectManager
except ImportError:
    AdvancedProjectManager = None
    ProjectManager = None

try:
    from .config_manager import AdvancedConfigManager, ConfigManager
except ImportError:
    AdvancedConfigManager = None
    ConfigManager = None

try:
    from .error_handler import AdvancedErrorHandler
except ImportError:
    AdvancedErrorHandler = None

# Legacy AI integration
try:
    from .ai_integration import AIManager, AIProvider, AIResponse
except ImportError:
    AIManager = None
    AIProvider = None
    AIResponse = None

# Legacy GUI
try:
    from .advanced_gui import AdvancedGUI
except ImportError:
    AdvancedGUI = None

__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "__url__",
    "VERSION_INFO",
    "LINUX_VERSION_INFO",
    "ULTRA_FEATURES_ENABLED",
    "LINUX_NATIVE_OPTIMIZATIONS",
    "ENTERPRISE_SECURITY",
    "ADVANCED_AI_MANAGEMENT",

    # Core application
    "TerminalCoder",
    "main",

    # Ultra-advanced AI features
    "AdvancedAIManager",
    "ModernAIIntegration",
    "AIProviderType",
    "AIRequest",

    # Quantum AI Integration
    "QuantumAIManager",
    "get_quantum_ai_manager",
    "initialize_quantum_ai",
    "QuantumTask",
    "QuantumResult",
    "quantum_optimize_code",
    "quantum_analyze_code",

    # Neural Acceleration
    "get_neural_engine",
    "initialize_neural_acceleration",
    "accelerated_compute",
    "NeuralComputeEngine",
    "AccelerationType",
    "PrecisionType",

    # Advanced Debugging
    "get_advanced_profiler",
    "initialize_advanced_debugging",
    "profile_method",
    "AdvancedProfiler",
    "DebugLevel",
    "ProfilerType",
    "PerformanceIssue",

    # Enterprise features
    "EnterpriseSecurityManager",
    "SecurityFramework",
    "UltraLinuxManager",
    "SystemOptimizationLevel",
    "AdvancedCodeAnalyzer",
    "EnterpriseProjectTemplates",
    "EnterpriseProjectType",

    # Legacy components (for compatibility)
    "AIManager",
    "AIProvider",
    "AIResponse",
    "AdvancedProjectManager",
    "ProjectManager",
    "AdvancedConfigManager",
    "ConfigManager",
    "AdvancedErrorHandler",
    "AdvancedGUI",
]