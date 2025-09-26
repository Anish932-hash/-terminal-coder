"""
Advanced Error Handling and Recovery System
Comprehensive error management with intelligent recovery suggestions
"""

import traceback
import sys
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import re


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    API = "api"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    FILE_SYSTEM = "file_system"
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    DEPENDENCY = "dependency"
    PERMISSION = "permission"
    RESOURCE = "resource"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    traceback: str
    context: Dict[str, Any]
    timestamp: datetime
    recovery_suggestions: List[str]
    auto_recovery_possible: bool
    user_action_required: bool


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    name: str
    description: str
    action: Callable
    auto_execute: bool
    success_message: str
    failure_message: str


class ErrorPattern:
    """Error pattern matching for intelligent error handling"""

    def __init__(self, pattern: str, category: ErrorCategory,
                 severity: ErrorSeverity, recovery_suggestions: List[str]):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.category = category
        self.severity = severity
        self.recovery_suggestions = recovery_suggestions

    def matches(self, error_message: str) -> bool:
        """Check if error message matches this pattern"""
        return bool(self.pattern.search(error_message))


class AdvancedErrorHandler:
    """Advanced error handling system with intelligent recovery"""

    def __init__(self, console=None, logger=None):
        self.console = console
        self.logger = logger or self._setup_logger()
        self.error_history: List[ErrorInfo] = []
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.error_patterns = self._initialize_error_patterns()
        self.auto_recovery_enabled = True
        self.max_history = 100

        # Statistics
        self.error_stats = {
            "total_errors": 0,
            "auto_recovered": 0,
            "user_recovered": 0,
            "unrecovered": 0,
            "by_category": {},
            "by_severity": {}
        }

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for error handler"""
        logger = logging.getLogger("error_handler")
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            handler = logging.FileHandler("error_handler.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Initialize common error patterns"""
        patterns = [
            # API Errors
            ErrorPattern(
                r"401|unauthorized|authentication failed",
                ErrorCategory.AUTHENTICATION,
                ErrorSeverity.HIGH,
                [
                    "Check API key configuration",
                    "Verify API key is valid and not expired",
                    "Ensure correct authentication method is used",
                    "Check API endpoint permissions"
                ]
            ),
            ErrorPattern(
                r"429|rate limit|too many requests",
                ErrorCategory.API,
                ErrorSeverity.MEDIUM,
                [
                    "Wait before retrying the request",
                    "Implement exponential backoff",
                    "Check rate limit settings",
                    "Consider upgrading API plan"
                ]
            ),
            ErrorPattern(
                r"404|not found|endpoint.*not.*found",
                ErrorCategory.API,
                ErrorSeverity.MEDIUM,
                [
                    "Verify API endpoint URL",
                    "Check API documentation for correct endpoints",
                    "Ensure the resource exists",
                    "Check API version compatibility"
                ]
            ),
            ErrorPattern(
                r"500|internal server error|server error",
                ErrorCategory.API,
                ErrorSeverity.HIGH,
                [
                    "Retry the request after a delay",
                    "Check API status page for outages",
                    "Contact API provider if issue persists",
                    "Implement fallback mechanism"
                ]
            ),

            # Network Errors
            ErrorPattern(
                r"connection.*refused|network.*unreachable|timeout",
                ErrorCategory.NETWORK,
                ErrorSeverity.MEDIUM,
                [
                    "Check internet connection",
                    "Verify firewall settings",
                    "Try different network endpoint",
                    "Increase timeout settings"
                ]
            ),
            ErrorPattern(
                r"ssl|certificate|tls",
                ErrorCategory.NETWORK,
                ErrorSeverity.HIGH,
                [
                    "Check SSL certificate validity",
                    "Update CA certificates",
                    "Verify system clock is correct",
                    "Try disabling SSL verification (not recommended for production)"
                ]
            ),

            # File System Errors
            ErrorPattern(
                r"permission denied|access denied|not permitted",
                ErrorCategory.PERMISSION,
                ErrorSeverity.HIGH,
                [
                    "Check file/directory permissions",
                    "Run with appropriate user privileges",
                    "Verify file ownership",
                    "Check SELinux/AppArmor policies if applicable"
                ]
            ),
            ErrorPattern(
                r"no such file|file not found|directory not found",
                ErrorCategory.FILE_SYSTEM,
                ErrorSeverity.MEDIUM,
                [
                    "Verify file/directory path exists",
                    "Check for typos in path",
                    "Ensure file hasn't been moved or deleted",
                    "Check current working directory"
                ]
            ),
            ErrorPattern(
                r"disk.*full|no space left|storage.*full",
                ErrorCategory.RESOURCE,
                ErrorSeverity.CRITICAL,
                [
                    "Free up disk space",
                    "Move files to different location",
                    "Clean temporary files",
                    "Check disk quota settings"
                ]
            ),

            # Python-specific Errors
            ErrorPattern(
                r"module.*not.*found|no module named",
                ErrorCategory.DEPENDENCY,
                ErrorSeverity.MEDIUM,
                [
                    "Install missing module with pip",
                    "Check virtual environment activation",
                    "Verify PYTHONPATH settings",
                    "Check module name spelling"
                ]
            ),
            ErrorPattern(
                r"syntax error|invalid syntax",
                ErrorCategory.SYNTAX,
                ErrorSeverity.HIGH,
                [
                    "Check code syntax carefully",
                    "Verify proper indentation",
                    "Check for missing brackets/quotes",
                    "Use a code linter for validation"
                ]
            ),

            # Configuration Errors
            ErrorPattern(
                r"config|configuration|setting.*invalid",
                ErrorCategory.CONFIGURATION,
                ErrorSeverity.MEDIUM,
                [
                    "Check configuration file syntax",
                    "Verify all required settings are present",
                    "Validate configuration values",
                    "Reset to default configuration if needed"
                ]
            ),
        ]

        return patterns

    def register_recovery_action(self, name: str, recovery_action: RecoveryAction):
        """Register a custom recovery action"""
        self.recovery_actions[name] = recovery_action

    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Main error handling method"""
        try:
            # Extract error information
            error_info = self._analyze_error(error, context or {})

            # Add to history
            self._add_to_history(error_info)

            # Update statistics
            self._update_stats(error_info)

            # Log the error
            self._log_error(error_info)

            # Display error to user
            if self.console:
                self._display_error(error_info)

            # Attempt recovery if enabled
            if self.auto_recovery_enabled and error_info.auto_recovery_possible:
                recovery_success = await self._attempt_auto_recovery(error_info)
                if recovery_success:
                    error_info.context["auto_recovered"] = True
                    self.error_stats["auto_recovered"] += 1

            return error_info

        except Exception as handler_error:
            # Error in error handler - log and re-raise
            self.logger.critical(f"Error in error handler: {handler_error}")
            raise

    def _analyze_error(self, error: Exception, context: Dict[str, Any]) -> ErrorInfo:
        """Analyze error and extract comprehensive information"""
        error_message = str(error)
        error_type = type(error).__name__
        traceback_str = traceback.format_exc()

        # Classify error using patterns
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        recovery_suggestions = ["Check error message for details"]

        for pattern in self.error_patterns:
            if pattern.matches(error_message):
                category = pattern.category
                severity = pattern.severity
                recovery_suggestions = pattern.recovery_suggestions
                break

        # Determine if auto recovery is possible
        auto_recovery_possible = self._can_auto_recover(error, category)

        # Determine if user action is required
        user_action_required = severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]

        return ErrorInfo(
            error_type=error_type,
            message=error_message,
            category=category,
            severity=severity,
            traceback=traceback_str,
            context=context,
            timestamp=datetime.now(),
            recovery_suggestions=recovery_suggestions,
            auto_recovery_possible=auto_recovery_possible,
            user_action_required=user_action_required
        )

    def _can_auto_recover(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if error can be automatically recovered"""
        auto_recoverable_categories = {
            ErrorCategory.NETWORK,  # Can retry
            ErrorCategory.API,      # Can retry with backoff
            ErrorCategory.RESOURCE  # Can clean up resources
        }

        # Don't auto-recover from critical errors
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return False

        return category in auto_recoverable_categories

    async def _attempt_auto_recovery(self, error_info: ErrorInfo) -> bool:
        """Attempt automatic error recovery"""
        try:
            if error_info.category == ErrorCategory.NETWORK:
                return await self._recover_network_error(error_info)
            elif error_info.category == ErrorCategory.API:
                return await self._recover_api_error(error_info)
            elif error_info.category == ErrorCategory.RESOURCE:
                return await self._recover_resource_error(error_info)

            return False

        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            return False

    async def _recover_network_error(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from network errors"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                await asyncio.sleep(retry_delay)
                # Test connectivity (implementation depends on context)
                # This is a placeholder - actual implementation would
                # retry the failed operation
                self.logger.info(f"Network recovery attempt {attempt + 1}/{max_retries}")
                return True  # Placeholder success

            except Exception as retry_error:
                retry_delay *= 2  # Exponential backoff
                if attempt == max_retries - 1:
                    self.logger.error(f"Network recovery failed after {max_retries} attempts")
                    return False

        return False

    async def _recover_api_error(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from API errors"""
        if "rate limit" in error_info.message.lower():
            # Wait for rate limit to reset
            await asyncio.sleep(60)  # Wait 1 minute
            return True
        elif "401" in error_info.message or "unauthorized" in error_info.message.lower():
            # Cannot auto-recover from auth errors
            return False

        return False

    async def _recover_resource_error(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from resource errors"""
        if "disk" in error_info.message.lower() or "space" in error_info.message.lower():
            # Attempt to free up space (placeholder)
            # Actual implementation would clean temp files, etc.
            self.logger.info("Attempting to free disk space...")
            return False  # Cannot automatically free space safely

        return False

    def _add_to_history(self, error_info: ErrorInfo):
        """Add error to history with size limit"""
        self.error_history.append(error_info)

        # Maintain history size limit
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

    def _update_stats(self, error_info: ErrorInfo):
        """Update error statistics"""
        self.error_stats["total_errors"] += 1

        # Update category stats
        category = error_info.category.value
        if category not in self.error_stats["by_category"]:
            self.error_stats["by_category"][category] = 0
        self.error_stats["by_category"][category] += 1

        # Update severity stats
        severity = error_info.severity.value
        if severity not in self.error_stats["by_severity"]:
            self.error_stats["by_severity"][severity] = 0
        self.error_stats["by_severity"][severity] += 1

    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = f"[{error_info.category.value}] {error_info.error_type}: {error_info.message}"

        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _display_error(self, error_info: ErrorInfo):
        """Display error information to user via console"""
        if not self.console:
            return

        # Color code by severity
        severity_colors = {
            ErrorSeverity.LOW: "blue",
            ErrorSeverity.MEDIUM: "yellow",
            ErrorSeverity.HIGH: "red",
            ErrorSeverity.CRITICAL: "bright_red"
        }

        color = severity_colors.get(error_info.severity, "white")

        # Create error display
        from rich.panel import Panel
        from rich.text import Text
        from rich import print as rich_print

        error_text = Text()
        error_text.append(f"🚨 {error_info.error_type}: ", style=f"bold {color}")
        error_text.append(f"{error_info.message}\n", style=color)
        error_text.append(f"Category: {error_info.category.value.title()}\n", style="dim")
        error_text.append(f"Severity: {error_info.severity.value.title()}\n", style="dim")

        if error_info.recovery_suggestions:
            error_text.append("\n💡 Recovery Suggestions:\n", style="bold cyan")
            for i, suggestion in enumerate(error_info.recovery_suggestions, 1):
                error_text.append(f"  {i}. {suggestion}\n", style="cyan")

        panel = Panel(
            error_text,
            title="Error Details",
            border_style=color,
            expand=False
        )

        self.console.print(panel)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        recent_errors = self.error_history[-10:]  # Last 10 errors

        return {
            "statistics": self.error_stats,
            "recent_errors": [
                {
                    "type": error.error_type,
                    "message": error.message,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "timestamp": error.timestamp.isoformat(),
                    "auto_recovered": error.context.get("auto_recovered", False)
                }
                for error in recent_errors
            ],
            "most_common_categories": self._get_most_common_categories(),
            "recovery_rate": self._calculate_recovery_rate()
        }

    def _get_most_common_categories(self) -> List[Dict[str, Any]]:
        """Get most common error categories"""
        categories = list(self.error_stats["by_category"].items())
        categories.sort(key=lambda x: x[1], reverse=True)

        return [
            {"category": cat, "count": count}
            for cat, count in categories[:5]
        ]

    def _calculate_recovery_rate(self) -> float:
        """Calculate automatic recovery success rate"""
        total = self.error_stats["total_errors"]
        recovered = self.error_stats["auto_recovered"]

        if total == 0:
            return 0.0

        return (recovered / total) * 100

    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.error_stats = {
            "total_errors": 0,
            "auto_recovered": 0,
            "user_recovered": 0,
            "unrecovered": 0,
            "by_category": {},
            "by_severity": {}
        }

    def export_error_report(self, filename: str = None) -> str:
        """Export detailed error report"""
        if not filename:
            filename = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_error_summary(),
            "detailed_errors": [
                {
                    "error_type": error.error_type,
                    "message": error.message,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "timestamp": error.timestamp.isoformat(),
                    "traceback": error.traceback,
                    "context": error.context,
                    "recovery_suggestions": error.recovery_suggestions,
                    "auto_recovery_possible": error.auto_recovery_possible,
                    "user_action_required": error.user_action_required
                }
                for error in self.error_history
            ]
        }

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            return filename

        except Exception as export_error:
            self.logger.error(f"Failed to export error report: {export_error}")
            return None


# Context managers for error handling
class ErrorContext:
    """Context manager for handling errors in specific operations"""

    def __init__(self, error_handler: AdvancedErrorHandler,
                 operation_name: str, context: Dict[str, Any] = None):
        self.error_handler = error_handler
        self.operation_name = operation_name
        self.context = context or {}
        self.context["operation"] = operation_name

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.error_handler.handle_error(exc_val, self.context)
            # Return False to propagate the exception
            return False


# Decorator for automatic error handling
def handle_errors(error_handler: AdvancedErrorHandler,
                 operation_name: str = None,
                 context: Dict[str, Any] = None):
    """Decorator for automatic error handling"""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            ctx = context or {}
            ctx["function"] = func.__name__

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                await error_handler.handle_error(e, ctx)
                raise  # Re-raise after handling

        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            ctx = context or {}
            ctx["function"] = func.__name__

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we need to handle this differently
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(error_handler.handle_error(e, ctx))
                except:
                    # Fallback if no event loop
                    error_handler._log_error(error_handler._analyze_error(e, ctx))
                raise  # Re-raise after handling

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator