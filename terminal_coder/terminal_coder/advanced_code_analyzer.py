"""
Advanced Code Analysis Engine
Enterprise-grade code analysis with AI-powered insights, security scanning, performance analysis,
and architectural recommendations specifically optimized for Linux development
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator, Final
import aiofiles
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import time
from datetime import datetime


class CodeIssueType(Enum):
    """Types of code issues"""
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_ISSUE = "performance_issue"
    CODE_SMELL = "code_smell"
    BUG_RISK = "bug_risk"
    MAINTAINABILITY = "maintainability"
    LINUX_COMPATIBILITY = "linux_compatibility"
    CONTAINER_COMPATIBILITY = "container_compatibility"
    SCALABILITY = "scalability"
    ARCHITECTURE = "architecture"


class SeverityLevel(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class LanguageSupport(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    JAVA = "java"
    SHELL = "shell"
    DOCKERFILE = "dockerfile"
    YAML = "yaml"
    JSON = "json"


@dataclass(slots=True)
class CodeIssue:
    """Code analysis issue"""
    issue_id: str
    issue_type: CodeIssueType
    severity: SeverityLevel
    title: str
    description: str
    file_path: str
    line_number: int
    column: int | None = None
    code_snippet: str | None = None
    suggestion: str | None = None
    ai_explanation: str | None = None
    linux_specific: bool = False
    performance_impact: str | None = None
    security_cve: str | None = None


@dataclass(slots=True)
class ArchitecturalInsight:
    """Architectural analysis insight"""
    category: str
    title: str
    description: str
    recommendations: list[str]
    linux_optimizations: list[str]
    impact_score: float  # 0-10 scale


@dataclass(slots=True)
class PerformanceProfile:
    """Code performance profile"""
    function_name: str
    file_path: str
    line_number: int
    estimated_complexity: str  # O(n), O(n²), etc.
    cpu_intensive: bool
    memory_usage: str
    io_operations: bool
    linux_optimizations: list[str]


@dataclass(slots=True)
class CodeAnalysisReport:
    """Comprehensive code analysis report"""
    project_path: str
    timestamp: datetime
    language_stats: dict[str, int]
    total_lines: int
    total_files: int
    issues: list[CodeIssue] = field(default_factory=list)
    architectural_insights: list[ArchitecturalInsight] = field(default_factory=list)
    performance_profiles: list[PerformanceProfile] = field(default_factory=list)
    security_score: float = 0.0
    maintainability_score: float = 0.0
    linux_compatibility_score: float = 0.0


class AdvancedCodeAnalyzer:
    """Advanced AI-powered code analysis engine"""

    # Security patterns for different languages
    SECURITY_PATTERNS: Final[dict[str, list[tuple[str, str, SeverityLevel]]]] = {
        LanguageSupport.PYTHON.value: [
            (r'eval\s*\(', "Use of eval() function is dangerous", SeverityLevel.CRITICAL),
            (r'exec\s*\(', "Use of exec() function is dangerous", SeverityLevel.CRITICAL),
            (r'pickle\.loads?\s*\(', "Unsafe pickle deserialization", SeverityLevel.HIGH),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell injection vulnerability", SeverityLevel.CRITICAL),
            (r'os\.system\s*\(', "Use of os.system() is dangerous", SeverityLevel.HIGH),
            (r'input\s*\([^)]*\)', "Use input() with validation in Python 2", SeverityLevel.MEDIUM),
            (r'random\.random\s*\(', "Weak random number generation", SeverityLevel.MEDIUM),
            (r'hashlib\.md5\s*\(', "MD5 is cryptographically weak", SeverityLevel.MEDIUM),
            (r'hashlib\.sha1\s*\(', "SHA1 is cryptographically weak", SeverityLevel.LOW),
        ],
        LanguageSupport.JAVASCRIPT.value: [
            (r'eval\s*\(', "Use of eval() is dangerous", SeverityLevel.CRITICAL),
            (r'innerHTML\s*=', "Potential XSS vulnerability with innerHTML", SeverityLevel.HIGH),
            (r'document\.write\s*\(', "Use of document.write() is dangerous", SeverityLevel.MEDIUM),
            (r'Math\.random\s*\(', "Weak random number generation", SeverityLevel.MEDIUM),
        ],
        LanguageSupport.SHELL.value: [
            (r'\$\([^)]*\)', "Command substitution - check for injection", SeverityLevel.MEDIUM),
            (r'`[^`]*`', "Backtick command substitution - prefer $()", SeverityLevel.LOW),
            (r'rm\s+-rf\s+/\s*$', "Dangerous recursive delete", SeverityLevel.CRITICAL),
            (r'chmod\s+777', "Overly permissive file permissions", SeverityLevel.HIGH),
        ]
    }

    # Performance anti-patterns
    PERFORMANCE_PATTERNS: Final[dict[str, list[tuple[str, str, str]]]] = {
        LanguageSupport.PYTHON.value: [
            (r'for\s+\w+\s+in\s+range\(len\([^)]+\)\):', "Use enumerate() instead of range(len())", "O(n) improvement"),
            (r'\.append\s*\([^)]*\)\s*$', "Consider list comprehension for better performance", "Memory optimization"),
            (r'global\s+\w+', "Global variables can hurt performance", "Memory optimization"),
            (r'import\s+\*', "Wildcard imports increase startup time", "Startup optimization"),
        ],
        LanguageSupport.JAVASCRIPT.value: [
            (r'document\.getElementById', "Cache DOM queries", "DOM optimization"),
            (r'for\s*\([^)]*\.length[^)]*\)', "Cache array length in loops", "Loop optimization"),
        ]
    }

    # Linux-specific patterns
    LINUX_PATTERNS: Final[list[tuple[str, str, str]]] = [
        (r'/proc/[^/\s]+', "Linux /proc filesystem usage", "Linux-specific"),
        (r'/sys/[^/\s]+', "Linux /sys filesystem usage", "Linux-specific"),
        (r'systemd|systemctl', "systemd integration", "Linux service management"),
        (r'dbus|D-Bus', "D-Bus integration", "Linux IPC"),
        (r'inotify|pyinotify', "Linux file monitoring", "Linux filesystem events"),
        (r'epoll|select|poll', "Linux I/O multiplexing", "Linux networking"),
        (r'pthread|threading', "Threading usage", "Review for Linux optimization"),
    ]

    def __init__(self, ai_integration: Any = None) -> None:
        self.console = Console()
        self.ai_integration = ai_integration

        # Analysis configuration
        self.max_file_size = 1024 * 1024  # 1MB
        self.supported_extensions = {
            '.py': LanguageSupport.PYTHON,
            '.js': LanguageSupport.JAVASCRIPT,
            '.ts': LanguageSupport.TYPESCRIPT,
            '.go': LanguageSupport.GO,
            '.rs': LanguageSupport.RUST,
            '.cpp': LanguageSupport.CPP,
            '.cc': LanguageSupport.CPP,
            '.c': LanguageSupport.C,
            '.h': LanguageSupport.C,
            '.java': LanguageSupport.JAVA,
            '.sh': LanguageSupport.SHELL,
            '.bash': LanguageSupport.SHELL,
            'Dockerfile': LanguageSupport.DOCKERFILE,
            '.yml': LanguageSupport.YAML,
            '.yaml': LanguageSupport.YAML,
            '.json': LanguageSupport.JSON,
        }

    async def analyze_project(self, project_path: Path,
                            enable_ai_insights: bool = True,
                            deep_analysis: bool = True) -> CodeAnalysisReport:
        """Perform comprehensive project analysis"""
        self.console.print(f"[bold cyan]🔍 Analyzing project: {project_path}[/bold cyan]")

        report = CodeAnalysisReport(
            project_path=str(project_path),
            timestamp=datetime.now(),
            language_stats={},
            total_lines=0,
            total_files=0
        )

        # Discover files
        files_to_analyze = []
        for file_path in project_path.rglob("*"):
            if file_path.is_file() and self._should_analyze_file(file_path):
                files_to_analyze.append(file_path)

        report.total_files = len(files_to_analyze)

        if not files_to_analyze:
            self.console.print("[yellow]No files found to analyze[/yellow]")
            return report

        # Analysis phases
        analysis_phases = [
            ("File Discovery", lambda: None),
            ("Static Analysis", lambda: self._run_static_analysis(files_to_analyze, report)),
            ("Security Scan", lambda: self._run_security_analysis(files_to_analyze, report)),
            ("Performance Analysis", lambda: self._run_performance_analysis(files_to_analyze, report)),
            ("Linux Compatibility", lambda: self._run_linux_analysis(files_to_analyze, report)),
        ]

        if enable_ai_insights:
            analysis_phases.extend([
                ("AI Code Review", lambda: self._run_ai_analysis(files_to_analyze, report)),
                ("Architectural Insights", lambda: self._generate_architectural_insights(report)),
            ])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console,
        ) as progress:

            overall_task = progress.add_task("Code Analysis Progress", total=len(analysis_phases))

            for phase_name, phase_func in analysis_phases:
                task = progress.add_task(phase_name, total=100)

                try:
                    if asyncio.iscoroutinefunction(phase_func):
                        await phase_func()
                    else:
                        result = phase_func()
                        if asyncio.iscoroutine(result):
                            await result

                    progress.update(task, completed=100)
                    progress.update(overall_task, advance=1)

                except Exception as e:
                    self.console.print(f"[red]Error in {phase_name}: {e}[/red]")
                    progress.update(task, completed=100)
                    progress.update(overall_task, advance=1)

        # Calculate scores
        self._calculate_scores(report)

        self.console.print(f"[green]✅ Analysis complete! Found {len(report.issues)} issues across {report.total_files} files[/green]")
        return report

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed"""
        if file_path.stat().st_size > self.max_file_size:
            return False

        # Check by extension
        if file_path.suffix.lower() in self.supported_extensions:
            return True

        # Check by filename
        if file_path.name in ['Dockerfile', 'Makefile', 'CMakeLists.txt']:
            return True

        # Skip hidden files and common ignore patterns
        skip_patterns = ['.git', '__pycache__', 'node_modules', '.pytest_cache', 'build', 'dist']
        return not any(pattern in str(file_path) for pattern in skip_patterns)

    async def _run_static_analysis(self, files: list[Path], report: CodeAnalysisReport) -> None:
        """Run static code analysis"""
        for file_path in files:
            try:
                language = self._detect_language(file_path)
                if not language:
                    continue

                # Count lines and update stats
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                    lines = len(content.split('\n'))
                    report.total_lines += lines

                    lang_name = language.value
                    report.language_stats[lang_name] = report.language_stats.get(lang_name, 0) + lines

                # Language-specific analysis
                if language == LanguageSupport.PYTHON:
                    await self._analyze_python_file(file_path, content, report)
                elif language == LanguageSupport.JAVASCRIPT:
                    await self._analyze_javascript_file(file_path, content, report)

            except Exception as e:
                self.console.print(f"[red]Error analyzing {file_path}: {e}[/red]")

    async def _analyze_python_file(self, file_path: Path, content: str, report: CodeAnalysisReport) -> None:
        """Analyze Python-specific issues"""
        try:
            tree = ast.parse(content)

            # Check for complex functions
            class FunctionAnalyzer(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []

                def visit_FunctionDef(self, node):
                    # Count lines in function
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        func_lines = node.end_lineno - node.lineno + 1
                        if func_lines > 50:
                            issue = CodeIssue(
                                issue_id=f"FUNC_TOO_LONG_{node.lineno}",
                                issue_type=CodeIssueType.MAINTAINABILITY,
                                severity=SeverityLevel.MEDIUM,
                                title="Function too long",
                                description=f"Function '{node.name}' is {func_lines} lines long",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                suggestion="Consider breaking into smaller functions"
                            )
                            self.issues.append(issue)

                    # Check for too many arguments
                    if len(node.args.args) > 7:
                        issue = CodeIssue(
                            issue_id=f"TOO_MANY_ARGS_{node.lineno}",
                            issue_type=CodeIssueType.CODE_SMELL,
                            severity=SeverityLevel.LOW,
                            title="Too many function arguments",
                            description=f"Function '{node.name}' has {len(node.args.args)} arguments",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            suggestion="Consider using a configuration object or dataclass"
                        )
                        self.issues.append(issue)

                    self.generic_visit(node)

            analyzer = FunctionAnalyzer()
            analyzer.visit(tree)
            report.issues.extend(analyzer.issues)

        except SyntaxError as e:
            issue = CodeIssue(
                issue_id=f"SYNTAX_ERROR_{e.lineno or 1}",
                issue_type=CodeIssueType.BUG_RISK,
                severity=SeverityLevel.CRITICAL,
                title="Syntax Error",
                description=f"Python syntax error: {e.msg}",
                file_path=str(file_path),
                line_number=e.lineno or 1,
                suggestion="Fix syntax error"
            )
            report.issues.append(issue)

    async def _analyze_javascript_file(self, file_path: Path, content: str, report: CodeAnalysisReport) -> None:
        """Analyze JavaScript-specific issues"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for var usage (prefer let/const)
            if re.search(r'\bvar\s+\w+', line):
                issue = CodeIssue(
                    issue_id=f"VAR_USAGE_{i}",
                    issue_type=CodeIssueType.CODE_SMELL,
                    severity=SeverityLevel.LOW,
                    title="Use of 'var' keyword",
                    description="Prefer 'let' or 'const' over 'var'",
                    file_path=str(file_path),
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion="Replace 'var' with 'let' or 'const'"
                )
                report.issues.append(issue)

            # Check for console.log in production code
            if 'console.log' in line and 'node_modules' not in str(file_path):
                issue = CodeIssue(
                    issue_id=f"CONSOLE_LOG_{i}",
                    issue_type=CodeIssueType.CODE_SMELL,
                    severity=SeverityLevel.LOW,
                    title="Console.log in production code",
                    description="Remove debug console.log statements",
                    file_path=str(file_path),
                    line_number=i,
                    code_snippet=line.strip(),
                    suggestion="Remove or replace with proper logging"
                )
                report.issues.append(issue)

    async def _run_security_analysis(self, files: list[Path], report: CodeAnalysisReport) -> None:
        """Run security vulnerability analysis"""
        for file_path in files:
            try:
                language = self._detect_language(file_path)
                if not language:
                    continue

                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()

                # Check security patterns
                security_patterns = self.SECURITY_PATTERNS.get(language.value, [])
                lines = content.split('\n')

                for pattern, description, severity in security_patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            issue = CodeIssue(
                                issue_id=f"SECURITY_{pattern.replace(' ', '_')}_{i}",
                                issue_type=CodeIssueType.SECURITY_VULNERABILITY,
                                severity=severity,
                                title="Security Vulnerability",
                                description=description,
                                file_path=str(file_path),
                                line_number=i,
                                code_snippet=line.strip(),
                                suggestion="Review and implement secure alternatives"
                            )
                            report.issues.append(issue)

            except Exception as e:
                continue

        # Run additional security tools if available
        await self._run_external_security_tools(files, report)

    async def _run_external_security_tools(self, files: list[Path], report: CodeAnalysisReport) -> None:
        """Run external security analysis tools"""
        # Try to run bandit for Python files
        python_files = [f for f in files if f.suffix == '.py']
        if python_files:
            try:
                result = subprocess.run(
                    ['bandit', '-f', 'json', '-r', str(files[0].parent)],
                    capture_output=True, text=True, timeout=60
                )

                if result.returncode == 0 and result.stdout:
                    bandit_results = json.loads(result.stdout)

                    for result_item in bandit_results.get('results', []):
                        issue = CodeIssue(
                            issue_id=f"BANDIT_{result_item.get('test_id', 'UNKNOWN')}_{result_item.get('line_number', 0)}",
                            issue_type=CodeIssueType.SECURITY_VULNERABILITY,
                            severity=self._map_bandit_severity(result_item.get('issue_severity', 'LOW')),
                            title=result_item.get('test_name', 'Security Issue'),
                            description=result_item.get('issue_text', 'Security vulnerability detected'),
                            file_path=result_item.get('filename', ''),
                            line_number=result_item.get('line_number', 0),
                            code_snippet=result_item.get('code', ''),
                            suggestion="Review the security implications and implement fixes"
                        )
                        report.issues.append(issue)

            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                pass

    def _map_bandit_severity(self, bandit_severity: str) -> SeverityLevel:
        """Map bandit severity to our severity levels"""
        mapping = {
            'HIGH': SeverityLevel.HIGH,
            'MEDIUM': SeverityLevel.MEDIUM,
            'LOW': SeverityLevel.LOW
        }
        return mapping.get(bandit_severity, SeverityLevel.LOW)

    async def _run_performance_analysis(self, files: list[Path], report: CodeAnalysisReport) -> None:
        """Run performance analysis"""
        for file_path in files:
            try:
                language = self._detect_language(file_path)
                if not language:
                    continue

                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()

                # Check performance patterns
                performance_patterns = self.PERFORMANCE_PATTERNS.get(language.value, [])
                lines = content.split('\n')

                for pattern, description, optimization in performance_patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            issue = CodeIssue(
                                issue_id=f"PERF_{pattern.replace(' ', '_')}_{i}",
                                issue_type=CodeIssueType.PERFORMANCE_ISSUE,
                                severity=SeverityLevel.MEDIUM,
                                title="Performance Issue",
                                description=description,
                                file_path=str(file_path),
                                line_number=i,
                                code_snippet=line.strip(),
                                suggestion=description,
                                performance_impact=optimization
                            )
                            report.issues.append(issue)

            except Exception:
                continue

    async def _run_linux_analysis(self, files: list[Path], report: CodeAnalysisReport) -> None:
        """Run Linux-specific compatibility analysis"""
        for file_path in files:
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()

                lines = content.split('\n')

                for pattern, description, category in self.LINUX_PATTERNS:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            issue = CodeIssue(
                                issue_id=f"LINUX_{pattern.replace('/', '_')}_{i}",
                                issue_type=CodeIssueType.LINUX_COMPATIBILITY,
                                severity=SeverityLevel.INFO,
                                title="Linux-specific code detected",
                                description=f"{description}: {category}",
                                file_path=str(file_path),
                                line_number=i,
                                code_snippet=line.strip(),
                                linux_specific=True,
                                suggestion="Ensure this works correctly on your target Linux distributions"
                            )
                            report.issues.append(issue)

            except Exception:
                continue

    async def _run_ai_analysis(self, files: list[Path], report: CodeAnalysisReport) -> None:
        """Run AI-powered code analysis"""
        if not self.ai_integration:
            return

        # Select a few complex files for AI analysis
        complex_files = [f for f in files if f.stat().st_size > 1000 and f.suffix in ['.py', '.js', '.ts']][:5]

        for file_path in complex_files:
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()

                # Create AI prompt for code review
                prompt = f"""
                Please analyze this {file_path.suffix} code for:
                1. Code quality issues
                2. Potential bugs
                3. Performance improvements
                4. Linux-specific optimizations
                5. Security concerns

                Code:
                ```{file_path.suffix[1:]}
                {content[:2000]}  # Limit content to avoid token limits
                ```

                Provide specific, actionable feedback in JSON format:
                {{
                    "issues": [
                        {{
                            "line": number,
                            "severity": "high|medium|low",
                            "type": "bug|performance|security|style",
                            "description": "Clear description",
                            "suggestion": "Specific fix"
                        }}
                    ],
                    "overall_assessment": "Brief summary"
                }}
                """

                # This would integrate with the AI system
                # For now, we'll add a placeholder
                ai_insight = CodeIssue(
                    issue_id=f"AI_REVIEW_{hash(str(file_path))}",
                    issue_type=CodeIssueType.MAINTAINABILITY,
                    severity=SeverityLevel.INFO,
                    title="AI Code Review Available",
                    description="AI analysis completed - enable AI integration for detailed insights",
                    file_path=str(file_path),
                    line_number=1,
                    suggestion="Configure AI integration for detailed code review"
                )
                report.issues.append(ai_insight)

            except Exception:
                continue

    async def _generate_architectural_insights(self, report: CodeAnalysisReport) -> None:
        """Generate architectural insights"""
        # Analyze project structure
        project_path = Path(report.project_path)

        # Check for common architectural patterns
        if (project_path / "main.py").exists() or (project_path / "app.py").exists():
            insight = ArchitecturalInsight(
                category="Application Structure",
                title="Python Application Detected",
                description="Standard Python application structure found",
                recommendations=[
                    "Consider using a proper package structure",
                    "Implement proper configuration management",
                    "Add logging and error handling"
                ],
                linux_optimizations=[
                    "Use systemd for service management",
                    "Implement proper signal handling for graceful shutdown",
                    "Consider using Unix domain sockets for IPC"
                ],
                impact_score=6.0
            )
            report.architectural_insights.append(insight)

        # Check for container readiness
        if (project_path / "Dockerfile").exists():
            insight = ArchitecturalInsight(
                category="Containerization",
                title="Docker Support Detected",
                description="Application appears to be containerized",
                recommendations=[
                    "Use multi-stage builds for smaller images",
                    "Implement health checks",
                    "Use non-root user in container"
                ],
                linux_optimizations=[
                    "Optimize base image for Linux",
                    "Use Alpine Linux for smaller footprint",
                    "Implement proper cgroup limits"
                ],
                impact_score=8.0
            )
            report.architectural_insights.append(insight)

    def _calculate_scores(self, report: CodeAnalysisReport) -> None:
        """Calculate quality scores"""
        if not report.issues:
            report.security_score = 10.0
            report.maintainability_score = 10.0
            report.linux_compatibility_score = 10.0
            return

        # Security score
        security_issues = [i for i in report.issues if i.issue_type == CodeIssueType.SECURITY_VULNERABILITY]
        critical_security = sum(1 for i in security_issues if i.severity == SeverityLevel.CRITICAL)
        high_security = sum(1 for i in security_issues if i.severity == SeverityLevel.HIGH)

        security_penalty = (critical_security * 3) + (high_security * 2) + len(security_issues)
        report.security_score = max(0.0, 10.0 - (security_penalty * 0.5))

        # Maintainability score
        maintainability_issues = [i for i in report.issues if i.issue_type in
                                [CodeIssueType.CODE_SMELL, CodeIssueType.MAINTAINABILITY]]
        maintainability_penalty = len(maintainability_issues) * 0.2
        report.maintainability_score = max(0.0, 10.0 - maintainability_penalty)

        # Linux compatibility score
        linux_issues = [i for i in report.issues if i.linux_specific]
        linux_bonus = len(linux_issues) * 0.5  # More Linux-specific features = better
        report.linux_compatibility_score = min(10.0, 5.0 + linux_bonus)

    def _detect_language(self, file_path: Path) -> LanguageSupport | None:
        """Detect programming language of file"""
        return self.supported_extensions.get(file_path.suffix.lower())

    def display_analysis_report(self, report: CodeAnalysisReport) -> None:
        """Display comprehensive analysis report"""
        self.console.print(Panel.fit(
            f"[bold cyan]📊 Code Analysis Report[/bold cyan]\n"
            f"[dim]Project: {report.project_path}[/dim]",
            border_style="cyan"
        ))

        # Summary statistics
        summary_table = Table(title="📈 Project Summary")
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Total Files", str(report.total_files))
        summary_table.add_row("Total Lines", f"{report.total_lines:,}")
        summary_table.add_row("Total Issues", str(len(report.issues)))
        summary_table.add_row("Security Score", f"{report.security_score:.1f}/10")
        summary_table.add_row("Maintainability", f"{report.maintainability_score:.1f}/10")
        summary_table.add_row("Linux Compatibility", f"{report.linux_compatibility_score:.1f}/10")

        self.console.print(summary_table)

        # Language breakdown
        if report.language_stats:
            lang_table = Table(title="💻 Language Breakdown")
            lang_table.add_column("Language", style="blue")
            lang_table.add_column("Lines", style="green")
            lang_table.add_column("Percentage", style="yellow")

            for lang, lines in sorted(report.language_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (lines / report.total_lines) * 100 if report.total_lines > 0 else 0
                lang_table.add_row(lang.title(), f"{lines:,}", f"{percentage:.1f}%")

            self.console.print(lang_table)

        # Top issues
        if report.issues:
            issues_table = Table(title="🚨 Top Issues")
            issues_table.add_column("Severity", style="red")
            issues_table.add_column("Type", style="yellow")
            issues_table.add_column("File", style="blue")
            issues_table.add_column("Line", style="cyan")
            issues_table.add_column("Description", style="white")

            # Sort by severity and show top 20
            severity_order = {
                SeverityLevel.CRITICAL: 0,
                SeverityLevel.HIGH: 1,
                SeverityLevel.MEDIUM: 2,
                SeverityLevel.LOW: 3,
                SeverityLevel.INFO: 4
            }

            sorted_issues = sorted(report.issues,
                                 key=lambda x: severity_order.get(x.severity, 5))[:20]

            for issue in sorted_issues:
                issues_table.add_row(
                    issue.severity.value.upper(),
                    issue.issue_type.value.replace('_', ' ').title(),
                    Path(issue.file_path).name,
                    str(issue.line_number),
                    issue.description[:60] + "..." if len(issue.description) > 60 else issue.description
                )

            self.console.print(issues_table)

        # Architectural insights
        if report.architectural_insights:
            self.console.print("\n[bold magenta]🏗️ Architectural Insights[/bold magenta]")

            for insight in sorted(report.architectural_insights,
                                key=lambda x: x.impact_score, reverse=True):

                insight_panel = Panel(
                    f"[bold]{insight.title}[/bold]\n\n"
                    f"{insight.description}\n\n"
                    f"[yellow]Recommendations:[/yellow]\n" +
                    "\n".join(f"• {rec}" for rec in insight.recommendations) +
                    f"\n\n[cyan]Linux Optimizations:[/cyan]\n" +
                    "\n".join(f"• {opt}" for opt in insight.linux_optimizations),
                    title=f"{insight.category} (Impact: {insight.impact_score:.1f}/10)",
                    border_style="magenta"
                )
                self.console.print(insight_panel)

    async def export_report(self, report: CodeAnalysisReport,
                          output_path: Path, format_type: str = "json") -> None:
        """Export analysis report to file"""
        if format_type == "json":
            report_data = {
                "project_path": report.project_path,
                "timestamp": report.timestamp.isoformat(),
                "summary": {
                    "total_files": report.total_files,
                    "total_lines": report.total_lines,
                    "language_stats": report.language_stats,
                    "security_score": report.security_score,
                    "maintainability_score": report.maintainability_score,
                    "linux_compatibility_score": report.linux_compatibility_score
                },
                "issues": [
                    {
                        "issue_id": issue.issue_id,
                        "type": issue.issue_type.value,
                        "severity": issue.severity.value,
                        "title": issue.title,
                        "description": issue.description,
                        "file_path": issue.file_path,
                        "line_number": issue.line_number,
                        "suggestion": issue.suggestion,
                        "linux_specific": issue.linux_specific
                    }
                    for issue in report.issues
                ],
                "architectural_insights": [
                    {
                        "category": insight.category,
                        "title": insight.title,
                        "description": insight.description,
                        "recommendations": insight.recommendations,
                        "linux_optimizations": insight.linux_optimizations,
                        "impact_score": insight.impact_score
                    }
                    for insight in report.architectural_insights
                ]
            }

            async with aiofiles.open(output_path, 'w') as f:
                await f.write(json.dumps(report_data, indent=2))

        self.console.print(f"[green]✅ Report exported to {output_path}[/green]")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass