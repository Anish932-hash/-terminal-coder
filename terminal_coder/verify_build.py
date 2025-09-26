#!/usr/bin/env python3
"""
Terminal Coder - Build Verification Script
Comprehensive verification of build system and all generated artifacts
"""

import asyncio
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import zipfile
import tarfile
from datetime import datetime
import logging

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Confirm
    from rich.text import Text
    from rich.align import Align
    import rich.traceback
    RICH_AVAILABLE = True
    rich.traceback.install()
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs): print(*args)

    def Panel(text, **kwargs): return f"=== {text} ==="
    def Table(**kwargs):
        class MockTable:
            def add_column(self, *args, **kwargs): pass
            def add_row(self, *args, **kwargs): pass
        return MockTable()


class BuildVerifier:
    """Comprehensive build verification system"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else Console()
        self.project_root = Path(__file__).parent
        self.verification_results = {}
        self.errors = []
        self.warnings = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('build_verification.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def verify_all_builds(self) -> bool:
        """Verify all build components and artifacts"""
        self._display_verification_banner()

        verification_steps = [
            ("Source Code", self._verify_source_code),
            ("Dependencies", self._verify_dependencies),
            ("Build Scripts", self._verify_build_scripts),
            ("Configuration Files", self._verify_configuration_files),
            ("Build Artifacts", self._verify_build_artifacts),
            ("Installation Scripts", self._verify_installation_scripts),
            ("Cross-Platform Compatibility", self._verify_cross_platform),
            ("Security Validation", self._verify_security),
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            main_task = progress.add_task("Verifying Build System", total=len(verification_steps))

            for i, (step_name, verify_func) in enumerate(verification_steps):
                progress.update(main_task, description=f"Verifying {step_name}...", completed=i)

                try:
                    result = await verify_func()
                    self.verification_results[step_name] = result

                    if result['status'] == 'success':
                        self.console.print(f"[green]✅ {step_name} verification passed[/green]")
                    elif result['status'] == 'warning':
                        self.console.print(f"[yellow]⚠️ {step_name} verification passed with warnings[/yellow]")
                        self.warnings.extend(result.get('warnings', []))
                    else:
                        self.console.print(f"[red]❌ {step_name} verification failed[/red]")
                        self.errors.extend(result.get('errors', []))

                except Exception as e:
                    self.console.print(f"[red]❌ {step_name} verification error: {e}[/red]")
                    self.errors.append(f"{step_name}: {str(e)}")
                    self.verification_results[step_name] = {
                        'status': 'error',
                        'errors': [str(e)]
                    }

            progress.update(main_task, description="Verification completed", completed=len(verification_steps))

        # Generate verification report
        await self._generate_verification_report()

        # Display results
        self._display_verification_results()

        return len(self.errors) == 0

    def _display_verification_banner(self):
        """Display verification banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔍 TERMINAL CODER BUILD VERIFIER                         ║
║                   Comprehensive Build System Validation                      ║
║                        System Quality Assurance                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """

        self.console.print(Panel(
            Align.center(banner),
            style="bold magenta",
            border_style="bright_magenta"
        ))

    async def _verify_source_code(self) -> Dict[str, Any]:
        """Verify source code integrity and structure"""
        result = {'status': 'success', 'details': [], 'warnings': [], 'errors': []}

        # Check required source files
        required_files = [
            'linux/main.py',
            'linux/gui.py',
            'linux/advanced_gui_extensions.py',
            'linux/ai_integration.py',
            'linux/advanced_ai_integration.py',
            'linux/project_manager.py',
            'linux/system_manager.py',
            'windows/main.py',
            'windows/gui.py',
            'windows/advanced_gui_extensions.py',
            'windows/ai_integration.py',
            'windows/advanced_ai_integration.py',
            'windows/project_manager.py',
            'windows/system_manager.py',
            '__init__.py'
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            result['status'] = 'error'
            result['errors'].extend([f"Missing source file: {f}" for f in missing_files])

        # Check Python syntax
        python_files = list(self.project_root.rglob("*.py"))
        syntax_errors = []

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
                result['details'].append(f"Syntax OK: {py_file.relative_to(self.project_root)}")
            except SyntaxError as e:
                syntax_errors.append(f"{py_file.relative_to(self.project_root)}: {e}")

        if syntax_errors:
            result['status'] = 'error'
            result['errors'].extend(syntax_errors)

        # Check file encoding
        encoding_issues = []
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError as e:
                encoding_issues.append(f"{py_file.relative_to(self.project_root)}: {e}")

        if encoding_issues:
            result['status'] = 'warning'
            result['warnings'].extend(encoding_issues)

        return result

    async def _verify_dependencies(self) -> Dict[str, Any]:
        """Verify dependency specifications and availability"""
        result = {'status': 'success', 'details': [], 'warnings': [], 'errors': []}

        # Check requirements.txt
        req_file = self.project_root / 'requirements.txt'
        if not req_file.exists():
            result['status'] = 'error'
            result['errors'].append("requirements.txt not found")
            return result

        # Parse requirements
        requirements = []
        try:
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        requirements.append(line)

            result['details'].append(f"Found {len(requirements)} dependencies in requirements.txt")
        except Exception as e:
            result['status'] = 'error'
            result['errors'].append(f"Failed to parse requirements.txt: {e}")
            return result

        # Check setup.py dependencies
        setup_file = self.project_root / 'setup.py'
        if setup_file.exists():
            result['details'].append("setup.py found - package configuration available")
        else:
            result['warnings'].append("setup.py not found - package installation may not work")

        # Test import of critical dependencies
        critical_deps = ['rich', 'asyncio', 'threading', 'json', 'pathlib']
        import_failures = []

        for dep in critical_deps:
            try:
                __import__(dep)
                result['details'].append(f"Critical dependency available: {dep}")
            except ImportError:
                import_failures.append(dep)

        if import_failures:
            result['status'] = 'warning'
            result['warnings'].extend([f"Critical dependency missing: {dep}" for dep in import_failures])

        return result

    async def _verify_build_scripts(self) -> Dict[str, Any]:
        """Verify build script functionality and syntax"""
        result = {'status': 'success', 'details': [], 'warnings': [], 'errors': []}

        build_scripts = [
            'build.py',
            'compile.py',
            'install.py',
            'setup.py'
        ]

        for script in build_scripts:
            script_path = self.project_root / script

            if not script_path.exists():
                result['status'] = 'error' if script in ['build.py', 'install.py'] else 'warning'
                error_msg = f"Build script missing: {script}"
                if result['status'] == 'error':
                    result['errors'].append(error_msg)
                else:
                    result['warnings'].append(error_msg)
                continue

            # Check syntax
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(script_path), 'exec')
                result['details'].append(f"Build script syntax OK: {script}")
            except SyntaxError as e:
                result['status'] = 'error'
                result['errors'].append(f"Syntax error in {script}: {e}")

            # Check for required functions/classes
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if script == 'build.py' and 'class ComprehensiveBuildSystem' not in content:
                    result['warnings'].append(f"{script}: Main build class not found")
                elif script == 'compile.py' and 'class PlatformCompiler' not in content:
                    result['warnings'].append(f"{script}: Main compiler class not found")
                elif script == 'install.py' and 'class InstallationManager' not in content:
                    result['warnings'].append(f"{script}: Main installer class not found")

            except Exception as e:
                result['warnings'].append(f"Could not analyze {script}: {e}")

        return result

    async def _verify_configuration_files(self) -> Dict[str, Any]:
        """Verify configuration file format and content"""
        result = {'status': 'success', 'details': [], 'warnings': [], 'errors': []}

        config_files = {
            'requirements.txt': 'text',
            'setup.py': 'python',
            'pyproject.toml': 'toml',
            '.gitignore': 'text'
        }

        for config_file, file_type in config_files.items():
            config_path = self.project_root / config_file

            if not config_path.exists():
                if config_file == 'requirements.txt':
                    result['status'] = 'error'
                    result['errors'].append(f"Critical config file missing: {config_file}")
                else:
                    result['warnings'].append(f"Optional config file missing: {config_file}")
                continue

            try:
                if file_type == 'python':
                    with open(config_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(config_path), 'exec')
                elif file_type == 'toml':
                    try:
                        import toml
                        with open(config_path, 'r', encoding='utf-8') as f:
                            toml.load(f)
                    except ImportError:
                        result['warnings'].append(f"Cannot validate {config_file}: toml library not available")
                elif file_type == 'json':
                    with open(config_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                else:  # text
                    with open(config_path, 'r', encoding='utf-8') as f:
                        f.read()

                result['details'].append(f"Configuration file valid: {config_file}")

            except Exception as e:
                result['status'] = 'error'
                result['errors'].append(f"Invalid configuration file {config_file}: {e}")

        return result

    async def _verify_build_artifacts(self) -> Dict[str, Any]:
        """Verify build artifacts if they exist"""
        result = {'status': 'success', 'details': [], 'warnings': [], 'errors': []}

        dist_dir = self.project_root / 'dist'

        if not dist_dir.exists():
            result['warnings'].append("No dist directory found - build artifacts not available for verification")
            return result

        # Check for different types of artifacts
        artifact_types = {
            'executables': ['.exe', ''],  # Windows exe and Unix executables
            'packages': ['.whl', '.tar.gz'],
            'installers': ['.msi', '.deb', '.rpm', '.pkg'],
            'archives': ['.zip', '.tar.gz']
        }

        found_artifacts = {}

        for artifact_type, extensions in artifact_types.items():
            type_dir = dist_dir / artifact_type
            found_artifacts[artifact_type] = []

            if type_dir.exists():
                for item in type_dir.rglob('*'):
                    if item.is_file():
                        if any(item.name.endswith(ext) for ext in extensions if ext) or \
                           (not any(extensions) and not item.suffix):
                            found_artifacts[artifact_type].append(item)

        # Verify artifact integrity
        for artifact_type, artifacts in found_artifacts.items():
            if artifacts:
                result['details'].append(f"Found {len(artifacts)} {artifact_type} artifacts")

                for artifact in artifacts:
                    try:
                        # Basic integrity checks
                        size = artifact.stat().st_size
                        if size == 0:
                            result['warnings'].append(f"Empty artifact: {artifact.name}")
                        else:
                            result['details'].append(f"Artifact OK ({size} bytes): {artifact.name}")

                        # Check specific artifact types
                        if artifact.suffix == '.zip':
                            try:
                                with zipfile.ZipFile(artifact, 'r') as zf:
                                    zf.testzip()
                                result['details'].append(f"ZIP integrity OK: {artifact.name}")
                            except Exception as e:
                                result['warnings'].append(f"ZIP integrity issue in {artifact.name}: {e}")

                    except Exception as e:
                        result['warnings'].append(f"Cannot verify artifact {artifact.name}: {e}")
            else:
                result['warnings'].append(f"No {artifact_type} artifacts found")

        return result

    async def _verify_installation_scripts(self) -> Dict[str, Any]:
        """Verify installation script functionality"""
        result = {'status': 'success', 'details': [], 'warnings': [], 'errors': []}

        install_script = self.project_root / 'install.py'

        if not install_script.exists():
            result['status'] = 'error'
            result['errors'].append("install.py not found")
            return result

        # Test installation script help functionality
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(install_script), '--help',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                result['details'].append("Installation script help function works")
            else:
                result['warnings'].append(f"Installation script help issue: {stderr.decode()}")

        except Exception as e:
            result['warnings'].append(f"Could not test installation script: {e}")

        # Check for required installation components
        try:
            with open(install_script, 'r', encoding='utf-8') as f:
                content = f.read()

            required_components = [
                'class InstallationManager',
                'class DependencyManager',
                'class PlatformDetector',
                'def run_installation',
                'async def main'
            ]

            for component in required_components:
                if component in content:
                    result['details'].append(f"Installation component found: {component}")
                else:
                    result['warnings'].append(f"Installation component missing: {component}")

        except Exception as e:
            result['errors'].append(f"Could not analyze installation script: {e}")
            result['status'] = 'error'

        return result

    async def _verify_cross_platform(self) -> Dict[str, Any]:
        """Verify cross-platform compatibility"""
        result = {'status': 'success', 'details': [], 'warnings': [], 'errors': []}

        current_platform = platform.system().lower()
        result['details'].append(f"Current platform: {current_platform}")

        # Check platform-specific directories
        platform_dirs = ['linux', 'windows']

        for platform_dir in platform_dirs:
            platform_path = self.project_root / platform_dir

            if platform_path.exists() and platform_path.is_dir():
                platform_files = list(platform_path.glob('*.py'))
                result['details'].append(f"Platform {platform_dir}: {len(platform_files)} Python files")

                # Check for main.py in each platform
                main_file = platform_path / 'main.py'
                if main_file.exists():
                    result['details'].append(f"Platform {platform_dir}: main.py found")
                else:
                    result['errors'].append(f"Platform {platform_dir}: main.py missing")
                    result['status'] = 'error'
            else:
                result['errors'].append(f"Platform directory missing: {platform_dir}")
                result['status'] = 'error'

        # Check for platform-specific imports
        try:
            platform_specific_imports = {
                'windows': ['win32api', 'win32con', 'winreg'],
                'linux': ['pwd', 'grp', 'fcntl'],
                'darwin': ['Foundation', 'AppKit']
            }

            for platform_name, imports in platform_specific_imports.items():
                available_imports = []
                for imp in imports:
                    try:
                        __import__(imp)
                        available_imports.append(imp)
                    except ImportError:
                        pass

                if available_imports:
                    result['details'].append(f"Platform {platform_name} imports available: {len(available_imports)}/{len(imports)}")
                else:
                    result['warnings'].append(f"No platform-specific imports available for {platform_name}")

        except Exception as e:
            result['warnings'].append(f"Could not check platform-specific imports: {e}")

        return result

    async def _verify_security(self) -> Dict[str, Any]:
        """Verify security aspects of the build"""
        result = {'status': 'success', 'details': [], 'warnings': [], 'errors': []}

        # Check for potential security issues in source code
        security_patterns = [
            ('hardcoded_password', r'password\s*=\s*["\'][^"\']+["\']'),
            ('hardcoded_key', r'api_key\s*=\s*["\'][^"\']+["\']'),
            ('shell_injection', r'os\.system\s*\([^)]*input'),
            ('eval_usage', r'eval\s*\('),
            ('exec_usage', r'exec\s*\(')
        ]

        import re

        python_files = list(self.project_root.rglob("*.py"))
        security_issues = []

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for pattern_name, pattern in security_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        security_issues.append(f"{py_file.relative_to(self.project_root)}: {pattern_name} - {len(matches)} occurrences")

            except Exception as e:
                result['warnings'].append(f"Could not scan {py_file.relative_to(self.project_root)}: {e}")

        if security_issues:
            result['status'] = 'warning'
            result['warnings'].extend(security_issues)
        else:
            result['details'].append("No obvious security issues found in source code")

        # Check file permissions
        sensitive_files = ['install.py', 'compile.py', 'build.py']

        for sensitive_file in sensitive_files:
            file_path = self.project_root / sensitive_file
            if file_path.exists():
                try:
                    stat_info = file_path.stat()
                    # Check if file is world-writable (potential security risk)
                    if stat_info.st_mode & 0o002:
                        result['warnings'].append(f"File is world-writable: {sensitive_file}")
                    else:
                        result['details'].append(f"File permissions OK: {sensitive_file}")
                except Exception as e:
                    result['warnings'].append(f"Could not check permissions for {sensitive_file}: {e}")

        return result

    async def _generate_verification_report(self):
        """Generate detailed verification report"""
        report = {
            'verification_info': {
                'timestamp': datetime.now().isoformat(),
                'platform': platform.platform(),
                'python_version': sys.version,
                'verifier_version': '1.0.0'
            },
            'summary': {
                'total_steps': len(self.verification_results),
                'passed': sum(1 for r in self.verification_results.values() if r['status'] == 'success'),
                'warnings': sum(1 for r in self.verification_results.values() if r['status'] == 'warning'),
                'failed': sum(1 for r in self.verification_results.values() if r['status'] in ['error', 'failed']),
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings)
            },
            'results': self.verification_results,
            'errors': self.errors,
            'warnings': self.warnings
        }

        report_file = self.project_root / 'build_verification_report.json'

        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)

            self.console.print(f"[green]✅ Verification report saved: {report_file}[/green]")

        except Exception as e:
            self.console.print(f"[red]❌ Could not save verification report: {e}[/red]")

    def _display_verification_results(self):
        """Display verification results summary"""
        # Summary table
        summary_table = Table(title="🔍 Verification Summary", style="blue")
        summary_table.add_column("Step", style="bold")
        summary_table.add_column("Status", style="white")
        summary_table.add_column("Details", style="dim")

        for step_name, result in self.verification_results.items():
            status = result['status']
            detail_count = len(result.get('details', []))
            warning_count = len(result.get('warnings', []))
            error_count = len(result.get('errors', []))

            if status == 'success':
                status_text = "✅ PASS"
                status_style = "green"
            elif status == 'warning':
                status_text = "⚠️ WARN"
                status_style = "yellow"
            else:
                status_text = "❌ FAIL"
                status_style = "red"

            details_text = f"{detail_count} details"
            if warning_count > 0:
                details_text += f", {warning_count} warnings"
            if error_count > 0:
                details_text += f", {error_count} errors"

            summary_table.add_row(step_name, Text(status_text, style=status_style), details_text)

        self.console.print(summary_table)

        # Overall result
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                result_panel = """
🎉 BUILD VERIFICATION COMPLETED SUCCESSFULLY!

✅ All verification steps passed
✅ No errors detected
✅ No warnings detected

🚀 Your Terminal Coder build system is ready for production use!
            """
                self.console.print(Panel(
                    result_panel,
                    title="🎊 Verification Success",
                    style="bold green",
                    border_style="bright_green"
                ))
            else:
                result_panel = f"""
✅ BUILD VERIFICATION PASSED WITH WARNINGS

✅ No critical errors detected
⚠️ {len(self.warnings)} warnings found

📋 Warnings should be reviewed but do not prevent build usage.
🚀 Your Terminal Coder build system is ready for use!
            """
                self.console.print(Panel(
                    result_panel,
                    title="⚠️ Verification Warning",
                    style="bold yellow",
                    border_style="bright_yellow"
                ))
        else:
            result_panel = f"""
❌ BUILD VERIFICATION FAILED

❌ {len(self.errors)} errors detected
⚠️ {len(self.warnings)} warnings found

🔧 Errors must be fixed before the build system can be used safely.
📋 Check the verification report for detailed information.
            """
            self.console.print(Panel(
                result_panel,
                title="❌ Verification Failed",
                style="bold red",
                border_style="bright_red"
            ))

        # Display errors and warnings
        if self.errors:
            self.console.print("\n[red]❌ ERRORS:[/red]")
            for error in self.errors:
                self.console.print(f"  • {error}")

        if self.warnings:
            self.console.print("\n[yellow]⚠️ WARNINGS:[/yellow]")
            for warning in self.warnings:
                self.console.print(f"  • {warning}")


async def main():
    """Main verification function"""
    verifier = BuildVerifier()
    success = await verifier.verify_all_builds()

    if success:
        print("\n🎉 Build verification completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Build verification failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nVerification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)