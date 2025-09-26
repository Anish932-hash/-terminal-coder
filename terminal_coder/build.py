#!/usr/bin/env python3
"""
Terminal Coder - Comprehensive Build Script
One-click build system for all platforms and formats
"""

import asyncio
import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import hashlib
from datetime import datetime

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Confirm
    import rich.traceback
    RICH_AVAILABLE = True
    rich.traceback.install()
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs): print(*args)


@dataclass
class BuildConfig:
    """Comprehensive build configuration"""
    # Output options
    output_dir: Path = field(default_factory=lambda: Path('./dist'))

    # Platform options
    target_platforms: List[str] = field(default_factory=lambda: ['current'])

    # Build types
    build_installer: bool = True
    build_executable: bool = True
    build_package: bool = True
    build_docker: bool = False

    # Feature options
    include_gui: bool = True
    include_ai: bool = True
    include_ml: bool = True
    include_dev_tools: bool = False

    # Optimization options
    optimize: bool = True
    compress: bool = True
    onefile: bool = True
    debug: bool = False

    # Signing and security
    sign_executables: bool = False
    certificate_path: Optional[Path] = None

    # Version info
    version: str = "2.0.0"
    build_number: Optional[int] = None


class ComprehensiveBuildSystem:
    """All-in-one build system for Terminal Coder"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else Console()
        self.temp_dir = Path(tempfile.mkdtemp(prefix='terminal_coder_build_'))
        self.build_artifacts = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.temp_dir / 'build.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def build_all(self, config: BuildConfig) -> bool:
        """Build everything according to configuration"""
        try:
            self._display_build_banner()

            # Pre-build checks
            if not await self._pre_build_checks(config):
                return False

            # Display build plan
            self._display_build_plan(config)

            if not Confirm.ask("Proceed with build?"):
                self.console.print("[yellow]Build cancelled by user[/yellow]")
                return False

            success = True

            # Build progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
                transient=False,
            ) as progress:

                main_task = progress.add_task("Building Terminal Coder", total=100)

                # Step 1: Prepare build environment (10%)
                progress.update(main_task, description="Preparing build environment...", completed=10)
                if not await self._prepare_build_environment(config):
                    return False

                # Step 2: Build executables (40%)
                if config.build_executable:
                    progress.update(main_task, description="Building executables...", completed=30)
                    if not await self._build_executables(config):
                        success = False

                # Step 3: Build packages (20%)
                if config.build_package:
                    progress.update(main_task, description="Building packages...", completed=50)
                    if not await self._build_packages(config):
                        success = False

                # Step 4: Build installers (20%)
                if config.build_installer:
                    progress.update(main_task, description="Building installers...", completed=70)
                    if not await self._build_installers(config):
                        success = False

                # Step 5: Build Docker images (Optional)
                if config.build_docker:
                    progress.update(main_task, description="Building Docker images...", completed=85)
                    if not await self._build_docker_images(config):
                        success = False

                # Step 6: Post-build tasks (10%)
                progress.update(main_task, description="Finalizing build...", completed=95)
                await self._post_build_tasks(config)

                progress.update(main_task, description="Build completed!", completed=100)

            # Generate build report
            await self._generate_build_report(config, success)

            if success:
                self._display_success_message(config)
            else:
                self.console.print("[red]❌ Build completed with errors[/red]")

            return success

        except Exception as e:
            self.console.print(f"[red]❌ Build failed: {e}[/red]")
            self.logger.error(f"Build error: {e}", exc_info=True)
            return False

        finally:
            await self._cleanup()

    def _display_build_banner(self):
        """Display build system banner"""
        system_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        python_info = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🏗️ TERMINAL CODER BUILD SYSTEM v2.0                      ║
║                   Comprehensive Cross-Platform Builder                       ║
║                                                                              ║
║  System: {system_info:<60} ║
║  Python: {python_info:<60} ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """

        self.console.print(Panel(
            banner,
            style="bold cyan",
            border_style="bright_blue"
        ))

    def _display_build_plan(self, config: BuildConfig):
        """Display comprehensive build plan"""
        plan_table = Table(title="🗂️ Build Plan", style="green")
        plan_table.add_column("Component", style="bold")
        plan_table.add_column("Status", style="white")
        plan_table.add_column("Platforms", style="cyan")
        plan_table.add_column("Details", style="dim")

        # Build components
        components = [
            ("Executables", "✅ Enabled" if config.build_executable else "❌ Disabled",
             ", ".join(config.target_platforms), "PyInstaller-based executables"),
            ("Packages", "✅ Enabled" if config.build_package else "❌ Disabled",
             "Python Package", "Wheel and source distributions"),
            ("Installers", "✅ Enabled" if config.build_installer else "❌ Disabled",
             ", ".join(config.target_platforms), "Platform-specific installers"),
            ("Docker Images", "✅ Enabled" if config.build_docker else "❌ Disabled",
             "Linux x64", "Containerized applications"),
        ]

        # Feature components
        features = [
            ("GUI Components", "✅ Included" if config.include_gui else "❌ Excluded"),
            ("AI Integration", "✅ Included" if config.include_ai else "❌ Excluded"),
            ("ML Libraries", "✅ Included" if config.include_ml else "❌ Excluded"),
            ("Dev Tools", "✅ Included" if config.include_dev_tools else "❌ Excluded"),
        ]

        for component, status, platforms, details in components:
            plan_table.add_row(component, status, platforms, details)

        self.console.print(plan_table)

        # Feature table
        feature_table = Table(title="📦 Features", style="blue")
        feature_table.add_column("Feature", style="bold")
        feature_table.add_column("Status", style="white")

        for feature, status in features:
            feature_table.add_row(feature, status)

        self.console.print(feature_table)

    async def _pre_build_checks(self, config: BuildConfig) -> bool:
        """Perform pre-build validation"""
        self.console.print("[bold blue]🔍 Running pre-build checks...[/bold blue]")

        checks_passed = True
        issues = []

        # Check Python version
        if sys.version_info < (3, 8):
            checks_passed = False
            issues.append("Python 3.8+ required")

        # Check required dependencies
        required_packages = ['setuptools', 'wheel', 'build']
        if config.build_executable:
            required_packages.extend(['PyInstaller', 'pyinstaller-hooks-contrib'])

        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                issues.append(f"Missing package: {package}")
                checks_passed = False

        # Check disk space (estimate 2GB needed)
        try:
            free_space = shutil.disk_usage(config.output_dir.parent)[2]
            if free_space < 2 * 1024 * 1024 * 1024:  # 2GB
                issues.append(f"Low disk space: {free_space / (1024**3):.1f}GB available")
        except Exception:
            issues.append("Could not check disk space")

        # Platform-specific checks
        system = platform.system().lower()
        if system == 'windows' and config.build_installer:
            if not shutil.which('makensis'):
                issues.append("NSIS installer not found (for Windows installer)")

        if system == 'linux' and config.build_package:
            if not shutil.which('dpkg-deb') and not shutil.which('rpmbuild'):
                issues.append("Package builders not found (dpkg-deb or rpmbuild)")

        # Display results
        if checks_passed:
            self.console.print("[green]✅ All pre-build checks passed[/green]")
        else:
            self.console.print("[red]❌ Pre-build checks failed:[/red]")
            for issue in issues:
                self.console.print(f"  • {issue}")

        if issues:
            self.console.print("[yellow]⚠️ Issues detected:[/yellow]")
            for issue in issues:
                self.console.print(f"  • {issue}")

        return checks_passed

    async def _prepare_build_environment(self, config: BuildConfig) -> bool:
        """Prepare build environment"""
        try:
            # Create output directories
            config.output_dir.mkdir(parents=True, exist_ok=True)
            (config.output_dir / 'executables').mkdir(exist_ok=True)
            (config.output_dir / 'packages').mkdir(exist_ok=True)
            (config.output_dir / 'installers').mkdir(exist_ok=True)

            # Create build info
            build_info = {
                'version': config.version,
                'build_number': config.build_number or int(datetime.now().timestamp()),
                'build_date': datetime.now().isoformat(),
                'build_system': platform.system(),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'config': {
                    'target_platforms': config.target_platforms,
                    'include_gui': config.include_gui,
                    'include_ai': config.include_ai,
                    'include_ml': config.include_ml,
                    'optimize': config.optimize,
                    'compress': config.compress,
                }
            }

            build_info_file = config.output_dir / 'build_info.json'
            with open(build_info_file, 'w') as f:
                json.dump(build_info, f, indent=2)

            self.console.print(f"[green]✅ Build environment prepared: {config.output_dir}[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Failed to prepare build environment: {e}[/red]")
            return False

    async def _build_executables(self, config: BuildConfig) -> bool:
        """Build executables using PyInstaller"""
        self.console.print("[bold blue]🔨 Building executables...[/bold blue]")

        try:
            # Run the compile.py script
            compile_script = Path(__file__).parent / 'compile.py'
            if not compile_script.exists():
                self.console.print("[red]❌ compile.py not found[/red]")
                return False

            # Prepare compile arguments
            compile_args = [
                sys.executable, str(compile_script),
                '--output-dir', str(config.output_dir / 'executables'),
                '--platforms'] + config.target_platforms

            if not config.include_gui:
                compile_args.append('--no-gui')
            if not config.include_ai:
                compile_args.append('--no-ai')
            if config.debug:
                compile_args.append('--debug')
            if not config.onefile:
                compile_args.append('--no-onefile')
            if config.compress:
                compile_args.append('--upx')

            # Run compilation
            process = await asyncio.create_subprocess_exec(
                *compile_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.console.print("[green]✅ Executables built successfully[/green]")
                self.logger.info(f"Executable build output:\n{stdout.decode()}")
                return True
            else:
                self.console.print(f"[red]❌ Executable build failed[/red]")
                self.console.print(f"[red]Error: {stderr.decode()}[/red]")
                return False

        except Exception as e:
            self.console.print(f"[red]❌ Executable build error: {e}[/red]")
            return False

    async def _build_packages(self, config: BuildConfig) -> bool:
        """Build Python packages (wheel and source)"""
        self.console.print("[bold blue]📦 Building Python packages...[/bold blue]")

        try:
            package_dir = config.output_dir / 'packages'

            # Build wheel
            wheel_cmd = [sys.executable, '-m', 'build', '--wheel', '--outdir', str(package_dir)]
            process = await asyncio.create_subprocess_exec(*wheel_cmd)
            await process.communicate()

            if process.returncode != 0:
                self.console.print("[red]❌ Wheel build failed[/red]")
                return False

            # Build source distribution
            sdist_cmd = [sys.executable, '-m', 'build', '--sdist', '--outdir', str(package_dir)]
            process = await asyncio.create_subprocess_exec(*sdist_cmd)
            await process.communicate()

            if process.returncode != 0:
                self.console.print("[red]❌ Source distribution build failed[/red]")
                return False

            self.console.print("[green]✅ Python packages built successfully[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Package build error: {e}[/red]")
            return False

    async def _build_installers(self, config: BuildConfig) -> bool:
        """Build platform-specific installers"""
        self.console.print("[bold blue]💿 Building installers...[/bold blue]")

        try:
            installer_dir = config.output_dir / 'installers'

            # Run the install.py script in installer mode
            install_script = Path(__file__).parent / 'install.py'
            if not install_script.exists():
                self.console.print("[red]❌ install.py not found[/red]")
                return False

            # For now, just copy the installer script
            shutil.copy2(install_script, installer_dir / 'terminal_coder_installer.py')

            # Create a simple batch/shell wrapper
            system = platform.system().lower()
            if system == 'windows':
                wrapper_content = f'''@echo off
python "{installer_dir / 'terminal_coder_installer.py'}" %*
'''
                wrapper_file = installer_dir / 'install.bat'
                with open(wrapper_file, 'w') as f:
                    f.write(wrapper_content)

            else:
                wrapper_content = f'''#!/bin/bash
python3 "{installer_dir / 'terminal_coder_installer.py'}" "$@"
'''
                wrapper_file = installer_dir / 'install.sh'
                with open(wrapper_file, 'w') as f:
                    f.write(wrapper_content)
                os.chmod(wrapper_file, 0o755)

            self.console.print("[green]✅ Installers prepared successfully[/green]")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Installer build error: {e}[/red]")
            return False

    async def _build_docker_images(self, config: BuildConfig) -> bool:
        """Build Docker images"""
        self.console.print("[bold blue]🐳 Building Docker images...[/bold blue]")

        if not shutil.which('docker'):
            self.console.print("[yellow]⚠️ Docker not found, skipping Docker build[/yellow]")
            return True

        try:
            # Create Dockerfile
            dockerfile_content = f'''FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the application
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash terminal_coder
USER terminal_coder

# Expose port (if needed)
EXPOSE 8000

# Default command
CMD ["terminal-coder"]
'''

            dockerfile_path = self.temp_dir / 'Dockerfile'
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)

            # Build Docker image
            image_tag = f"terminal-coder:{config.version}"
            docker_cmd = [
                'docker', 'build',
                '-t', image_tag,
                '-f', str(dockerfile_path),
                str(Path(__file__).parent)
            ]

            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                self.console.print(f"[green]✅ Docker image built: {image_tag}[/green]")

                # Save image to file
                docker_save_cmd = ['docker', 'save', '-o', str(config.output_dir / f'terminal-coder-{config.version}.tar'), image_tag]
                process = await asyncio.create_subprocess_exec(*docker_save_cmd)
                await process.communicate()

                if process.returncode == 0:
                    self.console.print("[green]✅ Docker image saved to file[/green]")

                return True
            else:
                self.console.print(f"[red]❌ Docker build failed: {stderr.decode()}[/red]")
                return False

        except Exception as e:
            self.console.print(f"[red]❌ Docker build error: {e}[/red]")
            return False

    async def _post_build_tasks(self, config: BuildConfig):
        """Post-build tasks like signing and verification"""
        self.console.print("[bold blue]🔧 Running post-build tasks...[/bold blue]")

        try:
            # Sign executables if requested
            if config.sign_executables and config.certificate_path:
                await self._sign_executables(config)

            # Create checksums
            await self._create_checksums(config)

            # Verify builds
            await self._verify_builds(config)

            self.console.print("[green]✅ Post-build tasks completed[/green]")

        except Exception as e:
            self.console.print(f"[yellow]⚠️ Post-build tasks failed: {e}[/yellow]")

    async def _sign_executables(self, config: BuildConfig):
        """Sign executables with certificate"""
        # This would implement code signing for different platforms
        self.console.print("[yellow]⚠️ Code signing not implemented yet[/yellow]")

    async def _create_checksums(self, config: BuildConfig):
        """Create SHA256 checksums for all build artifacts"""
        checksum_file = config.output_dir / 'SHA256SUMS'

        with open(checksum_file, 'w') as f:
            for artifact_dir in [config.output_dir / 'executables',
                               config.output_dir / 'packages',
                               config.output_dir / 'installers']:
                if artifact_dir.exists():
                    for file_path in artifact_dir.rglob('*'):
                        if file_path.is_file():
                            checksum = await self._calculate_checksum(file_path)
                            relative_path = file_path.relative_to(config.output_dir)
                            f.write(f"{checksum}  {relative_path}\n")

        self.console.print(f"[green]✅ Checksums created: {checksum_file}[/green]")

    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def _verify_builds(self, config: BuildConfig):
        """Verify build artifacts"""
        # Basic verification - check if files exist and are not empty
        verification_passed = True

        for artifact_dir in [config.output_dir / 'executables',
                           config.output_dir / 'packages',
                           config.output_dir / 'installers']:
            if artifact_dir.exists():
                files = list(artifact_dir.rglob('*'))
                if files:
                    self.console.print(f"[green]✅ {artifact_dir.name}: {len([f for f in files if f.is_file()])} files[/green]")
                else:
                    self.console.print(f"[red]❌ {artifact_dir.name}: No files found[/red]")
                    verification_passed = False

        return verification_passed

    async def _generate_build_report(self, config: BuildConfig, success: bool):
        """Generate comprehensive build report"""
        report = {
            'build_info': {
                'version': config.version,
                'build_date': datetime.now().isoformat(),
                'build_system': platform.platform(),
                'python_version': sys.version,
                'success': success
            },
            'config': {
                'target_platforms': config.target_platforms,
                'build_types': {
                    'executable': config.build_executable,
                    'package': config.build_package,
                    'installer': config.build_installer,
                    'docker': config.build_docker
                },
                'features': {
                    'gui': config.include_gui,
                    'ai': config.include_ai,
                    'ml': config.include_ml,
                    'dev_tools': config.include_dev_tools
                }
            },
            'artifacts': [],
            'checksums': {}
        }

        # Collect artifacts
        for artifact_dir in [config.output_dir / 'executables',
                           config.output_dir / 'packages',
                           config.output_dir / 'installers']:
            if artifact_dir.exists():
                for file_path in artifact_dir.rglob('*'):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        checksum = await self._calculate_checksum(file_path)
                        relative_path = str(file_path.relative_to(config.output_dir))

                        report['artifacts'].append({
                            'path': relative_path,
                            'size': size,
                            'checksum': checksum
                        })

        # Save report
        report_file = config.output_dir / 'build_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.console.print(f"[green]✅ Build report saved: {report_file}[/green]")

    def _display_success_message(self, config: BuildConfig):
        """Display build success message"""
        artifacts_count = len(list(config.output_dir.rglob('*')))
        total_size = sum(f.stat().st_size for f in config.output_dir.rglob('*') if f.is_file())

        success_message = f"""
🎉 Build Completed Successfully!

📁 Output Directory: {config.output_dir}
📊 Artifacts Generated: {artifacts_count} files
💾 Total Size: {total_size / (1024*1024):.1f} MB
🏗️ Platforms: {', '.join(config.target_platforms)}
⏱️ Build Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🚀 Your Terminal Coder distribution is ready!

Distribution Contents:
• 📱 Executables: {config.output_dir}/executables/
• 📦 Packages: {config.output_dir}/packages/
• 💿 Installers: {config.output_dir}/installers/
• 📋 Build Report: {config.output_dir}/build_report.json
• 🔐 Checksums: {config.output_dir}/SHA256SUMS
        """

        self.console.print(Panel(
            success_message,
            title="🎊 Build Complete",
            style="bold green",
            border_style="bright_green"
        ))

    async def _cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.console.print(f"[dim]🧹 Cleaned up temporary files[/dim]")
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Could not clean up temp files: {e}[/yellow]")


async def main():
    """Main build function"""
    parser = argparse.ArgumentParser(description="Terminal Coder Comprehensive Build System")

    # Output options
    parser.add_argument("--output-dir", type=Path, default=Path("./dist"),
                       help="Output directory for build artifacts")

    # Platform options
    parser.add_argument("--platforms", nargs="+", default=["current"],
                       choices=["current", "windows", "linux", "macos", "all"],
                       help="Target platforms")

    # Build types
    parser.add_argument("--no-executable", action="store_true",
                       help="Skip executable build")
    parser.add_argument("--no-package", action="store_true",
                       help="Skip package build")
    parser.add_argument("--no-installer", action="store_true",
                       help="Skip installer build")
    parser.add_argument("--docker", action="store_true",
                       help="Build Docker images")

    # Feature options
    parser.add_argument("--no-gui", action="store_true",
                       help="Exclude GUI components")
    parser.add_argument("--no-ai", action="store_true",
                       help="Exclude AI components")
    parser.add_argument("--no-ml", action="store_true",
                       help="Exclude ML libraries")
    parser.add_argument("--include-dev", action="store_true",
                       help="Include development tools")

    # Build options
    parser.add_argument("--no-optimize", action="store_true",
                       help="Disable optimizations")
    parser.add_argument("--no-compress", action="store_true",
                       help="Disable compression")
    parser.add_argument("--no-onefile", action="store_true",
                       help="Create directory distributions")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    # Version options
    parser.add_argument("--version", default="2.0.0",
                       help="Version number")
    parser.add_argument("--build-number", type=int,
                       help="Build number")

    args = parser.parse_args()

    # Handle platform "all"
    if "all" in args.platforms:
        args.platforms = ["windows", "linux", "macos"]

    # Create build configuration
    config = BuildConfig(
        output_dir=args.output_dir,
        target_platforms=args.platforms,
        build_executable=not args.no_executable,
        build_package=not args.no_package,
        build_installer=not args.no_installer,
        build_docker=args.docker,
        include_gui=not args.no_gui,
        include_ai=not args.no_ai,
        include_ml=not args.no_ml,
        include_dev_tools=args.include_dev,
        optimize=not args.no_optimize,
        compress=not args.no_compress,
        onefile=not args.no_onefile,
        debug=args.debug,
        version=args.version,
        build_number=args.build_number
    )

    # Run build system
    build_system = ComprehensiveBuildSystem()
    success = await build_system.build_all(config)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBuild cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Build system error: {e}")
        sys.exit(1)