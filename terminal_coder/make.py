#!/usr/bin/env python3
"""
Terminal Coder - Universal Build Orchestrator
Single entry point for all build, compilation, installation, and verification tasks
"""

import asyncio
import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import platform

# Rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Confirm, Prompt
    from rich.text import Text
    from rich.align import Align
    import rich.traceback
    RICH_AVAILABLE = True
    rich.traceback.install()
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs): print(*args)


class UniversalBuilder:
    """Universal build orchestrator for Terminal Coder"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else Console()
        self.project_root = Path(__file__).parent
        self.system = platform.system().lower()

    def display_banner(self):
        """Display build orchestrator banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🎯 TERMINAL CODER BUILD ORCHESTRATOR                     ║
║                       Universal Build & Deploy System                        ║
║                          One Command Does It All                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """

        self.console.print(Panel(
            Align.center(banner),
            style="bold cyan",
            border_style="bright_cyan"
        ))

    def display_available_commands(self):
        """Display available build commands"""
        commands_table = Table(title="🔧 Available Commands", style="cyan")
        commands_table.add_column("Command", style="bold yellow")
        commands_table.add_column("Description", style="white")
        commands_table.add_column("Script", style="dim")

        commands = [
            ("install", "Install Terminal Coder with dependencies", "install.py"),
            ("compile", "Compile to standalone executables", "compile.py"),
            ("build", "Complete build with all formats", "build.py"),
            ("package", "Create Python packages (wheel/sdist)", "setup.py"),
            ("verify", "Verify build system integrity", "verify_build.py"),
            ("clean", "Clean all build artifacts", "Built-in"),
            ("deps", "Install development dependencies", "Built-in"),
            ("test", "Run verification tests", "Built-in"),
            ("all", "Complete build pipeline", "All scripts"),
            ("interactive", "Interactive build wizard", "Built-in"),
        ]

        for cmd, desc, script in commands:
            commands_table.add_row(cmd, desc, script)

        self.console.print(commands_table)

    async def run_command(self, command: str, args: List[str] = None) -> bool:
        """Run a specific build command"""
        args = args or []

        if command == "install":
            return await self._run_install(args)
        elif command == "compile":
            return await self._run_compile(args)
        elif command == "build":
            return await self._run_build(args)
        elif command == "package":
            return await self._run_package(args)
        elif command == "verify":
            return await self._run_verify(args)
        elif command == "clean":
            return await self._run_clean(args)
        elif command == "deps":
            return await self._run_deps(args)
        elif command == "test":
            return await self._run_test(args)
        elif command == "all":
            return await self._run_all_pipeline(args)
        elif command == "interactive":
            return await self._run_interactive()
        else:
            self.console.print(f"[red]❌ Unknown command: {command}[/red]")
            return False

    async def _run_install(self, args: List[str]) -> bool:
        """Run installation script"""
        self.console.print("[bold blue]🚀 Running Terminal Coder Installation...[/bold blue]")

        install_script = self.project_root / "install.py"
        if not install_script.exists():
            self.console.print("[red]❌ install.py not found[/red]")
            return False

        cmd = [sys.executable, str(install_script)] + args
        return await self._execute_subprocess(cmd, "Installation")

    async def _run_compile(self, args: List[str]) -> bool:
        """Run compilation script"""
        self.console.print("[bold blue]🔨 Running Terminal Coder Compilation...[/bold blue]")

        compile_script = self.project_root / "compile.py"
        if not compile_script.exists():
            self.console.print("[red]❌ compile.py not found[/red]")
            return False

        cmd = [sys.executable, str(compile_script)] + args
        return await self._execute_subprocess(cmd, "Compilation")

    async def _run_build(self, args: List[str]) -> bool:
        """Run comprehensive build script"""
        self.console.print("[bold blue]🏗️ Running Comprehensive Build System...[/bold blue]")

        build_script = self.project_root / "build.py"
        if not build_script.exists():
            self.console.print("[red]❌ build.py not found[/red]")
            return False

        cmd = [sys.executable, str(build_script)] + args
        return await self._execute_subprocess(cmd, "Build")

    async def _run_package(self, args: List[str]) -> bool:
        """Run Python packaging"""
        self.console.print("[bold blue]📦 Running Python Package Creation...[/bold blue]")

        # First try with build module
        try:
            cmd = [sys.executable, "-m", "build"] + args
            result = await self._execute_subprocess(cmd, "Package Build", timeout=300)
            if result:
                return True
        except Exception:
            pass

        # Fallback to setup.py
        setup_script = self.project_root / "setup.py"
        if setup_script.exists():
            cmd = [sys.executable, str(setup_script), "bdist_wheel", "sdist"] + args
            return await self._execute_subprocess(cmd, "Package Setup")
        else:
            self.console.print("[red]❌ No packaging configuration found[/red]")
            return False

    async def _run_verify(self, args: List[str]) -> bool:
        """Run build verification"""
        self.console.print("[bold blue]🔍 Running Build Verification...[/bold blue]")

        verify_script = self.project_root / "verify_build.py"
        if not verify_script.exists():
            self.console.print("[red]❌ verify_build.py not found[/red]")
            return False

        cmd = [sys.executable, str(verify_script)] + args
        return await self._execute_subprocess(cmd, "Verification")

    async def _run_clean(self, args: List[str]) -> bool:
        """Clean build artifacts"""
        self.console.print("[bold blue]🧹 Cleaning Build Artifacts...[/bold blue]")

        import shutil

        clean_dirs = ["dist", "build", "__pycache__", ".pytest_cache", "*.egg-info"]
        cleaned = []

        for pattern in clean_dirs:
            if pattern.startswith("*"):
                # Handle glob patterns
                for path in self.project_root.glob(pattern):
                    if path.is_dir():
                        shutil.rmtree(path)
                        cleaned.append(str(path))
                    elif path.is_file():
                        path.unlink()
                        cleaned.append(str(path))
            else:
                path = self.project_root / pattern
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        cleaned.append(str(path))
                    else:
                        path.unlink()
                        cleaned.append(str(path))

        # Clean Python cache files
        for pyc_file in self.project_root.rglob("*.pyc"):
            try:
                pyc_file.unlink()
                cleaned.append(str(pyc_file))
            except Exception:
                pass

        for pycache_dir in self.project_root.rglob("__pycache__"):
            try:
                if pycache_dir.is_dir():
                    shutil.rmtree(pycache_dir)
                    cleaned.append(str(pycache_dir))
            except Exception:
                pass

        if cleaned:
            self.console.print(f"[green]✅ Cleaned {len(cleaned)} build artifacts[/green]")
            if "--verbose" in args:
                for item in cleaned[:10]:  # Show first 10 items
                    self.console.print(f"[dim]  • Cleaned: {item}[/dim]")
                if len(cleaned) > 10:
                    self.console.print(f"[dim]  • ... and {len(cleaned) - 10} more items[/dim]")
        else:
            self.console.print("[yellow]⚠️ No build artifacts found to clean[/yellow]")

        return True

    async def _run_deps(self, args: List[str]) -> bool:
        """Install development dependencies"""
        self.console.print("[bold blue]📚 Installing Development Dependencies...[/bold blue]")

        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.console.print("[red]❌ requirements.txt not found[/red]")
            return False

        # Install requirements
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        if "--upgrade" in args:
            cmd.append("--upgrade")

        result = await self._execute_subprocess(cmd, "Dependency Installation")

        # Also install development extras if setup.py exists
        setup_file = self.project_root / "setup.py"
        if setup_file.exists() and result:
            dev_cmd = [sys.executable, "-m", "pip", "install", "-e", ".[dev,build]"]
            return await self._execute_subprocess(dev_cmd, "Development Extras")

        return result

    async def _run_test(self, args: List[str]) -> bool:
        """Run verification tests"""
        self.console.print("[bold blue]🧪 Running Tests and Verification...[/bold blue]")

        # Run verification script
        success = await self._run_verify(args)

        # Run pytest if available
        try:
            pytest_cmd = [sys.executable, "-m", "pytest"] + args
            pytest_result = await self._execute_subprocess(pytest_cmd, "PyTest", timeout=300)
            return success and pytest_result
        except Exception:
            self.console.print("[yellow]⚠️ PyTest not available, skipping unit tests[/yellow]")
            return success

    async def _run_all_pipeline(self, args: List[str]) -> bool:
        """Run complete build pipeline"""
        self.console.print("[bold blue]🚀 Running Complete Build Pipeline...[/bold blue]")

        pipeline_steps = [
            ("Clean", self._run_clean, []),
            ("Install Dependencies", self._run_deps, ["--upgrade"]),
            ("Verify System", self._run_verify, []),
            ("Create Packages", self._run_package, []),
            ("Compile Executables", self._run_compile, []),
            ("Comprehensive Build", self._run_build, []),
            ("Final Verification", self._run_verify, [])
        ]

        self.console.print(f"[cyan]📋 Pipeline has {len(pipeline_steps)} steps[/cyan]")

        for i, (step_name, step_func, step_args) in enumerate(pipeline_steps, 1):
            self.console.print(f"\n[bold blue]📍 Step {i}/{len(pipeline_steps)}: {step_name}[/bold blue]")

            success = await step_func(step_args + args)
            if not success:
                self.console.print(f"[red]❌ Pipeline failed at step {i}: {step_name}[/red]")
                return False

            self.console.print(f"[green]✅ Step {i} completed: {step_name}[/green]")

        self.console.print("\n[bold green]🎉 Complete build pipeline finished successfully![/bold green]")
        return True

    async def _run_interactive(self) -> bool:
        """Run interactive build wizard"""
        self.console.print("[bold blue]🧙 Interactive Build Wizard[/bold blue]")

        # Ask user what they want to do
        action_choices = [
            "1. Quick Install (install with defaults)",
            "2. Development Setup (install + dev dependencies)",
            "3. Compile Executables (create standalone apps)",
            "4. Create Distribution (packages + executables)",
            "5. Full Pipeline (clean + build everything)",
            "6. Verify Build System",
            "7. Clean Build Artifacts",
            "8. Custom Command"
        ]

        self.console.print("\n[cyan]What would you like to do?[/cyan]")
        for choice in action_choices:
            self.console.print(f"  {choice}")

        try:
            choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3", "4", "5", "6", "7", "8"])
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Interactive wizard cancelled[/yellow]")
            return False

        if choice == "1":
            return await self._run_install([])
        elif choice == "2":
            success1 = await self._run_deps(["--upgrade"])
            success2 = await self._run_install(["--dev-deps"])
            return success1 and success2
        elif choice == "3":
            # Ask for compilation options
            platforms = []
            if Confirm.ask("Compile for current platform?", default=True):
                platforms.append("--platforms=current")

            options = []
            if Confirm.ask("Include GUI version?", default=True):
                pass  # Default is to include GUI
            else:
                options.append("--no-gui")

            if Confirm.ask("Include AI features?", default=True):
                pass  # Default is to include AI
            else:
                options.append("--no-ai")

            if Confirm.ask("Create single-file executable?", default=True):
                pass  # Default is onefile
            else:
                options.append("--no-onefile")

            return await self._run_compile(platforms + options)
        elif choice == "4":
            success1 = await self._run_package([])
            success2 = await self._run_compile([])
            return success1 and success2
        elif choice == "5":
            confirm = Confirm.ask("This will run the complete pipeline (may take several minutes). Continue?", default=True)
            if confirm:
                return await self._run_all_pipeline([])
            return True
        elif choice == "6":
            return await self._run_verify([])
        elif choice == "7":
            verbose = Confirm.ask("Show detailed cleaning output?", default=False)
            args = ["--verbose"] if verbose else []
            return await self._run_clean(args)
        elif choice == "8":
            self.display_available_commands()
            try:
                custom_cmd = Prompt.ask("\nEnter command")
                custom_args = Prompt.ask("Enter arguments (optional)", default="").split()
                return await self.run_command(custom_cmd, custom_args)
            except (KeyboardInterrupt, EOFError):
                return False

        return False

    async def _execute_subprocess(self, cmd: List[str], operation_name: str, timeout: int = 600) -> bool:
        """Execute subprocess with proper error handling"""
        try:
            self.console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                self.console.print(f"[red]❌ {operation_name} timed out after {timeout} seconds[/red]")
                return False

            if process.returncode == 0:
                self.console.print(f"[green]✅ {operation_name} completed successfully[/green]")
                if stdout:
                    self.console.print("[dim]Output preview:[/dim]")
                    output_lines = stdout.decode().strip().split('\n')
                    for line in output_lines[-5:]:  # Show last 5 lines
                        self.console.print(f"[dim]  {line}[/dim]")
                return True
            else:
                self.console.print(f"[red]❌ {operation_name} failed (exit code: {process.returncode})[/red]")
                if stderr:
                    error_lines = stderr.decode().strip().split('\n')
                    self.console.print("[red]Error output:[/red]")
                    for line in error_lines[-10:]:  # Show last 10 error lines
                        self.console.print(f"[red]  {line}[/red]")
                return False

        except Exception as e:
            self.console.print(f"[red]❌ Failed to execute {operation_name}: {e}[/red]")
            return False


async def main():
    """Main orchestrator function"""
    parser = argparse.ArgumentParser(
        description="Terminal Coder Universal Build Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  install     - Install Terminal Coder with dependencies
  compile     - Compile to standalone executables
  build       - Complete build with all formats
  package     - Create Python packages (wheel/sdist)
  verify      - Verify build system integrity
  clean       - Clean all build artifacts
  deps        - Install development dependencies
  test        - Run verification tests
  all         - Complete build pipeline
  interactive - Interactive build wizard

Examples:
  python make.py install
  python make.py compile --platforms=windows,linux
  python make.py build --no-docker
  python make.py all
  python make.py interactive
        """
    )

    parser.add_argument("command", nargs="?", default="interactive",
                       help="Build command to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                       help="Arguments to pass to the build command")

    args = parser.parse_args()

    builder = UniversalBuilder()
    builder.display_banner()

    if args.command == "help" or (args.command == "interactive" and len(sys.argv) == 1):
        builder.display_available_commands()

        if args.command != "interactive":
            return

    try:
        success = await builder.run_command(args.command, args.args)

        if success:
            print("\n🎉 Build operation completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Build operation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Build operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Build operation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())