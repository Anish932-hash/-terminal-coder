#!/usr/bin/env python3
"""
Terminal Coder - Advanced Compilation Script
Comprehensive cross-platform compiler with executable generation
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
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import hashlib
from datetime import datetime
import threading
import time

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.align import Align
    from rich.layout import Layout
    from rich.live import Live
    import rich.traceback
    RICH_AVAILABLE = True
    rich.traceback.install(show_locals=True)
except ImportError:
    RICH_AVAILABLE = False
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    def Panel(text, **kwargs):
        return f"=== {text} ==="

    def Table(**kwargs):
        class MockTable:
            def add_column(self, *args, **kwargs): pass
            def add_row(self, *args, **kwargs): pass
        return MockTable()


@dataclass
class CompileConfig:
    """Compilation configuration"""
    output_dir: Path
    target_platforms: List[str] = field(default_factory=lambda: ['current'])
    include_dependencies: bool = True
    optimize: bool = True
    include_gui: bool = True
    include_ai: bool = True
    debug_mode: bool = False
    onefile: bool = True
    upx_compress: bool = False
    console_mode: bool = True
    windowed_mode: bool = False
    icon_path: Optional[Path] = None
    version_info: Optional[Dict[str, str]] = None
    exclude_modules: List[str] = field(default_factory=list)
    include_data: List[str] = field(default_factory=list)
    custom_hooks: List[str] = field(default_factory=list)


class PlatformCompiler:
    """Platform-specific compilation management"""

    def __init__(self, console: Console):
        self.console = console
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.temp_dir = Path(tempfile.mkdtemp(prefix='terminal_coder_compile_'))

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.temp_dir / 'compile.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_compilation_requirements(self) -> Tuple[bool, List[str]]:
        """Check if compilation requirements are met"""
        requirements_met = True
        missing_requirements = []

        # Check Python version
        if sys.version_info < (3, 8):
            requirements_met = False
            missing_requirements.append("Python 3.8+ required")

        # Check PyInstaller
        try:
            import PyInstaller
            self.console.print(f"[green]✅ PyInstaller {PyInstaller.__version__} found[/green]")
        except ImportError:
            requirements_met = False
            missing_requirements.append("PyInstaller not installed")

        # Check platform-specific tools
        if self.system == 'windows':
            # Check for Visual Studio Build Tools
            vs_paths = [
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
                r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
                r"C:\Program Files\Microsoft Visual Studio\2019\Community",
                r"C:\Program Files\Microsoft Visual Studio\2022\Community"
            ]

            vs_found = any(Path(path).exists() for path in vs_paths)
            if not vs_found:
                missing_requirements.append("Visual Studio Build Tools recommended")

        elif self.system == 'linux':
            # Check for development tools
            required_tools = ['gcc', 'make', 'pkg-config']
            for tool in required_tools:
                if not shutil.which(tool):
                    missing_requirements.append(f"Development tool missing: {tool}")
                    requirements_met = False

        # Check for UPX if compression is enabled
        if shutil.which('upx'):
            self.console.print("[green]✅ UPX compressor found[/green]")
        else:
            missing_requirements.append("UPX compressor not found (optional)")

        return requirements_met, missing_requirements

    async def compile_application(self, config: CompileConfig) -> bool:
        """Compile the application for specified platforms"""
        try:
            self._display_compilation_banner()

            # Check requirements
            requirements_met, missing = self.check_compilation_requirements()
            if not requirements_met:
                self.console.print("[red]❌ Compilation requirements not met:[/red]")
                for req in missing:
                    self.console.print(f"  • {req}")
                return False

            if missing:
                self.console.print("[yellow]⚠️ Optional requirements missing:[/yellow]")
                for req in missing:
                    self.console.print(f"  • {req}")

            # Prepare compilation environment
            if not await self._prepare_compilation_environment(config):
                return False

            # Compile for each target platform
            success = True
            for platform_target in config.target_platforms:
                self.console.print(f"\n[bold blue]🔨 Compiling for {platform_target}...[/bold blue]")

                if not await self._compile_for_platform(platform_target, config):
                    success = False
                    self.console.print(f"[red]❌ Compilation failed for {platform_target}[/red]")
                else:
                    self.console.print(f"[green]✅ Compilation successful for {platform_target}[/green]")

            # Create distribution packages
            if success:
                await self._create_distribution_packages(config)

            # Generate compilation report
            await self._generate_compilation_report(config)

            return success

        except Exception as e:
            self.console.print(f"[red]❌ Compilation failed: {e}[/red]")
            self.logger.error(f"Compilation error: {e}", exc_info=True)
            return False

        finally:
            await self._cleanup_temp_files()

    def _display_compilation_banner(self):
        """Display compilation banner"""
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔨 TERMINAL CODER COMPILER v2.0                          ║
║                   Cross-Platform Executable Generation                       ║
║                        System: {self.system.title()} {self.machine}                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """

        self.console.print(Panel(
            Align.center(banner),
            style="bold cyan",
            border_style="bright_blue"
        ))

    async def _prepare_compilation_environment(self, config: CompileConfig) -> bool:
        """Prepare compilation environment"""
        try:
            # Create output directories
            config.output_dir.mkdir(parents=True, exist_ok=True)

            # Create build directory structure
            build_dir = self.temp_dir / 'build'
            build_dir.mkdir(exist_ok=True)

            # Copy source files
            source_dir = Path(__file__).parent
            target_dir = build_dir / 'terminal_coder'

            self.console.print("[blue]📂 Copying source files...[/blue]")

            # Copy Python source files
            await self._copy_source_files(source_dir, target_dir)

            # Generate entry points
            await self._generate_entry_points(target_dir, config)

            # Create spec files for PyInstaller
            await self._create_pyinstaller_specs(target_dir, config)

            return True

        except Exception as e:
            self.console.print(f"[red]❌ Environment preparation failed: {e}[/red]")
            return False

    async def _copy_source_files(self, source_dir: Path, target_dir: Path):
        """Copy source files for compilation"""
        target_dir.mkdir(parents=True, exist_ok=True)

        # Files and directories to copy
        files_to_copy = [
            'linux/',
            'windows/',
            'install.py',
            '__init__.py'
        ]

        for item in files_to_copy:
            source_path = source_dir / item
            target_path = target_dir / item

            if source_path.is_file():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                self.console.print(f"[dim]  • Copied {item}[/dim]")
            elif source_path.is_dir():
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                self.console.print(f"[dim]  • Copied directory {item}[/dim]")

        # Create __init__.py files where needed
        init_dirs = [
            target_dir,
            target_dir / 'linux',
            target_dir / 'windows'
        ]

        for init_dir in init_dirs:
            if init_dir.exists():
                init_file = init_dir / '__init__.py'
                if not init_file.exists():
                    init_file.write_text('# Terminal Coder Package\n__version__ = "2.0.0"\n')

    async def _generate_entry_points(self, target_dir: Path, config: CompileConfig):
        """Generate entry point scripts for compilation"""

        # Main entry point
        main_entry = target_dir / 'main.py'
        main_content = '''#!/usr/bin/env python3
"""
Terminal Coder - Main Entry Point
Cross-platform launcher that detects system and runs appropriate version
"""

import sys
import platform
from pathlib import Path

def main():
    """Main entry point that detects platform and launches appropriate version"""
    system = platform.system().lower()

    try:
        if system == 'linux':
            from terminal_coder.linux.main import main as linux_main
            linux_main()
        elif system == 'windows':
            from terminal_coder.windows.main import main as windows_main
            windows_main()
        elif system == 'darwin':
            # For now, use Linux version on macOS
            from terminal_coder.linux.main import main as linux_main
            linux_main()
        else:
            print(f"Unsupported platform: {system}")
            sys.exit(1)

    except ImportError as e:
        print(f"Error importing platform-specific modules: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running Terminal Coder: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        main_entry.write_text(main_content)
        self.console.print("[green]✅ Generated main entry point[/green]")

        # GUI entry point
        if config.include_gui:
            gui_entry = target_dir / 'gui_main.py'
            gui_content = '''#!/usr/bin/env python3
"""
Terminal Coder - GUI Entry Point
Cross-platform GUI launcher
"""

import sys
import platform
import tkinter as tk
from tkinter import messagebox

def main():
    """GUI entry point"""
    system = platform.system().lower()

    try:
        # Initialize GUI
        root = tk.Tk()
        root.withdraw()  # Hide root window initially

        if system == 'linux':
            from terminal_coder.linux.gui import LinuxGUI
            gui = LinuxGUI()
        elif system == 'windows':
            from terminal_coder.windows.gui import WindowsGUI
            gui = WindowsGUI()
        else:
            messagebox.showerror("Error", f"GUI not supported on {system}")
            sys.exit(1)

        # Run GUI
        if gui.initialize():
            gui.run()
        else:
            messagebox.showerror("Error", "Failed to initialize GUI")

    except ImportError as e:
        messagebox.showerror("Import Error", f"Error importing GUI modules: {e}")
        sys.exit(1)
    except Exception as e:
        messagebox.showerror("Error", f"Error running Terminal Coder GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

            gui_entry.write_text(gui_content)
            self.console.print("[green]✅ Generated GUI entry point[/green]")

    async def _create_pyinstaller_specs(self, target_dir: Path, config: CompileConfig):
        """Create PyInstaller specification files"""

        # Common spec template
        spec_template = '''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Add source directory to Python path
sys.path.insert(0, '{source_dir}')

# Analysis configuration
a = Analysis(
    ['{entry_point}'],
    pathex=['{source_dir}'],
    binaries=[],
    datas={datas},
    hiddenimports={hidden_imports},
    hookspath={hook_paths},
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ configuration
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# EXE configuration
exe = EXE(
    pyz,
    a.scripts,
    {exe_options}
    exclude_binaries={onefile_exclude},
    name='{exe_name}',
    debug={debug},
    bootloader_ignore_signals=False,
    strip=False,
    upx={upx_compress},
    upx_exclude=[],
    runtime_tmpdir=None,
    console={console_mode},
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {icon_option}
    {version_info}
)

{coll_section}
'''

        # Console application spec
        console_spec_path = target_dir / 'terminal_coder_console.spec'
        console_spec = self._generate_spec_content(
            spec_template, target_dir, config,
            entry_point='main.py',
            exe_name='terminal-coder',
            console_mode=True,
            windowed_mode=False
        )
        console_spec_path.write_text(console_spec)

        # GUI application spec (if enabled)
        if config.include_gui:
            gui_spec_path = target_dir / 'terminal_coder_gui.spec'
            gui_spec = self._generate_spec_content(
                spec_template, target_dir, config,
                entry_point='gui_main.py',
                exe_name='terminal-coder-gui',
                console_mode=False,
                windowed_mode=True
            )
            gui_spec_path.write_text(gui_spec)

        self.console.print("[green]✅ Generated PyInstaller spec files[/green]")

    def _generate_spec_content(self, template: str, target_dir: Path, config: CompileConfig,
                             entry_point: str, exe_name: str, console_mode: bool, windowed_mode: bool) -> str:
        """Generate PyInstaller spec file content"""

        # Data files to include
        datas = [
            "('linux/*.py', 'terminal_coder/linux')",
            "('windows/*.py', 'terminal_coder/windows')",
        ]

        if config.include_data:
            for data_path in config.include_data:
                datas.append(f"('{data_path}', '.')")

        # Hidden imports
        hidden_imports = [
            "'asyncio'", "'threading'", "'json'", "'pathlib'", "'subprocess'",
            "'logging'", "'datetime'", "'hashlib'", "'tempfile'", "'shutil'",
            "'platform'", "'sys'", "'os'", "'time'", "'argparse'",
            "'rich.console'", "'rich.panel'", "'rich.table'", "'rich.progress'",
            "'rich.prompt'", "'rich.text'", "'rich.align'", "'rich.layout'",
            "'psutil'", "'requests'", "'cryptography'", "'pydantic'"
        ]

        # Platform-specific hidden imports
        if self.system == 'windows':
            hidden_imports.extend([
                "'win32api'", "'win32con'", "'win32gui'", "'wmi'", "'winshell'"
            ])
        elif self.system == 'linux':
            hidden_imports.extend([
                "'distro'", "'keyring'", "'secretstorage'"
            ])

        # AI-related imports
        if config.include_ai:
            hidden_imports.extend([
                "'openai'", "'anthropic'", "'google.generativeai'", "'cohere'",
                "'tiktoken'", "'numpy'", "'sklearn'", "'torch'", "'transformers'"
            ])

        # GUI-related imports
        if config.include_gui:
            hidden_imports.extend([
                "'tkinter'", "'tkinter.ttk'", "'tkinter.messagebox'",
                "'tkinter.filedialog'", "'tkinter.scrolledtext'",
                "'matplotlib'", "'seaborn'", "'PIL'"
            ])

        # Hook paths
        hook_paths = []
        if config.custom_hooks:
            hook_paths = [f"'{hook}'" for hook in config.custom_hooks]

        # Exclude modules
        excludes = config.exclude_modules.copy()
        if not config.include_ai:
            excludes.extend(['torch', 'transformers', 'sklearn', 'tensorflow'])

        # EXE options for onefile vs directory
        if config.onefile:
            exe_options = "a.binaries,\n    a.zipfiles,\n    a.datas,"
            onefile_exclude = "False"
            coll_section = ""
        else:
            exe_options = ""
            onefile_exclude = "True"
            coll_section = f"""
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx={str(config.upx_compress).lower()},
    upx_exclude=[],
    name='{exe_name}'
)
"""

        # Icon option
        icon_option = ""
        if config.icon_path and config.icon_path.exists():
            icon_option = f"icon='{config.icon_path}',"

        # Version info
        version_info = ""
        if config.version_info and self.system == 'windows':
            version_info = f"version='{self._create_version_file(config.version_info)}',"

        return template.format(
            source_dir=str(target_dir).replace('\\', '/'),
            entry_point=entry_point,
            exe_name=exe_name,
            datas=str(datas),
            hidden_imports=str(hidden_imports),
            hook_paths=str(hook_paths),
            excludes=str(excludes),
            exe_options=exe_options,
            onefile_exclude=onefile_exclude,
            debug=str(config.debug_mode).lower(),
            upx_compress=str(config.upx_compress).lower(),
            console_mode=str(console_mode).lower(),
            icon_option=icon_option,
            version_info=version_info,
            coll_section=coll_section
        )

    def _create_version_file(self, version_info: Dict[str, str]) -> str:
        """Create Windows version info file"""
        version_file = self.temp_dir / 'version_info.txt'

        version_content = f'''# UTF-8
#
# Version Information for Terminal Coder

VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({version_info.get('version', '2.0.0.0').replace('.', ',')},),
    prodvers=({version_info.get('version', '2.0.0.0').replace('.', ',')},),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'{version_info.get('company', 'Terminal Coder Team')}'),
        StringStruct(u'FileDescription', u'{version_info.get('description', 'AI-Powered Development Terminal')}'),
        StringStruct(u'FileVersion', u'{version_info.get('version', '2.0.0.0')}'),
        StringStruct(u'InternalName', u'terminal-coder'),
        StringStruct(u'LegalCopyright', u'{version_info.get('copyright', '© 2024 Terminal Coder Team')}'),
        StringStruct(u'OriginalFilename', u'terminal-coder.exe'),
        StringStruct(u'ProductName', u'Terminal Coder'),
        StringStruct(u'ProductVersion', u'{version_info.get('version', '2.0.0.0')}')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''

        version_file.write_text(version_content)
        return str(version_file)

    async def _compile_for_platform(self, platform_target: str, config: CompileConfig) -> bool:
        """Compile application for specific platform"""
        try:
            build_dir = self.temp_dir / 'build' / 'terminal_coder'

            # Compile console version
            console_success = await self._run_pyinstaller(
                build_dir / 'terminal_coder_console.spec',
                config,
                f"console-{platform_target}"
            )

            gui_success = True
            if config.include_gui:
                # Compile GUI version
                gui_success = await self._run_pyinstaller(
                    build_dir / 'terminal_coder_gui.spec',
                    config,
                    f"gui-{platform_target}"
                )

            return console_success and gui_success

        except Exception as e:
            self.console.print(f"[red]❌ Platform compilation failed: {e}[/red]")
            return False

    async def _run_pyinstaller(self, spec_file: Path, config: CompileConfig, build_name: str) -> bool:
        """Run PyInstaller with specified spec file"""
        try:
            cmd = [
                sys.executable, '-m', 'PyInstaller',
                '--clean',
                '--noconfirm',
                str(spec_file)
            ]

            if config.debug_mode:
                cmd.append('--debug=all')

            # Set work path and dist path
            work_path = self.temp_dir / 'work' / build_name
            dist_path = config.output_dir / build_name

            work_path.mkdir(parents=True, exist_ok=True)
            dist_path.mkdir(parents=True, exist_ok=True)

            cmd.extend([
                '--workpath', str(work_path),
                '--distpath', str(dist_path)
            ])

            self.console.print(f"[blue]🔨 Running PyInstaller for {build_name}...[/blue]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"Compiling {build_name}...", total=None)

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=spec_file.parent
                )

                # Monitor process
                while process.poll() is None:
                    await asyncio.sleep(0.5)
                    progress.update(task, description=f"Compiling {build_name}...")

                stdout, stderr = process.communicate()

                if process.returncode == 0:
                    progress.update(task, description=f"✅ {build_name} compiled successfully")
                    self.console.print(f"[green]✅ {build_name} compilation successful[/green]")

                    # Log output
                    self.logger.info(f"PyInstaller output for {build_name}:\n{stdout}")

                    return True
                else:
                    self.console.print(f"[red]❌ {build_name} compilation failed[/red]")
                    self.console.print(f"[red]Error: {stderr}[/red]")
                    self.logger.error(f"PyInstaller error for {build_name}:\n{stderr}")
                    return False

        except Exception as e:
            self.console.print(f"[red]❌ PyInstaller execution failed: {e}[/red]")
            return False

    async def _create_distribution_packages(self, config: CompileConfig):
        """Create distribution packages (ZIP, TAR.GZ, etc.)"""
        self.console.print("\n[bold blue]📦 Creating distribution packages...[/bold blue]")

        try:
            for platform_target in config.target_platforms:
                # Create archives for each build
                builds = [f"console-{platform_target}"]
                if config.include_gui:
                    builds.append(f"gui-{platform_target}")

                for build in builds:
                    build_path = config.output_dir / build
                    if not build_path.exists():
                        continue

                    # Create ZIP archive
                    zip_path = config.output_dir / f"terminal-coder-{build}-{datetime.now().strftime('%Y%m%d')}.zip"
                    await self._create_zip_archive(build_path, zip_path)

                    # Create TAR.GZ archive for Unix systems
                    if platform_target != 'windows':
                        tar_path = config.output_dir / f"terminal-coder-{build}-{datetime.now().strftime('%Y%m%d')}.tar.gz"
                        await self._create_tar_archive(build_path, tar_path)

                    self.console.print(f"[green]✅ Created distribution package for {build}[/green]")

        except Exception as e:
            self.console.print(f"[red]❌ Package creation failed: {e}[/red]")

    async def _create_zip_archive(self, source_path: Path, zip_path: Path):
        """Create ZIP archive"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_path)
                    zipf.write(file_path, arcname)

        self.console.print(f"[dim]  • Created ZIP: {zip_path.name}[/dim]")

    async def _create_tar_archive(self, source_path: Path, tar_path: Path):
        """Create TAR.GZ archive"""
        with tarfile.open(tar_path, 'w:gz') as tarf:
            tarf.add(source_path, arcname=source_path.name)

        self.console.print(f"[dim]  • Created TAR.GZ: {tar_path.name}[/dim]")

    async def _generate_compilation_report(self, config: CompileConfig):
        """Generate compilation report"""
        report_path = config.output_dir / 'compilation_report.json'

        report = {
            'compilation_info': {
                'timestamp': datetime.now().isoformat(),
                'compiler_version': sys.version,
                'platform': platform.platform(),
                'target_platforms': config.target_platforms,
                'configuration': {
                    'include_dependencies': config.include_dependencies,
                    'optimize': config.optimize,
                    'include_gui': config.include_gui,
                    'include_ai': config.include_ai,
                    'debug_mode': config.debug_mode,
                    'onefile': config.onefile,
                    'upx_compress': config.upx_compress
                }
            },
            'build_artifacts': [],
            'file_sizes': {},
            'checksums': {}
        }

        # Collect build artifacts
        for item in config.output_dir.iterdir():
            if item.is_file() and item.suffix in ['.exe', '.zip', '.tar.gz']:
                size = item.stat().st_size
                checksum = self._calculate_file_checksum(item)

                report['build_artifacts'].append(str(item.name))
                report['file_sizes'][item.name] = size
                report['checksums'][item.name] = checksum

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.console.print(f"[green]✅ Compilation report saved: {report_path}[/green]")

        # Display summary
        self._display_compilation_summary(report)

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _display_compilation_summary(self, report: Dict[str, Any]):
        """Display compilation summary"""
        summary_table = Table(title="📊 Compilation Summary", style="green")
        summary_table.add_column("Artifact", style="bold")
        summary_table.add_column("Size", style="white")
        summary_table.add_column("SHA256", style="dim")

        for artifact in report['build_artifacts']:
            size = report['file_sizes'][artifact]
            checksum = report['checksums'][artifact]

            # Format size
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{size / 1024:.1f} KB"

            summary_table.add_row(
                artifact,
                size_str,
                checksum[:16] + "..."
            )

        self.console.print(summary_table)

        # Success message
        success_panel = f"""
🎉 Compilation Completed Successfully!

📁 Output Directory: {report['compilation_info']['configuration']}
🕐 Compilation Time: {report['compilation_info']['timestamp']}
🖥️  Platform: {report['compilation_info']['platform']}

📦 Build Artifacts: {len(report['build_artifacts'])} files generated
📊 Total Size: {sum(report['file_sizes'].values()) / (1024 * 1024):.1f} MB

🚀 Your Terminal Coder executables are ready for distribution!
        """

        self.console.print(Panel(
            success_panel,
            title="🎊 Compilation Complete",
            style="bold green",
            border_style="bright_green"
        ))

    async def _cleanup_temp_files(self):
        """Clean up temporary compilation files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.console.print(f"[dim]🧹 Cleaned up temporary files: {self.temp_dir}[/dim]")
        except Exception as e:
            self.console.print(f"[yellow]⚠️ Could not clean up temp files: {e}[/yellow]")


class CompilationManager:
    """Main compilation management"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else Console()

    async def run_compilation(self, config: CompileConfig) -> bool:
        """Run the complete compilation process"""
        compiler = PlatformCompiler(self.console)
        return await compiler.compile_application(config)


async def main():
    """Main compilation function"""
    parser = argparse.ArgumentParser(description="Terminal Coder Advanced Compiler")
    parser.add_argument("--output-dir", type=Path, default=Path("./dist"),
                       help="Output directory for compiled executables")
    parser.add_argument("--platforms", nargs="+", default=["current"],
                       choices=["current", "windows", "linux", "macos"],
                       help="Target platforms for compilation")
    parser.add_argument("--no-dependencies", action="store_true",
                       help="Don't include dependencies")
    parser.add_argument("--no-optimize", action="store_true",
                       help="Disable optimizations")
    parser.add_argument("--no-gui", action="store_true",
                       help="Don't compile GUI version")
    parser.add_argument("--no-ai", action="store_true",
                       help="Don't include AI dependencies")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--no-onefile", action="store_true",
                       help="Create directory distribution instead of single file")
    parser.add_argument("--upx", action="store_true",
                       help="Enable UPX compression")
    parser.add_argument("--no-console", action="store_true",
                       help="Disable console mode for GUI")
    parser.add_argument("--windowed", action="store_true",
                       help="Enable windowed mode")
    parser.add_argument("--icon", type=Path,
                       help="Icon file for executable")
    parser.add_argument("--version-info", type=Path,
                       help="Version info JSON file")
    parser.add_argument("--exclude", nargs="+", default=[],
                       help="Modules to exclude")
    parser.add_argument("--include-data", nargs="+", default=[],
                       help="Additional data files to include")
    parser.add_argument("--hooks", nargs="+", default=[],
                       help="Custom PyInstaller hooks")

    args = parser.parse_args()

    # Load version info if provided
    version_info = None
    if args.version_info and args.version_info.exists():
        with open(args.version_info, 'r') as f:
            version_info = json.load(f)

    # Create compilation configuration
    config = CompileConfig(
        output_dir=args.output_dir,
        target_platforms=args.platforms,
        include_dependencies=not args.no_dependencies,
        optimize=not args.no_optimize,
        include_gui=not args.no_gui,
        include_ai=not args.no_ai,
        debug_mode=args.debug,
        onefile=not args.no_onefile,
        upx_compress=args.upx,
        console_mode=not args.no_console,
        windowed_mode=args.windowed,
        icon_path=args.icon,
        version_info=version_info,
        exclude_modules=args.exclude,
        include_data=args.include_data,
        custom_hooks=args.hooks
    )

    # Run compilation
    compiler = CompilationManager()
    success = await compiler.run_compilation(config)

    if success:
        print("\n🎉 Compilation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Compilation failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCompilation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Compilation failed: {e}")
        sys.exit(1)