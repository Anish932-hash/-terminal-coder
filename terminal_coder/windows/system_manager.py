#!/usr/bin/env python3
"""
Windows System Manager
Handles Windows-specific system operations and integrations
"""

import os
import sys
import subprocess
import winreg
import wmi
import psutil
import win32api
import win32con
import win32security
import win32service
import win32serviceutil
import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import asyncio
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@dataclass
class WindowsService:
    """Windows service information"""
    name: str
    display_name: str
    status: str
    start_type: str
    description: str


@dataclass
class WindowsProcess:
    """Windows process information"""
    pid: int
    name: str
    cpu_percent: float
    memory_info: dict
    status: str
    create_time: datetime


@dataclass
class WindowsSystemInfo:
    """Windows system information"""
    os_version: str
    build_number: str
    edition: str
    architecture: str
    processor: str
    total_memory: int
    available_memory: int
    disk_usage: dict
    network_interfaces: list


class WindowsSystemManager:
    """Advanced Windows system management"""

    def __init__(self):
        self.console = Console()
        self.wmi_client = None
        self._init_wmi()

    def _init_wmi(self):
        """Initialize WMI client"""
        try:
            self.wmi_client = wmi.WMI()
        except Exception as e:
            self.console.print(f"[yellow]Warning: WMI not available: {e}[/yellow]")

    def get_system_info(self) -> WindowsSystemInfo:
        """Get comprehensive Windows system information"""
        try:
            # Get OS information
            os_info = self._get_os_info()

            # Get hardware information
            processor = self._get_processor_info()
            memory = self._get_memory_info()
            disk = self._get_disk_info()
            network = self._get_network_info()

            return WindowsSystemInfo(
                os_version=os_info.get('version', 'Unknown'),
                build_number=os_info.get('build', 'Unknown'),
                edition=os_info.get('edition', 'Unknown'),
                architecture=os_info.get('architecture', 'Unknown'),
                processor=processor,
                total_memory=memory['total'],
                available_memory=memory['available'],
                disk_usage=disk,
                network_interfaces=network
            )
        except Exception as e:
            self.console.print(f"[red]Error getting system info: {e}[/red]")
            return None

    def _get_os_info(self) -> Dict[str, str]:
        """Get Windows OS information"""
        try:
            if self.wmi_client:
                os_info = self.wmi_client.Win32_OperatingSystem()[0]
                return {
                    'version': os_info.Version,
                    'build': os_info.BuildNumber,
                    'edition': os_info.Caption,
                    'architecture': os_info.OSArchitecture
                }
            else:
                import platform
                return {
                    'version': platform.version(),
                    'build': platform.win32_ver()[1],
                    'edition': platform.platform(),
                    'architecture': platform.architecture()[0]
                }
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not get OS info: {e}[/yellow]")
            return {}

    def _get_processor_info(self) -> str:
        """Get processor information"""
        try:
            if self.wmi_client:
                processor = self.wmi_client.Win32_Processor()[0]
                return f"{processor.Name} ({processor.NumberOfCores} cores)"
            else:
                import platform
                return platform.processor()
        except Exception:
            return "Unknown Processor"

    def _get_memory_info(self) -> Dict[str, int]:
        """Get memory information"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percentage': memory.percent
            }
        except Exception:
            return {'total': 0, 'available': 0, 'used': 0, 'percentage': 0}

    def _get_disk_info(self) -> Dict[str, Dict]:
        """Get disk usage information"""
        try:
            disk_info = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info[partition.device] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percentage': (usage.used / usage.total) * 100 if usage.total > 0 else 0,
                        'fstype': partition.fstype
                    }
                except PermissionError:
                    continue
            return disk_info
        except Exception:
            return {}

    def _get_network_info(self) -> List[Dict]:
        """Get network interface information"""
        try:
            interfaces = []
            for interface, addrs in psutil.net_if_addrs().items():
                interface_info = {'name': interface, 'addresses': []}
                for addr in addrs:
                    interface_info['addresses'].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
                interfaces.append(interface_info)
            return interfaces
        except Exception:
            return []

    def get_running_services(self) -> List[WindowsService]:
        """Get list of Windows services"""
        services = []
        try:
            for service in psutil.win_service_iter():
                try:
                    service_info = service.as_dict()
                    services.append(WindowsService(
                        name=service_info.get('name', 'Unknown'),
                        display_name=service_info.get('display_name', 'Unknown'),
                        status=service_info.get('status', 'Unknown'),
                        start_type=service_info.get('start_type', 'Unknown'),
                        description=service_info.get('description', 'No description')
                    ))
                except Exception:
                    continue
        except Exception as e:
            self.console.print(f"[red]Error getting services: {e}[/red]")

        return services

    def get_running_processes(self) -> List[WindowsProcess]:
        """Get list of running processes"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status', 'create_time']):
                try:
                    proc_info = proc.info
                    processes.append(WindowsProcess(
                        pid=proc_info['pid'],
                        name=proc_info['name'],
                        cpu_percent=proc_info['cpu_percent'] or 0.0,
                        memory_info=proc_info['memory_info']._asdict() if proc_info['memory_info'] else {},
                        status=proc_info['status'],
                        create_time=datetime.fromtimestamp(proc_info['create_time']) if proc_info['create_time'] else datetime.now()
                    ))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.console.print(f"[red]Error getting processes: {e}[/red]")

        return processes

    def manage_service(self, service_name: str, action: str) -> bool:
        """Manage Windows service (start, stop, restart)"""
        try:
            if action.lower() == 'start':
                win32serviceutil.StartService(service_name)
            elif action.lower() == 'stop':
                win32serviceutil.StopService(service_name)
            elif action.lower() == 'restart':
                win32serviceutil.RestartService(service_name)
            else:
                return False
            return True
        except Exception as e:
            self.console.print(f"[red]Error managing service {service_name}: {e}[/red]")
            return False

    def get_registry_value(self, hkey: int, subkey: str, value_name: str) -> Optional[Any]:
        """Get value from Windows registry"""
        try:
            with winreg.OpenKey(hkey, subkey) as key:
                value, _ = winreg.QueryValueEx(key, value_name)
                return value
        except Exception as e:
            self.console.print(f"[yellow]Registry read error: {e}[/yellow]")
            return None

    def set_registry_value(self, hkey: int, subkey: str, value_name: str, value: Any, value_type: int) -> bool:
        """Set value in Windows registry"""
        try:
            with winreg.CreateKey(hkey, subkey) as key:
                winreg.SetValueEx(key, value_name, 0, value_type, value)
                return True
        except Exception as e:
            self.console.print(f"[red]Registry write error: {e}[/red]")
            return False

    def execute_powershell(self, script: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute PowerShell script"""
        try:
            result = subprocess.run(
                ['powershell', '-Command', script],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8'
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "PowerShell script timed out"
        except Exception as e:
            return False, "", str(e)

    def execute_cmd(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute CMD command"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8'
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def check_admin_privileges(self) -> bool:
        """Check if running with administrator privileges"""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except Exception:
            return False

    def request_admin_privileges(self):
        """Request administrator privileges"""
        if not self.check_admin_privileges():
            try:
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, " ".join(sys.argv), None, 1
                )
                sys.exit(0)
            except Exception as e:
                self.console.print(f"[red]Failed to elevate privileges: {e}[/red]")
                return False
        return True

    def get_environment_variables(self) -> Dict[str, str]:
        """Get all environment variables"""
        return dict(os.environ)

    def set_environment_variable(self, name: str, value: str, permanent: bool = False) -> bool:
        """Set environment variable"""
        try:
            os.environ[name] = value
            if permanent:
                # Set in registry for permanent storage
                reg_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                return self.set_registry_value(
                    winreg.HKEY_LOCAL_MACHINE,
                    reg_path,
                    name,
                    value,
                    winreg.REG_SZ
                )
            return True
        except Exception as e:
            self.console.print(f"[red]Error setting environment variable: {e}[/red]")
            return False

    def get_installed_programs(self) -> List[Dict[str, str]]:
        """Get list of installed programs"""
        programs = []
        try:
            # Check both 32-bit and 64-bit registry keys
            reg_paths = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
            ]

            for reg_path in reg_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                        for i in range(winreg.QueryInfoKey(key)[0]):
                            try:
                                subkey_name = winreg.EnumKey(key, i)
                                with winreg.OpenKey(key, subkey_name) as subkey:
                                    try:
                                        name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                        try:
                                            version = winreg.QueryValueEx(subkey, "DisplayVersion")[0]
                                        except FileNotFoundError:
                                            version = "Unknown"

                                        programs.append({
                                            'name': name,
                                            'version': version,
                                            'registry_key': subkey_name
                                        })
                                    except FileNotFoundError:
                                        continue
                            except (OSError, FileNotFoundError):
                                continue
                except Exception:
                    continue
        except Exception as e:
            self.console.print(f"[red]Error getting installed programs: {e}[/red]")

        return programs

    def monitor_system_performance(self, duration: int = 60) -> Dict[str, Any]:
        """Monitor system performance for specified duration"""
        performance_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'timestamp': []
        }

        try:
            start_time = datetime.now()
            while (datetime.now() - start_time).seconds < duration:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                performance_data['cpu_usage'].append(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                performance_data['memory_usage'].append(memory.percent)

                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    performance_data['disk_io'].append({
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    })

                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    performance_data['network_io'].append({
                        'bytes_sent': network_io.bytes_sent,
                        'bytes_recv': network_io.bytes_recv
                    })

                performance_data['timestamp'].append(datetime.now())

        except Exception as e:
            self.console.print(f"[red]Error monitoring performance: {e}[/red]")

        return performance_data

    def display_system_info(self):
        """Display comprehensive system information"""
        system_info = self.get_system_info()

        if not system_info:
            self.console.print("[red]Could not retrieve system information[/red]")
            return

        # System Information Table
        table = Table(title="🖥️ Windows System Information", style="cyan")
        table.add_column("Property", style="magenta", width=20)
        table.add_column("Value", style="white")

        table.add_row("OS Version", system_info.os_version)
        table.add_row("Build Number", system_info.build_number)
        table.add_row("Edition", system_info.edition)
        table.add_row("Architecture", system_info.architecture)
        table.add_row("Processor", system_info.processor)
        table.add_row("Total Memory", f"{system_info.total_memory / (1024**3):.2f} GB")
        table.add_row("Available Memory", f"{system_info.available_memory / (1024**3):.2f} GB")

        self.console.print(table)

        # Disk Usage Table
        if system_info.disk_usage:
            disk_table = Table(title="💾 Disk Usage", style="green")
            disk_table.add_column("Drive", style="cyan")
            disk_table.add_column("Total", style="white")
            disk_table.add_column("Used", style="yellow")
            disk_table.add_column("Free", style="green")
            disk_table.add_column("Usage %", style="red")

            for drive, info in system_info.disk_usage.items():
                total_gb = info['total'] / (1024**3)
                used_gb = info['used'] / (1024**3)
                free_gb = info['free'] / (1024**3)

                disk_table.add_row(
                    drive,
                    f"{total_gb:.2f} GB",
                    f"{used_gb:.2f} GB",
                    f"{free_gb:.2f} GB",
                    f"{info['percentage']:.1f}%"
                )

            self.console.print(disk_table)

    def display_services_info(self, filter_running: bool = True):
        """Display Windows services information"""
        services = self.get_running_services()

        if filter_running:
            services = [s for s in services if s.status.lower() == 'running']

        table = Table(title=f"⚙️ Windows Services ({'Running Only' if filter_running else 'All'})", style="blue")
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="white")
        table.add_column("Status", style="green")
        table.add_column("Start Type", style="yellow")

        for service in services[:50]:  # Show first 50
            table.add_row(
                service.name,
                service.display_name[:50] + "..." if len(service.display_name) > 50 else service.display_name,
                service.status,
                service.start_type
            )

        self.console.print(table)

    def display_processes_info(self, sort_by: str = "cpu", top_n: int = 20):
        """Display running processes information"""
        processes = self.get_running_processes()

        # Sort processes
        if sort_by.lower() == "cpu":
            processes.sort(key=lambda x: x.cpu_percent, reverse=True)
        elif sort_by.lower() == "memory":
            processes.sort(key=lambda x: x.memory_info.get('rss', 0), reverse=True)

        table = Table(title=f"🔄 Top {top_n} Processes (by {sort_by.upper()})", style="magenta")
        table.add_column("PID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("CPU %", style="red")
        table.add_column("Memory (MB)", style="yellow")
        table.add_column("Status", style="green")

        for process in processes[:top_n]:
            memory_mb = process.memory_info.get('rss', 0) / (1024 * 1024)
            table.add_row(
                str(process.pid),
                process.name[:30] + "..." if len(process.name) > 30 else process.name,
                f"{process.cpu_percent:.1f}%",
                f"{memory_mb:.1f} MB",
                process.status
            )

        self.console.print(table)