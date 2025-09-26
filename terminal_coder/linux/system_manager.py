#!/usr/bin/env python3
"""
Linux System Manager
Handles Linux-specific system operations and integrations
"""

import os
import sys
import subprocess
import pwd
import grp
import psutil
import distro
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import asyncio
from datetime import datetime

try:
    import dbus
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@dataclass
class LinuxService:
    """Linux service information"""
    name: str
    status: str
    enabled: bool
    description: str
    unit_type: str


@dataclass
class LinuxProcess:
    """Linux process information"""
    pid: int
    ppid: int
    name: str
    username: str
    cpu_percent: float
    memory_percent: float
    status: str
    command: str


@dataclass
class LinuxSystemInfo:
    """Linux system information"""
    distribution: str
    distro_version: str
    kernel_version: str
    architecture: str
    hostname: str
    uptime: int
    load_average: tuple
    cpu_count: int
    memory_total: int
    memory_available: int
    disk_usage: dict
    network_interfaces: list


class LinuxSystemManager:
    """Advanced Linux system management"""

    def __init__(self):
        self.console = Console()
        self.distribution = distro.name()
        self.package_manager = self._detect_package_manager()
        self.init_system = self._detect_init_system()

        # D-Bus setup
        self.dbus_session = None
        self.dbus_system = None
        if DBUS_AVAILABLE:
            self._init_dbus()

    def _detect_package_manager(self) -> str:
        """Detect the system package manager"""
        managers = {
            'apt': '/usr/bin/apt',
            'dnf': '/usr/bin/dnf',
            'yum': '/usr/bin/yum',
            'pacman': '/usr/bin/pacman',
            'zypper': '/usr/bin/zypper',
            'emerge': '/usr/bin/emerge',
            'apk': '/sbin/apk'
        }

        for name, path in managers.items():
            if Path(path).exists():
                return name

        return 'unknown'

    def _detect_init_system(self) -> str:
        """Detect the init system"""
        if Path('/run/systemd/system').exists():
            return 'systemd'
        elif Path('/sbin/init').is_symlink():
            target = Path('/sbin/init').readlink()
            if 'upstart' in str(target):
                return 'upstart'
            elif 'systemd' in str(target):
                return 'systemd'

        return 'sysv'

    def _init_dbus(self):
        """Initialize D-Bus connections"""
        try:
            self.dbus_session = dbus.SessionBus()
            self.dbus_system = dbus.SystemBus()
        except Exception as e:
            self.console.print(f"[yellow]Warning: D-Bus not available: {e}[/yellow]")

    def get_system_info(self) -> LinuxSystemInfo:
        """Get comprehensive Linux system information"""
        try:
            # Get system information
            hostname = os.uname().nodename
            kernel_version = os.uname().release
            architecture = os.uname().machine

            # Get uptime
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().strip().split()[0])

            # Get load average
            load_avg = os.getloadavg()

            # Get memory information
            memory = psutil.virtual_memory()

            # Get CPU information
            cpu_count = psutil.cpu_count()

            # Get disk information
            disk_usage = self._get_disk_usage()

            # Get network interfaces
            network_interfaces = self._get_network_interfaces()

            return LinuxSystemInfo(
                distribution=self.distribution,
                distro_version=distro.version(),
                kernel_version=kernel_version,
                architecture=architecture,
                hostname=hostname,
                uptime=int(uptime_seconds),
                load_average=load_avg,
                cpu_count=cpu_count,
                memory_total=memory.total,
                memory_available=memory.available,
                disk_usage=disk_usage,
                network_interfaces=network_interfaces
            )

        except Exception as e:
            self.console.print(f"[red]Error getting system info: {e}[/red]")
            return None

    def _get_disk_usage(self) -> Dict[str, Dict]:
        """Get disk usage information"""
        disk_info = {}
        try:
            # Get mounted filesystems
            with open('/proc/mounts', 'r') as f:
                mounts = f.readlines()

            for mount in mounts:
                parts = mount.split()
                if len(parts) >= 3:
                    device, mountpoint, fstype = parts[0], parts[1], parts[2]

                    # Skip virtual filesystems
                    if fstype in ['proc', 'sysfs', 'devtmpfs', 'tmpfs', 'devpts', 'securityfs']:
                        continue

                    try:
                        usage = psutil.disk_usage(mountpoint)
                        disk_info[mountpoint] = {
                            'device': device,
                            'fstype': fstype,
                            'total': usage.total,
                            'used': usage.used,
                            'free': usage.free,
                            'percentage': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                        }
                    except (PermissionError, FileNotFoundError):
                        continue

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not get disk info: {e}[/yellow]")

        return disk_info

    def _get_network_interfaces(self) -> List[Dict]:
        """Get network interface information"""
        interfaces = []
        try:
            for interface, addrs in psutil.net_if_addrs().items():
                interface_info = {
                    'name': interface,
                    'addresses': [],
                    'stats': None
                }

                # Get addresses
                for addr in addrs:
                    interface_info['addresses'].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })

                # Get interface statistics
                try:
                    stats = psutil.net_if_stats()[interface]
                    interface_info['stats'] = {
                        'isup': stats.isup,
                        'duplex': str(stats.duplex),
                        'speed': stats.speed,
                        'mtu': stats.mtu
                    }
                except KeyError:
                    pass

                interfaces.append(interface_info)

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not get network info: {e}[/yellow]")

        return interfaces

    def get_systemd_services(self) -> List[LinuxService]:
        """Get systemd services information"""
        services = []

        if self.init_system != 'systemd':
            self.console.print("[yellow]Systemd not available[/yellow]")
            return services

        try:
            # Get list of services
            result = subprocess.run(
                ['systemctl', 'list-units', '--type=service', '--no-pager', '--no-legend'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            name = parts[0]
                            status = parts[2]
                            enabled = self._is_service_enabled(name)
                            description = ' '.join(parts[4:]) if len(parts) > 4 else ''

                            services.append(LinuxService(
                                name=name,
                                status=status,
                                enabled=enabled,
                                description=description,
                                unit_type='service'
                            ))

        except Exception as e:
            self.console.print(f"[red]Error getting systemd services: {e}[/red]")

        return services

    def _is_service_enabled(self, service_name: str) -> bool:
        """Check if a systemd service is enabled"""
        try:
            result = subprocess.run(
                ['systemctl', 'is-enabled', service_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and 'enabled' in result.stdout
        except Exception:
            return False

    def get_running_processes(self) -> List[LinuxProcess]:
        """Get list of running processes"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'ppid', 'name', 'username', 'cpu_percent', 'memory_percent', 'status', 'cmdline']):
                try:
                    proc_info = proc.info
                    command = ' '.join(proc_info.get('cmdline', [])) or proc_info['name']

                    processes.append(LinuxProcess(
                        pid=proc_info['pid'],
                        ppid=proc_info['ppid'],
                        name=proc_info['name'],
                        username=proc_info['username'],
                        cpu_percent=proc_info.get('cpu_percent', 0.0) or 0.0,
                        memory_percent=proc_info.get('memory_percent', 0.0) or 0.0,
                        status=proc_info['status'],
                        command=command[:100] + '...' if len(command) > 100 else command
                    ))

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        except Exception as e:
            self.console.print(f"[red]Error getting processes: {e}[/red]")

        return processes

    def manage_systemd_service(self, service_name: str, action: str) -> bool:
        """Manage systemd service"""
        if self.init_system != 'systemd':
            self.console.print("[red]Systemd not available[/red]")
            return False

        valid_actions = ['start', 'stop', 'restart', 'reload', 'enable', 'disable']
        if action not in valid_actions:
            self.console.print(f"[red]Invalid action. Valid actions: {', '.join(valid_actions)}[/red]")
            return False

        try:
            # Check if we need sudo
            need_sudo = os.geteuid() != 0

            command = ['systemctl', action, service_name]
            if need_sudo and action in ['start', 'stop', 'restart', 'reload', 'enable', 'disable']:
                command = ['sudo'] + command

            result = subprocess.run(command, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.console.print(f"[green]✅ Service {service_name} {action}ed successfully[/green]")
                return True
            else:
                self.console.print(f"[red]Error: {result.stderr}[/red]")
                return False

        except Exception as e:
            self.console.print(f"[red]Error managing service {service_name}: {e}[/red]")
            return False

    def execute_shell_command(self, command: str, shell: str = "/bin/bash", timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute shell command"""
        try:
            if shell not in ["/bin/bash", "/bin/zsh", "/bin/fish", "/bin/sh"]:
                shell = "/bin/bash"

            result = subprocess.run(
                [shell, '-c', command],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy()
            )

            return result.returncode == 0, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def get_environment_variables(self) -> Dict[str, str]:
        """Get all environment variables"""
        return dict(os.environ)

    def set_environment_variable(self, name: str, value: str, permanent: bool = False) -> bool:
        """Set environment variable"""
        try:
            os.environ[name] = value

            if permanent:
                # Add to shell profile (basic implementation)
                shell_profile = Path.home() / '.profile'
                with open(shell_profile, 'a') as f:
                    f.write(f'\nexport {name}="{value}"\n')

            return True
        except Exception as e:
            self.console.print(f"[red]Error setting environment variable: {e}[/red]")
            return False

    def get_installed_packages(self) -> List[Dict[str, str]]:
        """Get list of installed packages"""
        packages = []

        if self.package_manager == 'apt':
            packages = self._get_apt_packages()
        elif self.package_manager == 'dnf':
            packages = self._get_dnf_packages()
        elif self.package_manager == 'pacman':
            packages = self._get_pacman_packages()
        # Add more package managers as needed

        return packages

    def _get_apt_packages(self) -> List[Dict[str, str]]:
        """Get installed packages via apt"""
        packages = []
        try:
            result = subprocess.run(
                ['dpkg-query', '-W', '-f=${Package}\\t${Version}\\t${Architecture}\\n'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            packages.append({
                                'name': parts[0],
                                'version': parts[1],
                                'architecture': parts[2] if len(parts) > 2 else 'unknown',
                                'manager': 'apt'
                            })

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not get APT packages: {e}[/yellow]")

        return packages

    def _get_dnf_packages(self) -> List[Dict[str, str]]:
        """Get installed packages via DNF"""
        packages = []
        try:
            result = subprocess.run(
                ['rpm', '-qa', '--queryformat', '%{NAME}\\t%{VERSION}\\t%{ARCH}\\n'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            packages.append({
                                'name': parts[0],
                                'version': parts[1],
                                'architecture': parts[2] if len(parts) > 2 else 'unknown',
                                'manager': 'dnf'
                            })

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not get DNF packages: {e}[/yellow]")

        return packages

    def _get_pacman_packages(self) -> List[Dict[str, str]]:
        """Get installed packages via pacman"""
        packages = []
        try:
            result = subprocess.run(
                ['pacman', '-Q'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            packages.append({
                                'name': parts[0],
                                'version': parts[1],
                                'architecture': 'x86_64',  # Default assumption
                                'manager': 'pacman'
                            })

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not get Pacman packages: {e}[/yellow]")

        return packages

    def monitor_system_performance(self, duration: int = 60) -> Dict[str, Any]:
        """Monitor system performance for specified duration"""
        performance_data = {
            'cpu_usage': [],
            'memory_usage': [],
            'load_average': [],
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

                # Load average
                load_avg = os.getloadavg()
                performance_data['load_average'].append(load_avg)

                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    performance_data['disk_io'].append({
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes,
                        'read_count': disk_io.read_count,
                        'write_count': disk_io.write_count
                    })

                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    performance_data['network_io'].append({
                        'bytes_sent': network_io.bytes_sent,
                        'bytes_recv': network_io.bytes_recv,
                        'packets_sent': network_io.packets_sent,
                        'packets_recv': network_io.packets_recv
                    })

                performance_data['timestamp'].append(datetime.now())

        except Exception as e:
            self.console.print(f"[red]Error monitoring performance: {e}[/red]")

        return performance_data

    def get_kernel_modules(self) -> List[Dict[str, str]]:
        """Get loaded kernel modules"""
        modules = []
        try:
            with open('/proc/modules', 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        modules.append({
                            'name': parts[0],
                            'size': parts[1],
                            'used_count': parts[2],
                            'dependencies': parts[3] if len(parts) > 3 else ''
                        })
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not get kernel modules: {e}[/yellow]")

        return modules

    def get_mount_points(self) -> List[Dict[str, str]]:
        """Get filesystem mount points"""
        mounts = []
        try:
            with open('/proc/mounts', 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        mounts.append({
                            'device': parts[0],
                            'mountpoint': parts[1],
                            'fstype': parts[2],
                            'options': parts[3],
                            'dump': parts[4],
                            'pass': parts[5]
                        })
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not get mount points: {e}[/yellow]")

        return mounts

    def display_system_info(self):
        """Display comprehensive system information"""
        system_info = self.get_system_info()

        if not system_info:
            self.console.print("[red]Could not retrieve system information[/red]")
            return

        # System Information Table
        table = Table(title="🐧 Linux System Information", style="cyan")
        table.add_column("Property", style="magenta", width=20)
        table.add_column("Value", style="white")

        table.add_row("Distribution", f"{system_info.distribution} {system_info.distro_version}")
        table.add_row("Kernel", system_info.kernel_version)
        table.add_row("Architecture", system_info.architecture)
        table.add_row("Hostname", system_info.hostname)
        table.add_row("Uptime", f"{system_info.uptime // 3600}h {(system_info.uptime % 3600) // 60}m")
        table.add_row("Load Average", f"{system_info.load_average[0]:.2f}, {system_info.load_average[1]:.2f}, {system_info.load_average[2]:.2f}")
        table.add_row("CPU Cores", str(system_info.cpu_count))
        table.add_row("Total Memory", f"{system_info.memory_total / (1024**3):.2f} GB")
        table.add_row("Available Memory", f"{system_info.memory_available / (1024**3):.2f} GB")
        table.add_row("Package Manager", self.package_manager)
        table.add_row("Init System", self.init_system)

        self.console.print(table)

        # Disk Usage Table
        if system_info.disk_usage:
            disk_table = Table(title="💾 Filesystem Usage", style="green")
            disk_table.add_column("Mount Point", style="cyan")
            disk_table.add_column("Device", style="white")
            disk_table.add_column("Type", style="yellow")
            disk_table.add_column("Size", style="white")
            disk_table.add_column("Used", style="red")
            disk_table.add_column("Available", style="green")
            disk_table.add_column("Use%", style="magenta")

            for mountpoint, info in system_info.disk_usage.items():
                if mountpoint in ['/', '/home', '/var', '/tmp'] or not mountpoint.startswith('/'):
                    total_gb = info['total'] / (1024**3)
                    used_gb = info['used'] / (1024**3)
                    free_gb = info['free'] / (1024**3)

                    disk_table.add_row(
                        mountpoint,
                        info['device'],
                        info['fstype'],
                        f"{total_gb:.1f}G",
                        f"{used_gb:.1f}G",
                        f"{free_gb:.1f}G",
                        f"{info['percentage']:.1f}%"
                    )

            self.console.print(disk_table)

    def display_services_info(self, running_only: bool = True):
        """Display systemd services information"""
        services = self.get_systemd_services()

        if running_only:
            services = [s for s in services if s.status == 'running']

        table = Table(title=f"⚙️ Systemd Services ({'Running Only' if running_only else 'All'})", style="blue")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Enabled", style="yellow")
        table.add_column("Description", style="white")

        for service in services[:50]:  # Show first 50
            enabled_text = "✅" if service.enabled else "❌"
            table.add_row(
                service.name,
                service.status,
                enabled_text,
                service.description[:60] + "..." if len(service.description) > 60 else service.description
            )

        self.console.print(table)

    def display_processes_info(self, sort_by: str = "cpu", top_n: int = 20):
        """Display running processes information"""
        processes = self.get_running_processes()

        # Sort processes
        if sort_by.lower() == "cpu":
            processes.sort(key=lambda x: x.cpu_percent, reverse=True)
        elif sort_by.lower() == "memory":
            processes.sort(key=lambda x: x.memory_percent, reverse=True)

        table = Table(title=f"🔄 Top {top_n} Processes (by {sort_by.upper()})", style="magenta")
        table.add_column("PID", style="cyan")
        table.add_column("User", style="green")
        table.add_column("Name", style="white")
        table.add_column("CPU %", style="red")
        table.add_column("Memory %", style="yellow")
        table.add_column("Status", style="blue")

        for process in processes[:top_n]:
            table.add_row(
                str(process.pid),
                process.username,
                process.name[:20] + "..." if len(process.name) > 20 else process.name,
                f"{process.cpu_percent:.1f}%",
                f"{process.memory_percent:.1f}%",
                process.status
            )

        self.console.print(table)

    def display_network_info(self):
        """Display network interface information"""
        system_info = self.get_system_info()

        if not system_info or not system_info.network_interfaces:
            self.console.print("[yellow]No network information available[/yellow]")
            return

        for interface_info in system_info.network_interfaces:
            if interface_info['name'] in ['lo', 'docker0']:
                continue  # Skip loopback and docker interfaces

            table = Table(title=f"🌐 Network Interface: {interface_info['name']}", style="blue")
            table.add_column("Property", style="magenta")
            table.add_column("Value", style="white")

            # Interface statistics
            if interface_info['stats']:
                stats = interface_info['stats']
                table.add_row("Status", "UP" if stats['isup'] else "DOWN")
                table.add_row("Speed", f"{stats['speed']} Mbps" if stats['speed'] > 0 else "Unknown")
                table.add_row("MTU", str(stats['mtu']))

            # Addresses
            for addr in interface_info['addresses']:
                if addr['family'] == "AddressFamily.AF_INET":
                    table.add_row("IPv4 Address", addr['address'])
                    if addr['netmask']:
                        table.add_row("Netmask", addr['netmask'])
                elif addr['family'] == "AddressFamily.AF_INET6":
                    table.add_row("IPv6 Address", addr['address'][:40] + "..." if len(addr['address']) > 40 else addr['address'])

            self.console.print(table)

    def send_desktop_notification(self, title: str, message: str, urgency: str = "normal") -> bool:
        """Send desktop notification via D-Bus"""
        if not DBUS_AVAILABLE or not self.dbus_session:
            return False

        try:
            notify_service = self.dbus_session.get_object(
                'org.freedesktop.Notifications',
                '/org/freedesktop/Notifications'
            )

            notify_interface = dbus.Interface(notify_service, 'org.freedesktop.Notifications')

            urgency_levels = {"low": 0, "normal": 1, "critical": 2}
            urgency_level = urgency_levels.get(urgency.lower(), 1)

            notify_interface.Notify(
                "Terminal Coder",  # app_name
                0,                 # replaces_id
                "",                # app_icon
                title,             # summary
                message,           # body
                [],                # actions
                {"urgency": urgency_level},  # hints
                5000               # timeout (5 seconds)
            )

            return True

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not send notification: {e}[/yellow]")
            return False