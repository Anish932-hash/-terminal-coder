"""
Ultra-Advanced Linux System Manager
Enterprise-grade Linux system integration with advanced monitoring, optimization, and security features
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator, Final
import psutil
import distro
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.live import Live
from datetime import datetime, timedelta
import threading
import signal

try:
    import dbus
    import pyinotify
    import systemd.daemon
    import systemd.journal
    ADVANCED_LINUX_AVAILABLE = True
except ImportError:
    ADVANCED_LINUX_AVAILABLE = False


class SystemOptimizationLevel(Enum):
    """System optimization levels"""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXTREME = "extreme"
    CUSTOM = "custom"


class ServiceType(Enum):
    """Linux service types"""
    SYSTEMD = "systemd"
    INIT_D = "init.d"
    UPSTART = "upstart"
    RUNIT = "runit"
    OPENRC = "openrc"


class SecurityProfile(Enum):
    """Security profiles"""
    DEVELOPER = "developer"
    PRODUCTION = "production"
    HIGH_SECURITY = "high_security"
    CUSTOM = "custom"


@dataclass(slots=True)
class SystemMetrics:
    """Real-time system metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage: dict[str, float]
    network_io: dict[str, int]
    process_count: int
    load_average: tuple[float, float, float]
    uptime: float
    temperature: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(slots=True)
class ContainerInfo:
    """Container system information"""
    docker_running: bool = False
    podman_running: bool = False
    containerd_running: bool = False
    active_containers: int = 0
    images_count: int = 0
    volumes_count: int = 0


@dataclass(slots=True)
class KubernetesInfo:
    """Kubernetes cluster information"""
    kubectl_available: bool = False
    current_context: str | None = None
    nodes: list[dict] = field(default_factory=list)
    pods: list[dict] = field(default_factory=list)
    services: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class SecurityStatus:
    """System security status"""
    firewall_active: bool = False
    selinux_status: str | None = None
    apparmor_status: str | None = None
    fail2ban_active: bool = False
    ssh_key_auth_only: bool = False
    automatic_updates: bool = False
    last_security_update: datetime | None = None


class UltraLinuxManager:
    """Ultra-advanced Linux system manager with enterprise features"""

    def __init__(self) -> None:
        self.console = Console()
        self._verify_linux_system()

        # System information cache
        self._system_info_cache: dict[str, Any] = {}
        self._cache_timestamp = datetime.min
        self._cache_ttl = timedelta(seconds=30)

        # Monitoring
        self._monitoring_active = False
        self._monitoring_thread: threading.Thread | None = None
        self._metrics_history: list[SystemMetrics] = []
        self._max_history_size = 1000

        # File system monitoring
        self._fs_watcher: pyinotify.WatchManager | None = None
        self._fs_notifier: pyinotify.Notifier | None = None

        # Service management
        self._init_service_manager()

        # Security monitoring
        self._security_events: list[dict[str, Any]] = []

    def _verify_linux_system(self) -> None:
        """Verify we're running on a supported Linux system"""
        if not sys.platform.startswith('linux'):
            raise RuntimeError("Ultra Linux Manager requires a Linux system")

        if not ADVANCED_LINUX_AVAILABLE:
            self.console.print("[yellow]⚠️  Some advanced features require additional packages:[/yellow]")
            self.console.print("   sudo apt install python3-systemd python3-dbus python3-pyinotify")

    def _init_service_manager(self) -> None:
        """Initialize service management system"""
        self.service_type = ServiceType.SYSTEMD

        # Detect service manager
        if Path("/run/systemd/system").exists():
            self.service_type = ServiceType.SYSTEMD
        elif Path("/etc/init.d").exists():
            self.service_type = ServiceType.INIT_D
        elif Path("/etc/init").exists():
            self.service_type = ServiceType.UPSTART

    async def get_comprehensive_system_info(self) -> dict[str, Any]:
        """Get comprehensive system information with caching"""
        now = datetime.now()
        if now - self._cache_timestamp < self._cache_ttl and self._system_info_cache:
            return self._system_info_cache

        info = {}

        # Basic system info
        info["distribution"] = {
            "name": distro.name(),
            "version": distro.version(),
            "codename": distro.codename(),
            "id": distro.id(),
            "like": distro.like(),
        }

        info["kernel"] = {
            "release": os.uname().release,
            "version": os.uname().version,
            "machine": os.uname().machine,
        }

        # Hardware information
        info["hardware"] = await self._get_hardware_info()

        # Container systems
        info["containers"] = await self._get_container_info()

        # Kubernetes
        info["kubernetes"] = await self._get_kubernetes_info()

        # Security status
        info["security"] = await self._get_security_status()

        # Development environment
        info["development"] = await self._get_development_info()

        # System services
        info["services"] = await self._get_critical_services_status()

        # Network configuration
        info["network"] = await self._get_network_info()

        # Performance metrics
        info["performance"] = await self._get_performance_metrics()

        self._system_info_cache = info
        self._cache_timestamp = now
        return info

    async def _get_hardware_info(self) -> dict[str, Any]:
        """Get detailed hardware information"""
        return {
            "cpu": {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "current_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "max_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "architecture": os.uname().machine,
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "swap_total": psutil.swap_memory().total,
                "swap_used": psutil.swap_memory().used,
            },
            "disk": {
                partition.mountpoint: {
                    "total": psutil.disk_usage(partition.mountpoint).total,
                    "used": psutil.disk_usage(partition.mountpoint).used,
                    "free": psutil.disk_usage(partition.mountpoint).free,
                    "filesystem": partition.fstype,
                }
                for partition in psutil.disk_partitions()
                if partition.mountpoint.startswith('/')
            },
            "temperature": await self._get_temperature_info(),
        }

    async def _get_temperature_info(self) -> dict[str, float]:
        """Get system temperature information"""
        temps = {}
        try:
            if hasattr(psutil, "sensors_temperatures"):
                sensors = psutil.sensors_temperatures()
                for name, entries in sensors.items():
                    for entry in entries:
                        key = f"{name}_{entry.label}" if entry.label else name
                        temps[key] = entry.current
        except Exception:
            pass

        # Try thermal zones
        try:
            thermal_dir = Path("/sys/class/thermal")
            if thermal_dir.exists():
                for zone_dir in thermal_dir.glob("thermal_zone*"):
                    temp_file = zone_dir / "temp"
                    if temp_file.exists():
                        temp = float(temp_file.read_text().strip()) / 1000
                        temps[zone_dir.name] = temp
        except Exception:
            pass

        return temps

    async def _get_container_info(self) -> ContainerInfo:
        """Get container system information"""
        info = ContainerInfo()

        # Docker
        try:
            result = subprocess.run(
                ["docker", "ps", "-q"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info.docker_running = True
                info.active_containers += len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

                # Get images count
                result = subprocess.run(
                    ["docker", "images", "-q"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.images_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        except Exception:
            pass

        # Podman
        try:
            result = subprocess.run(
                ["podman", "ps", "-q"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info.podman_running = True
                info.active_containers += len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        except Exception:
            pass

        # Containerd
        try:
            result = subprocess.run(
                ["ctr", "containers", "list"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info.containerd_running = True
        except Exception:
            pass

        return info

    async def _get_kubernetes_info(self) -> KubernetesInfo:
        """Get Kubernetes cluster information"""
        info = KubernetesInfo()

        try:
            # Check kubectl availability
            result = subprocess.run(
                ["kubectl", "version", "--client"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info.kubectl_available = True

                # Get current context
                result = subprocess.run(
                    ["kubectl", "config", "current-context"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info.current_context = result.stdout.strip()

                # Get nodes
                result = subprocess.run(
                    ["kubectl", "get", "nodes", "-o", "json"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    nodes_data = json.loads(result.stdout)
                    info.nodes = nodes_data.get("items", [])

                # Get pods
                result = subprocess.run(
                    ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    pods_data = json.loads(result.stdout)
                    info.pods = pods_data.get("items", [])

        except Exception:
            pass

        return info

    async def _get_security_status(self) -> SecurityStatus:
        """Get comprehensive security status"""
        status = SecurityStatus()

        # Firewall status
        try:
            # UFW
            result = subprocess.run(
                ["ufw", "status"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "Status: active" in result.stdout:
                status.firewall_active = True

            # iptables
            if not status.firewall_active:
                result = subprocess.run(
                    ["iptables", "-L"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and len(result.stdout.split('\n')) > 10:
                    status.firewall_active = True
        except Exception:
            pass

        # SELinux
        try:
            selinux_file = Path("/sys/fs/selinux/enforce")
            if selinux_file.exists():
                enforce = selinux_file.read_text().strip()
                status.selinux_status = "Enforcing" if enforce == "1" else "Permissive"
        except Exception:
            pass

        # AppArmor
        try:
            result = subprocess.run(
                ["aa-status"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                status.apparmor_status = "Active"
        except Exception:
            pass

        # Fail2Ban
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "fail2ban"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "active" in result.stdout:
                status.fail2ban_active = True
        except Exception:
            pass

        # SSH configuration
        try:
            ssh_config = Path("/etc/ssh/sshd_config")
            if ssh_config.exists():
                config_text = ssh_config.read_text()
                if "PasswordAuthentication no" in config_text:
                    status.ssh_key_auth_only = True
        except Exception:
            pass

        return status

    async def _get_development_info(self) -> dict[str, Any]:
        """Get development environment information"""
        info = {
            "languages": {},
            "package_managers": {},
            "version_control": {},
            "editors": {},
            "databases": {},
        }

        # Programming languages
        languages = {
            "python": ["python3", "--version"],
            "python2": ["python2", "--version"],
            "node": ["node", "--version"],
            "go": ["go", "version"],
            "rust": ["rustc", "--version"],
            "java": ["java", "--version"],
            "gcc": ["gcc", "--version"],
            "clang": ["clang", "--version"],
            "php": ["php", "--version"],
            "ruby": ["ruby", "--version"],
        }

        for lang, cmd in languages.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info["languages"][lang] = result.stdout.split('\n')[0]
            except Exception:
                pass

        # Package managers
        package_managers = {
            "npm": ["npm", "--version"],
            "yarn": ["yarn", "--version"],
            "pip": ["pip3", "--version"],
            "cargo": ["cargo", "--version"],
            "composer": ["composer", "--version"],
            "gem": ["gem", "--version"],
        }

        for pm, cmd in package_managers.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info["package_managers"][pm] = result.stdout.strip()
            except Exception:
                pass

        # Version control
        vcs = {
            "git": ["git", "--version"],
            "svn": ["svn", "--version"],
            "hg": ["hg", "--version"],
        }

        for vc, cmd in vcs.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info["version_control"][vc] = result.stdout.split('\n')[0]
            except Exception:
                pass

        return info

    async def _get_critical_services_status(self) -> dict[str, str]:
        """Get status of critical system services"""
        services = {}
        critical_services = [
            "ssh", "sshd", "docker", "nginx", "apache2", "mysql",
            "postgresql", "redis", "mongodb", "elasticsearch"
        ]

        for service in critical_services:
            try:
                if self.service_type == ServiceType.SYSTEMD:
                    result = subprocess.run(
                        ["systemctl", "is-active", service],
                        capture_output=True, text=True, timeout=3
                    )
                    services[service] = result.stdout.strip()
                else:
                    result = subprocess.run(
                        [f"/etc/init.d/{service}", "status"],
                        capture_output=True, text=True, timeout=3
                    )
                    services[service] = "active" if result.returncode == 0 else "inactive"
            except Exception:
                services[service] = "unknown"

        return services

    async def _get_network_info(self) -> dict[str, Any]:
        """Get network configuration information"""
        info = {
            "interfaces": {},
            "routing": {},
            "dns": [],
            "firewall_rules": [],
        }

        # Network interfaces
        for interface, addrs in psutil.net_if_addrs().items():
            info["interfaces"][interface] = []
            for addr in addrs:
                if addr.family.name in ['AF_INET', 'AF_INET6']:
                    info["interfaces"][interface].append({
                        "family": addr.family.name,
                        "address": addr.address,
                        "netmask": addr.netmask,
                    })

        # DNS servers
        try:
            resolv_conf = Path("/etc/resolv.conf")
            if resolv_conf.exists():
                for line in resolv_conf.read_text().split('\n'):
                    if line.startswith("nameserver"):
                        info["dns"].append(line.split()[1])
        except Exception:
            pass

        return info

    async def _get_performance_metrics(self) -> SystemMetrics:
        """Get current performance metrics"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage={
                partition.mountpoint: psutil.disk_usage(partition.mountpoint).percent
                for partition in psutil.disk_partitions()
                if partition.mountpoint.startswith('/')
            },
            network_io={
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
            },
            process_count=len(psutil.pids()),
            load_average=os.getloadavg(),
            uptime=time.time() - psutil.boot_time(),
            temperature=await self._get_temperature_info(),
        )

    def display_ultra_system_dashboard(self) -> None:
        """Display comprehensive system dashboard"""
        async def update_dashboard():
            info = await self.get_comprehensive_system_info()

            # Create main layout
            layout = Table.grid(padding=1)
            layout.add_column(justify="left")
            layout.add_column(justify="left")

            # System Overview Panel
            system_panel = self._create_system_overview_panel(info)

            # Performance Panel
            performance_panel = await self._create_performance_panel(info)

            # Container Panel
            container_panel = self._create_container_panel(info["containers"])

            # Security Panel
            security_panel = self._create_security_panel(info["security"])

            layout.add_row(system_panel, performance_panel)
            layout.add_row(container_panel, security_panel)

            return Panel(
                layout,
                title="🐧 Ultra Linux System Dashboard",
                border_style="cyan",
                padding=(1, 2)
            )

        with Live(console=self.console, refresh_per_second=2) as live:
            try:
                while True:
                    dashboard = asyncio.run(update_dashboard())
                    live.update(dashboard)
                    time.sleep(2)
            except KeyboardInterrupt:
                pass

    def _create_system_overview_panel(self, info: dict[str, Any]) -> Panel:
        """Create system overview panel"""
        content = []

        # Distribution info
        distro_info = info["distribution"]
        content.append(f"[bold cyan]OS:[/bold cyan] {distro_info['name']} {distro_info['version']}")
        content.append(f"[bold cyan]Kernel:[/bold cyan] {info['kernel']['release']}")
        content.append(f"[bold cyan]Architecture:[/bold cyan] {info['kernel']['machine']}")

        # Hardware summary
        hw = info["hardware"]
        content.append(f"[bold cyan]CPU:[/bold cyan] {hw['cpu']['logical_cores']} cores")
        content.append(f"[bold cyan]Memory:[/bold cyan] {hw['memory']['total'] // (1024**3)} GB")

        return Panel(
            "\n".join(content),
            title="💻 System Overview",
            border_style="green"
        )

    async def _create_performance_panel(self, info: dict[str, Any]) -> Panel:
        """Create performance monitoring panel"""
        metrics = info["performance"]

        content = []
        content.append(f"[bold yellow]CPU Usage:[/bold yellow] {metrics.cpu_percent:.1f}%")
        content.append(f"[bold yellow]Memory Usage:[/bold yellow] {metrics.memory_percent:.1f}%")
        content.append(f"[bold yellow]Load Average:[/bold yellow] {metrics.load_average[0]:.2f}")
        content.append(f"[bold yellow]Processes:[/bold yellow] {metrics.process_count}")

        # Temperature info
        if metrics.temperature:
            avg_temp = sum(metrics.temperature.values()) / len(metrics.temperature)
            content.append(f"[bold yellow]Temperature:[/bold yellow] {avg_temp:.1f}°C")

        return Panel(
            "\n".join(content),
            title="📊 Performance",
            border_style="yellow"
        )

    def _create_container_panel(self, container_info: ContainerInfo) -> Panel:
        """Create container systems panel"""
        content = []

        if container_info.docker_running:
            content.append(f"[bold blue]Docker:[/bold blue] ✅ Running")
            content.append(f"  Active containers: {container_info.active_containers}")
            content.append(f"  Images: {container_info.images_count}")
        else:
            content.append("[bold blue]Docker:[/bold blue] ❌ Not running")

        if container_info.podman_running:
            content.append("[bold blue]Podman:[/bold blue] ✅ Running")
        else:
            content.append("[bold blue]Podman:[/bold blue] ❌ Not running")

        if not content:
            content.append("[dim]No container systems detected[/dim]")

        return Panel(
            "\n".join(content),
            title="🐳 Containers",
            border_style="blue"
        )

    def _create_security_panel(self, security_info: SecurityStatus) -> Panel:
        """Create security status panel"""
        content = []

        # Firewall
        status = "✅ Active" if security_info.firewall_active else "❌ Inactive"
        content.append(f"[bold red]Firewall:[/bold red] {status}")

        # SELinux
        if security_info.selinux_status:
            content.append(f"[bold red]SELinux:[/bold red] {security_info.selinux_status}")

        # AppArmor
        if security_info.apparmor_status:
            content.append(f"[bold red]AppArmor:[/bold red] {security_info.apparmor_status}")

        # Fail2Ban
        status = "✅ Active" if security_info.fail2ban_active else "❌ Inactive"
        content.append(f"[bold red]Fail2Ban:[/bold red] {status}")

        # SSH
        if security_info.ssh_key_auth_only:
            content.append("[bold red]SSH:[/bold red] ✅ Key-only auth")

        return Panel(
            "\n".join(content),
            title="🔒 Security",
            border_style="red"
        )

    async def optimize_system_ultra(self, level: SystemOptimizationLevel = SystemOptimizationLevel.ADVANCED) -> bool:
        """Ultra-advanced system optimization"""
        self.console.print(f"[bold cyan]🚀 Starting {level.value.upper()} system optimization...[/bold cyan]")

        optimizations = []

        if level in [SystemOptimizationLevel.BASIC, SystemOptimizationLevel.ADVANCED, SystemOptimizationLevel.EXTREME]:
            optimizations.extend([
                ("Updating package cache", self._update_package_cache),
                ("Optimizing kernel parameters", lambda: self._optimize_kernel_params(level)),
                ("Configuring system limits", lambda: self._configure_system_limits(level)),
                ("Optimizing I/O scheduler", self._optimize_io_scheduler),
                ("Configuring CPU governor", self._optimize_cpu_governor),
            ])

        if level in [SystemOptimizationLevel.ADVANCED, SystemOptimizationLevel.EXTREME]:
            optimizations.extend([
                ("Optimizing network stack", self._optimize_network_stack),
                ("Configuring memory management", lambda: self._optimize_memory_management(level)),
                ("Setting up development tools", self._setup_development_optimizations),
                ("Configuring container optimizations", self._optimize_containers),
            ])

        if level == SystemOptimizationLevel.EXTREME:
            optimizations.extend([
                ("Applying extreme kernel tweaks", self._apply_extreme_optimizations),
                ("Configuring real-time priorities", self._configure_realtime),
                ("Setting up performance monitoring", self._setup_performance_monitoring),
            ])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:

            total_task = progress.add_task("Overall Progress", total=len(optimizations))

            for description, optimization_func in optimizations:
                task = progress.add_task(description, total=100)

                try:
                    result = await optimization_func() if asyncio.iscoroutinefunction(optimization_func) else optimization_func()
                    progress.update(task, completed=100)

                    if result:
                        self.console.print(f"[green]✅ {description}[/green]")
                    else:
                        self.console.print(f"[yellow]⚠️  {description} - Partial success[/yellow]")

                except Exception as e:
                    self.console.print(f"[red]❌ {description} - Failed: {str(e)[:50]}[/red]")

                progress.update(total_task, advance=1)

        self.console.print("[bold green]🎉 System optimization completed![/bold green]")
        self.console.print("[yellow]💡 Reboot recommended for all changes to take effect[/yellow]")
        return True

    def _update_package_cache(self) -> bool:
        """Update system package cache"""
        try:
            distro_id = distro.id()

            if distro_id in ['ubuntu', 'debian']:
                subprocess.run(['sudo', 'apt', 'update'], check=True, capture_output=True)
            elif distro_id in ['fedora', 'centos', 'rhel']:
                subprocess.run(['sudo', 'dnf', 'check-update'], capture_output=True)
            elif distro_id in ['arch', 'manjaro']:
                subprocess.run(['sudo', 'pacman', '-Sy'], check=True, capture_output=True)

            return True
        except Exception:
            return False

    def _optimize_kernel_params(self, level: SystemOptimizationLevel) -> bool:
        """Optimize kernel parameters based on optimization level"""
        try:
            sysctl_conf = Path("/etc/sysctl.d/99-terminal-coder.conf")

            params = [
                "# Terminal Coder Linux Optimizations",
                "",
                "# File system optimizations",
                "fs.file-max = 2097152",
                "fs.inotify.max_user_watches = 1048576",
                "fs.inotify.max_user_instances = 1024",
                "",
                "# Network optimizations",
                "net.core.rmem_default = 262144",
                "net.core.wmem_default = 262144",
                "net.core.rmem_max = 16777216",
                "net.core.wmem_max = 16777216",
                "net.ipv4.tcp_rmem = 4096 87380 16777216",
                "net.ipv4.tcp_wmem = 4096 65536 16777216",
                "",
                "# Virtual memory optimizations",
                "vm.swappiness = 10",
                "vm.vfs_cache_pressure = 50",
                "vm.dirty_ratio = 15",
                "vm.dirty_background_ratio = 5",
            ]

            if level in [SystemOptimizationLevel.ADVANCED, SystemOptimizationLevel.EXTREME]:
                params.extend([
                    "",
                    "# Advanced optimizations",
                    "kernel.sched_migration_cost_ns = 5000000",
                    "kernel.sched_autogroup_enabled = 0",
                    "net.ipv4.tcp_congestion_control = bbr",
                    "net.core.default_qdisc = fq",
                ])

            if level == SystemOptimizationLevel.EXTREME:
                params.extend([
                    "",
                    "# Extreme optimizations (use with caution)",
                    "vm.dirty_expire_centisecs = 500",
                    "vm.dirty_writeback_centisecs = 100",
                    "kernel.sched_rt_runtime_us = 950000",
                ])

            # Write configuration
            sysctl_conf.write_text('\n'.join(params))

            # Apply immediately
            subprocess.run(['sudo', 'sysctl', '-p', str(sysctl_conf)],
                         check=True, capture_output=True)

            return True
        except Exception:
            return False

    def _configure_system_limits(self, level: SystemOptimizationLevel) -> bool:
        """Configure system resource limits"""
        try:
            limits_conf = Path("/etc/security/limits.d/99-terminal-coder.conf")

            limits = [
                "# Terminal Coder Linux Resource Limits",
                "",
                "* soft nofile 65536",
                "* hard nofile 65536",
                "* soft nproc 32768",
                "* hard nproc 32768",
            ]

            if level in [SystemOptimizationLevel.ADVANCED, SystemOptimizationLevel.EXTREME]:
                limits.extend([
                    "* soft memlock unlimited",
                    "* hard memlock unlimited",
                    "* soft nice -10",
                    "* hard nice -10",
                ])

            limits_conf.write_text('\n'.join(limits))
            return True
        except Exception:
            return False

    def _optimize_io_scheduler(self) -> bool:
        """Optimize I/O scheduler for development workloads"""
        try:
            # Get all block devices
            for device in Path("/sys/block").glob("sd*"):
                scheduler_file = device / "queue" / "scheduler"
                if scheduler_file.exists():
                    # Set to 'deadline' for SSDs, 'cfq' for HDDs
                    try:
                        subprocess.run([
                            'sudo', 'bash', '-c',
                            f'echo deadline > {scheduler_file}'
                        ], check=True, capture_output=True)
                    except Exception:
                        pass
            return True
        except Exception:
            return False

    def _optimize_cpu_governor(self) -> bool:
        """Set CPU governor to performance for development"""
        try:
            subprocess.run([
                'sudo', 'cpupower', 'frequency-set', '-g', 'performance'
            ], capture_output=True)
            return True
        except Exception:
            return False

    def _optimize_network_stack(self) -> bool:
        """Optimize network stack for development"""
        # Already handled in kernel params
        return True

    def _optimize_memory_management(self, level: SystemOptimizationLevel) -> bool:
        """Optimize memory management"""
        # Already handled in kernel params
        return True

    def _setup_development_optimizations(self) -> bool:
        """Setup development-specific optimizations"""
        try:
            # Install development packages
            distro_id = distro.id()

            packages = {
                'ubuntu': ['build-essential', 'git', 'curl', 'wget', 'htop', 'tree', 'jq'],
                'debian': ['build-essential', 'git', 'curl', 'wget', 'htop', 'tree', 'jq'],
                'fedora': ['@development-tools', 'git', 'curl', 'wget', 'htop', 'tree', 'jq'],
                'arch': ['base-devel', 'git', 'curl', 'wget', 'htop', 'tree', 'jq'],
            }

            if distro_id in packages:
                if distro_id in ['ubuntu', 'debian']:
                    subprocess.run(['sudo', 'apt', 'install', '-y'] + packages[distro_id],
                                 capture_output=True, timeout=300)
                elif distro_id == 'fedora':
                    subprocess.run(['sudo', 'dnf', 'install', '-y'] + packages[distro_id],
                                 capture_output=True, timeout=300)
                elif distro_id == 'arch':
                    subprocess.run(['sudo', 'pacman', '-S', '--noconfirm'] + packages[distro_id],
                                 capture_output=True, timeout=300)

            return True
        except Exception:
            return False

    def _optimize_containers(self) -> bool:
        """Optimize container systems"""
        try:
            # Docker optimizations
            docker_daemon_config = Path("/etc/docker/daemon.json")
            if docker_daemon_config.parent.exists():
                config = {
                    "log-driver": "json-file",
                    "log-opts": {
                        "max-size": "10m",
                        "max-file": "3"
                    },
                    "storage-driver": "overlay2",
                    "storage-opts": ["overlay2.override_kernel_check=true"]
                }

                docker_daemon_config.write_text(json.dumps(config, indent=2))

                # Restart docker service
                subprocess.run(['sudo', 'systemctl', 'restart', 'docker'],
                             capture_output=True)

            return True
        except Exception:
            return False

    def _apply_extreme_optimizations(self) -> bool:
        """Apply extreme optimization settings"""
        self.console.print("[yellow]⚠️  Applying extreme optimizations - use with caution![/yellow]")
        # Extreme optimizations are already included in kernel params
        return True

    def _configure_realtime(self) -> bool:
        """Configure real-time scheduling priorities"""
        try:
            # Add user to realtime group
            subprocess.run(['sudo', 'usermod', '-a', '-G', 'realtime', os.getenv('USER')],
                         capture_output=True)
            return True
        except Exception:
            return False

    def _setup_performance_monitoring(self) -> bool:
        """Setup performance monitoring tools"""
        try:
            # Start monitoring if not already running
            if not self._monitoring_active:
                self.start_monitoring()
            return True
        except Exception:
            return False

    def start_monitoring(self) -> None:
        """Start real-time system monitoring"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)

    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                metrics = asyncio.run(self._get_performance_metrics())
                self._metrics_history.append(metrics)

                # Trim history
                if len(self._metrics_history) > self._max_history_size:
                    self._metrics_history = self._metrics_history[-self._max_history_size:]

                # Check for alerts
                self._check_performance_alerts(metrics)

                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                if self._monitoring_active:  # Only log if we're still supposed to be monitoring
                    self.console.print(f"[red]Monitoring error: {e}[/red]")
                time.sleep(10)

    def _check_performance_alerts(self, metrics: SystemMetrics) -> None:
        """Check for performance alerts"""
        alerts = []

        if metrics.cpu_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > 90:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        for mount, usage in metrics.disk_usage.items():
            if usage > 90:
                alerts.append(f"High disk usage on {mount}: {usage:.1f}%")

        # Temperature alerts
        for sensor, temp in metrics.temperature.items():
            if temp > 80:  # 80°C threshold
                alerts.append(f"High temperature {sensor}: {temp:.1f}°C")

        for alert in alerts:
            self.console.print(f"[bold red]🚨 ALERT: {alert}[/bold red]")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.stop_monitoring()
        if self._fs_notifier:
            self._fs_notifier.stop()