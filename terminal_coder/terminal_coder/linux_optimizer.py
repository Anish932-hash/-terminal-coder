"""
Linux-Specific Optimization Module
Performance optimizations and system integration for Linux environments
"""

import os
import sys
import subprocess
import asyncio
import psutil
import mmap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import resource
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import distro
import socket
import fcntl
import struct

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information structure"""
    distro_name: str
    distro_version: str
    kernel_version: str
    architecture: str
    cpu_count: int
    memory_total: int
    has_systemd: bool
    has_docker: bool
    has_podman: bool
    supports_cgroups_v2: bool
    shell: str
    terminal: str


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    open_files: int
    threads: int


class LinuxOptimizer:
    """Linux-specific optimizations and system integration"""

    def __init__(self):
        self.system_info = self._detect_system_info()
        self.performance_monitor = PerformanceMonitor()
        self.process_manager = ProcessManager()
        self.memory_optimizer = MemoryOptimizer()
        self.io_optimizer = IOOptimizer()

        # Set optimal resource limits
        self._set_resource_limits()

        # Enable performance optimizations
        self._enable_performance_optimizations()

    def _detect_system_info(self) -> SystemInfo:
        """Detect comprehensive Linux system information"""
        try:
            # Get distribution info
            distro_info = distro.info()
            distro_name = distro_info.get('id', 'unknown')
            distro_version = distro_info.get('version', 'unknown')

            # Get kernel and architecture
            uname = os.uname()
            kernel_version = uname.release
            architecture = uname.machine

            # System resources
            cpu_count = psutil.cpu_count()
            memory_total = psutil.virtual_memory().total

            # Check for systemd
            has_systemd = Path("/run/systemd/system").exists()

            # Check for containerization
            has_docker = shutil.which("docker") is not None
            has_podman = shutil.which("podman") is not None

            # Check cgroups v2 support
            supports_cgroups_v2 = Path("/sys/fs/cgroup/cgroup.controllers").exists()

            # Environment info
            shell = os.environ.get("SHELL", "/bin/bash")
            terminal = os.environ.get("TERM", "unknown")

            return SystemInfo(
                distro_name=distro_name,
                distro_version=distro_version,
                kernel_version=kernel_version,
                architecture=architecture,
                cpu_count=cpu_count,
                memory_total=memory_total,
                has_systemd=has_systemd,
                has_docker=has_docker,
                has_podman=has_podman,
                supports_cgroups_v2=supports_cgroups_v2,
                shell=shell,
                terminal=terminal
            )

        except Exception as e:
            logger.error(f"Error detecting system info: {e}")
            return SystemInfo(
                distro_name="unknown",
                distro_version="unknown",
                kernel_version="unknown",
                architecture="unknown",
                cpu_count=1,
                memory_total=0,
                has_systemd=False,
                has_docker=False,
                has_podman=False,
                supports_cgroups_v2=False,
                shell="/bin/bash",
                terminal="unknown"
            )

    def _set_resource_limits(self):
        """Set optimal resource limits for performance"""
        try:
            # Increase file descriptor limit
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(8192, hard), hard))

            # Set stack size limit (8MB)
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, resource.RLIM_INFINITY))

            # Set virtual memory limit based on available RAM
            available_memory = psutil.virtual_memory().available
            vm_limit = min(available_memory * 2, resource.RLIM_INFINITY)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (vm_limit, resource.RLIM_INFINITY))
            except OSError:
                pass  # Some systems don't support this

            logger.info("Resource limits optimized for performance")

        except Exception as e:
            logger.warning(f"Could not set optimal resource limits: {e}")

    def _enable_performance_optimizations(self):
        """Enable Linux-specific performance optimizations"""
        try:
            # Set process priority (nice value)
            os.nice(-5)  # Higher priority (requires permissions)

            # Set I/O priority
            self._set_io_priority()

            # Enable memory optimizations
            self._enable_memory_optimizations()

            # Set CPU affinity for better cache locality
            self._set_cpu_affinity()

            logger.info("Performance optimizations enabled")

        except Exception as e:
            logger.debug(f"Some performance optimizations not available: {e}")

    def _set_io_priority(self):
        """Set I/O priority using ionice"""
        try:
            # Best effort class with high priority
            subprocess.run(["ionice", "-c", "1", "-n", "4", "-p", str(os.getpid())],
                         check=False, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _enable_memory_optimizations(self):
        """Enable memory-specific optimizations"""
        try:
            # Advise kernel about memory usage patterns
            if hasattr(os, 'posix_fadvise'):
                # Will be used for file operations
                pass

            # Set memory allocation strategy
            os.environ['MALLOC_ARENA_MAX'] = str(max(2, self.system_info.cpu_count))

        except Exception as e:
            logger.debug(f"Memory optimizations not fully available: {e}")

    def _set_cpu_affinity(self):
        """Set CPU affinity for optimal performance"""
        try:
            # Use all available CPUs
            available_cpus = list(range(self.system_info.cpu_count))
            os.sched_setaffinity(0, available_cpus)

            # Set scheduling policy to SCHED_BATCH for CPU-intensive work
            try:
                os.sched_setscheduler(0, os.SCHED_BATCH, os.sched_param(0))
            except OSError:
                pass  # Not all systems support this

        except Exception as e:
            logger.debug(f"CPU affinity optimization not available: {e}")

    async def optimize_for_ai_workload(self):
        """Optimize system specifically for AI/ML workloads"""
        optimizations = []

        try:
            # Disable CPU frequency scaling for consistent performance
            await self._set_cpu_governor("performance")
            optimizations.append("CPU governor set to performance")

            # Increase VM dirty ratios for better write performance
            await self._optimize_vm_settings()
            optimizations.append("VM settings optimized")

            # Set network buffer sizes for better API performance
            await self._optimize_network_buffers()
            optimizations.append("Network buffers optimized")

            # Configure swap usage
            await self._optimize_swap_settings()
            optimizations.append("Swap settings optimized")

            logger.info(f"AI workload optimizations applied: {optimizations}")

        except Exception as e:
            logger.error(f"Error applying AI optimizations: {e}")

    async def _set_cpu_governor(self, governor: str):
        """Set CPU frequency governor"""
        try:
            cpu_dirs = Path("/sys/devices/system/cpu").glob("cpu[0-9]*")
            for cpu_dir in cpu_dirs:
                governor_file = cpu_dir / "cpufreq" / "scaling_governor"
                if governor_file.exists():
                    try:
                        with open(governor_file, 'w') as f:
                            f.write(governor)
                    except PermissionError:
                        # Try using cpupower if available
                        result = await asyncio.create_subprocess_exec(
                            "cpupower", "frequency-set", "-g", governor,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL
                        )
                        await result.wait()
                        break
        except Exception as e:
            logger.debug(f"Could not set CPU governor: {e}")

    async def _optimize_vm_settings(self):
        """Optimize virtual memory settings"""
        vm_settings = {
            "/proc/sys/vm/dirty_ratio": "15",
            "/proc/sys/vm/dirty_background_ratio": "5",
            "/proc/sys/vm/swappiness": "10",
            "/proc/sys/vm/vfs_cache_pressure": "50"
        }

        for setting, value in vm_settings.items():
            try:
                if Path(setting).exists():
                    with open(setting, 'w') as f:
                        f.write(value)
            except PermissionError:
                # Try with sysctl
                setting_name = setting.replace("/proc/sys/", "").replace("/", ".")
                result = await asyncio.create_subprocess_exec(
                    "sysctl", "-w", f"{setting_name}={value}",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await result.wait()

    async def _optimize_network_buffers(self):
        """Optimize network buffer sizes"""
        net_settings = {
            "/proc/sys/net/core/rmem_default": "262144",
            "/proc/sys/net/core/rmem_max": "16777216",
            "/proc/sys/net/core/wmem_default": "262144",
            "/proc/sys/net/core/wmem_max": "16777216",
            "/proc/sys/net/core/netdev_max_backlog": "5000"
        }

        for setting, value in net_settings.items():
            try:
                if Path(setting).exists():
                    with open(setting, 'w') as f:
                        f.write(value)
            except PermissionError:
                pass  # Skip if no permission

    async def _optimize_swap_settings(self):
        """Optimize swap usage for AI workloads"""
        try:
            # Reduce swappiness for AI workloads that need memory
            with open("/proc/sys/vm/swappiness", "w") as f:
                f.write("1")
        except PermissionError:
            pass


class PerformanceMonitor:
    """Real-time performance monitoring for Linux"""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics_history = []
        self.max_history = 100

    async def start_monitoring(self, callback=None):
        """Start continuous performance monitoring"""
        self.monitoring = True

        while self.monitoring:
            try:
                metrics = await self.collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)

                # Call callback if provided
                if callback:
                    await callback(metrics)

                await asyncio.sleep(self.interval)

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(self.interval)

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_read = disk_io.read_bytes if disk_io else 0
            disk_io_write = disk_io.write_bytes if disk_io else 0

            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent = network_io.bytes_sent if network_io else 0
            network_recv = network_io.bytes_recv if network_io else 0

            # Process info
            process = psutil.Process()
            open_files = len(process.open_files())
            threads = process.num_threads()

            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_io_read=disk_io_read,
                disk_io_write=disk_io_write,
                network_sent=network_sent,
                network_recv=network_recv,
                open_files=open_files,
                threads=threads
            )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    def get_average_metrics(self, duration_seconds: int = 60) -> Optional[PerformanceMetrics]:
        """Get average metrics over specified duration"""
        if not self.metrics_history:
            return None

        # Get recent metrics
        cutoff_time = time.time() - duration_seconds
        recent_metrics = self.metrics_history[-min(len(self.metrics_history),
                                                  int(duration_seconds / self.interval)):]

        if not recent_metrics:
            return None

        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_disk_read = sum(m.disk_io_read for m in recent_metrics) / len(recent_metrics)
        avg_disk_write = sum(m.disk_io_write for m in recent_metrics) / len(recent_metrics)
        avg_net_sent = sum(m.network_sent for m in recent_metrics) / len(recent_metrics)
        avg_net_recv = sum(m.network_recv for m in recent_metrics) / len(recent_metrics)
        avg_files = sum(m.open_files for m in recent_metrics) / len(recent_metrics)
        avg_threads = sum(m.threads for m in recent_metrics) / len(recent_metrics)

        return PerformanceMetrics(
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            disk_io_read=int(avg_disk_read),
            disk_io_write=int(avg_disk_write),
            network_sent=int(avg_net_sent),
            network_recv=int(avg_net_recv),
            open_files=int(avg_files),
            threads=int(avg_threads)
        )


class ProcessManager:
    """Advanced process management for Linux"""

    def __init__(self):
        self.child_processes = []

    async def run_optimized_command(self, cmd: List[str], cwd: str = None,
                                  env: Dict[str, str] = None) -> Tuple[int, str, str]:
        """Run command with Linux-specific optimizations"""
        try:
            # Prepare environment
            proc_env = os.environ.copy()
            if env:
                proc_env.update(env)

            # Add performance-oriented environment variables
            proc_env.update({
                'PYTHONUNBUFFERED': '1',
                'PYTHONDONTWRITEBYTECODE': '1',
                'OMP_NUM_THREADS': str(os.cpu_count()),
                'MKL_NUM_THREADS': str(os.cpu_count())
            })

            # Create subprocess with optimizations
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=proc_env,
                preexec_fn=self._preexec_fn
            )

            self.child_processes.append(process)

            # Wait for completion
            stdout, stderr = await process.communicate()

            # Remove from tracking
            if process in self.child_processes:
                self.child_processes.remove(process)

            return process.returncode, stdout.decode(), stderr.decode()

        except Exception as e:
            logger.error(f"Error running command: {e}")
            return -1, "", str(e)

    def _preexec_fn(self):
        """Pre-execution function for subprocess optimization"""
        try:
            # Set process group
            os.setpgrp()

            # Set nice value
            os.nice(5)  # Lower priority for subprocesses

            # Set I/O priority
            try:
                subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())],
                             check=False, stderr=subprocess.DEVNULL)
            except:
                pass

        except Exception:
            pass  # Ignore errors in preexec

    def cleanup_processes(self):
        """Clean up any remaining child processes"""
        for process in self.child_processes:
            try:
                if process.returncode is None:
                    process.terminate()
            except:
                pass

        self.child_processes.clear()


class MemoryOptimizer:
    """Memory optimization utilities for Linux"""

    def __init__(self):
        self.memory_mapped_files = []

    def optimize_large_file_access(self, file_path: str) -> Optional[mmap.mmap]:
        """Optimize access to large files using memory mapping"""
        try:
            file_obj = open(file_path, 'r+b')
            mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)

            # Advise kernel about access pattern
            mm.madvise(mmap.MADV_SEQUENTIAL)

            self.memory_mapped_files.append((file_obj, mm))
            return mm

        except Exception as e:
            logger.error(f"Error memory mapping file {file_path}: {e}")
            return None

    def enable_huge_pages(self):
        """Enable transparent huge pages for better memory performance"""
        try:
            thp_path = Path("/sys/kernel/mm/transparent_hugepage/enabled")
            if thp_path.exists():
                with open(thp_path, 'w') as f:
                    f.write("always")
                logger.info("Transparent huge pages enabled")
        except PermissionError:
            logger.debug("Cannot enable transparent huge pages (permission denied)")
        except Exception as e:
            logger.error(f"Error enabling huge pages: {e}")

    def cleanup_memory_maps(self):
        """Clean up memory mapped files"""
        for file_obj, mm in self.memory_mapped_files:
            try:
                mm.close()
                file_obj.close()
            except:
                pass
        self.memory_mapped_files.clear()

    async def monitor_memory_pressure(self) -> Dict[str, Any]:
        """Monitor memory pressure indicators"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Check for memory pressure indicators
            pressure_indicators = {
                "memory_available_percent": (memory.available / memory.total) * 100,
                "swap_usage_percent": swap.percent if swap.total > 0 else 0,
                "memory_cached_percent": (memory.cached / memory.total) * 100,
                "memory_buffers_percent": (memory.buffers / memory.total) * 100,
            }

            # Check /proc/pressure/memory if available (kernel 4.20+)
            pressure_file = Path("/proc/pressure/memory")
            if pressure_file.exists():
                try:
                    with open(pressure_file, 'r') as f:
                        content = f.read()
                        # Parse pressure metrics
                        for line in content.strip().split('\n'):
                            if line.startswith('some'):
                                # Extract avg10, avg60, avg300 values
                                parts = line.split()
                                for part in parts[1:]:
                                    key, value = part.split('=')
                                    if key in ['avg10', 'avg60', 'avg300']:
                                        pressure_indicators[f"memory_pressure_{key}"] = float(value)
                except Exception:
                    pass

            return pressure_indicators

        except Exception as e:
            logger.error(f"Error monitoring memory pressure: {e}")
            return {}


class IOOptimizer:
    """I/O optimization utilities for Linux"""

    def __init__(self):
        self.optimized_files = set()

    async def optimize_file_io(self, file_path: str, access_pattern: str = "sequential"):
        """Optimize file I/O based on access pattern"""
        try:
            fd = os.open(file_path, os.O_RDONLY)

            # Set appropriate advice based on access pattern
            if access_pattern == "sequential":
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            elif access_pattern == "random":
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            elif access_pattern == "willneed":
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            elif access_pattern == "dontneed":
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

            os.close(fd)
            self.optimized_files.add(file_path)

            logger.debug(f"Optimized I/O for {file_path} with pattern {access_pattern}")

        except Exception as e:
            logger.error(f"Error optimizing I/O for {file_path}: {e}")

    def get_io_stats(self) -> Dict[str, Any]:
        """Get detailed I/O statistics"""
        try:
            # Get process I/O stats
            process = psutil.Process()
            io_counters = process.io_counters()

            # Get system-wide I/O stats
            disk_io = psutil.disk_io_counters()

            stats = {
                "process_read_bytes": io_counters.read_bytes,
                "process_write_bytes": io_counters.write_bytes,
                "process_read_count": io_counters.read_count,
                "process_write_count": io_counters.write_count,
            }

            if disk_io:
                stats.update({
                    "system_read_bytes": disk_io.read_bytes,
                    "system_write_bytes": disk_io.write_bytes,
                    "system_read_count": disk_io.read_count,
                    "system_write_count": disk_io.write_count,
                    "system_read_time": disk_io.read_time,
                    "system_write_time": disk_io.write_time,
                })

            return stats

        except Exception as e:
            logger.error(f"Error getting I/O stats: {e}")
            return {}

    async def setup_async_io(self):
        """Setup asynchronous I/O optimizations"""
        try:
            # Set I/O scheduler to deadline or noop for better performance
            block_devices = Path("/sys/block").glob("sd*")
            for device in block_devices:
                scheduler_path = device / "queue" / "scheduler"
                if scheduler_path.exists():
                    try:
                        with open(scheduler_path, 'w') as f:
                            f.write("deadline")
                    except PermissionError:
                        pass

        except Exception as e:
            logger.debug(f"I/O scheduler optimization not available: {e}")


# Global instance
linux_optimizer = LinuxOptimizer()


async def initialize_linux_optimizations():
    """Initialize all Linux-specific optimizations"""
    try:
        logger.info("Initializing Linux optimizations...")

        # Apply AI workload optimizations
        await linux_optimizer.optimize_for_ai_workload()

        # Setup memory optimizations
        linux_optimizer.memory_optimizer.enable_huge_pages()

        # Setup I/O optimizations
        await linux_optimizer.io_optimizer.setup_async_io()

        logger.info("Linux optimizations initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing Linux optimizations: {e}")


def get_system_info() -> SystemInfo:
    """Get comprehensive system information"""
    return linux_optimizer.system_info


def get_performance_metrics() -> Optional[PerformanceMetrics]:
    """Get current performance metrics"""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(linux_optimizer.performance_monitor.collect_metrics())
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return None


async def monitor_system_health(callback=None):
    """Start system health monitoring"""
    await linux_optimizer.performance_monitor.start_monitoring(callback)