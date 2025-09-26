"""
Kernel-Level Performance Optimizations
Advanced Linux kernel interactions, eBPF programs, and system-level optimizations
"""

import os
import sys
import asyncio
import logging
import subprocess
import time
import ctypes
import struct
import mmap
import resource
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import signal

try:
    # Try to import eBPF libraries
    from bcc import BPF
    import pyroute2
    EBPF_AVAILABLE = True
except ImportError:
    EBPF_AVAILABLE = False
    logging.warning("eBPF libraries not available. Some kernel optimizations will be limited.")

try:
    import psutil
    import prctl
    PROCESS_CONTROL_AVAILABLE = True
except ImportError:
    PROCESS_CONTROL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class KernelMetrics:
    """Kernel-level performance metrics"""
    cpu_frequency: Dict[int, float]
    memory_pressure: float
    io_wait_time: float
    context_switches: int
    interrupts: int
    system_calls: int
    cache_misses: int
    page_faults: int
    network_packets: Tuple[int, int]  # rx, tx
    disk_io: Tuple[int, int]  # read, write


@dataclass
class ProcessPriority:
    """Process scheduling priority settings"""
    nice: int = 0
    io_class: int = 1  # Best effort
    io_priority: int = 4  # Normal
    cpu_affinity: List[int] = None
    scheduler_policy: str = "SCHED_OTHER"


class eBPFPrograms:
    """eBPF programs for kernel-level monitoring and optimization"""

    def __init__(self):
        self.programs = {}
        self.active_programs = {}

        # CPU performance monitoring eBPF program
        self.cpu_monitor_program = """
#include <uapi/linux/ptrace.h>
#include <uapi/linux/bpf_perf_event.h>
#include <linux/sched.h>

BPF_HASH(cpu_time, u32, u64);
BPF_HASH(context_switches, u32, u64);
BPF_PERF_OUTPUT(events);

struct data_t {
    u32 pid;
    u64 cpu_time;
    u64 timestamp;
    char comm[TASK_COMM_LEN];
};

int trace_cpu_cycles(struct bpf_perf_event_data *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 ts = bpf_ktime_get_ns();

    u64 *val = cpu_time.lookup(&pid);
    if (val) {
        *val += ctx->sample_period;
    } else {
        u64 initial = ctx->sample_period;
        cpu_time.update(&pid, &initial);
    }

    struct data_t data = {};
    data.pid = pid;
    data.cpu_time = ctx->sample_period;
    data.timestamp = ts;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

TRACEPOINT_PROBE(sched, sched_switch) {
    u32 pid = args->prev_pid;
    u64 *val = context_switches.lookup(&pid);
    if (val) {
        (*val)++;
    } else {
        u64 initial = 1;
        context_switches.update(&pid, &initial);
    }
    return 0;
}
"""

        # Memory access pattern eBPF program
        self.memory_monitor_program = """
#include <uapi/linux/ptrace.h>
#include <linux/mm.h>

BPF_HASH(page_faults, u32, u64);
BPF_HASH(memory_allocations, u32, u64);
BPF_PERF_OUTPUT(memory_events);

struct memory_event_t {
    u32 pid;
    u64 address;
    u64 size;
    u64 timestamp;
    u32 type;  // 0: alloc, 1: free, 2: page_fault
};

TRACEPOINT_PROBE(exceptions, page_fault_user) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    u64 *val = page_faults.lookup(&pid);
    if (val) {
        (*val)++;
    } else {
        u64 initial = 1;
        page_faults.update(&pid, &initial);
    }

    struct memory_event_t event = {};
    event.pid = pid;
    event.address = args->address;
    event.timestamp = bpf_ktime_get_ns();
    event.type = 2;  // page fault

    memory_events.perf_submit(args, &event, sizeof(event));
    return 0;
}

TRACEPOINT_PROBE(kmem, kmalloc) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    u64 *val = memory_allocations.lookup(&pid);
    if (val) {
        *val += args->bytes_alloc;
    } else {
        memory_allocations.update(&pid, &args->bytes_alloc);
    }

    struct memory_event_t event = {};
    event.pid = pid;
    event.size = args->bytes_alloc;
    event.timestamp = bpf_ktime_get_ns();
    event.type = 0;  // allocation

    memory_events.perf_submit(args, &event, sizeof(event));
    return 0;
}
"""

        # I/O performance monitoring eBPF program
        self.io_monitor_program = """
#include <uapi/linux/ptrace.h>
#include <linux/blkdev.h>

BPF_HASH(io_latency, u64, u64);
BPF_HASH(io_size, u32, u64);
BPF_PERF_OUTPUT(io_events);

struct io_event_t {
    u32 pid;
    u32 dev;
    u64 sector;
    u32 len;
    u64 timestamp;
    u64 latency;
    u32 rwbs;
};

TRACEPOINT_PROBE(block, block_rq_complete) {
    struct io_event_t event = {};
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.dev = args->dev;
    event.sector = args->sector;
    event.len = args->nr_sector * 512;
    event.timestamp = bpf_ktime_get_ns();

    // Calculate I/O latency (simplified)
    u64 *start_time = io_latency.lookup(&args->sector);
    if (start_time) {
        event.latency = event.timestamp - *start_time;
        io_latency.delete(&args->sector);
    }

    io_events.perf_submit(args, &event, sizeof(event));
    return 0;
}

TRACEPOINT_PROBE(block, block_rq_issue) {
    u64 ts = bpf_ktime_get_ns();
    io_latency.update(&args->sector, &ts);
    return 0;
}
"""

    async def load_program(self, program_name: str, program_source: str) -> bool:
        """Load an eBPF program into the kernel"""
        try:
            if not EBPF_AVAILABLE:
                logger.warning("eBPF not available, cannot load programs")
                return False

            # Compile and load the eBPF program
            bpf_program = BPF(text=program_source)
            self.programs[program_name] = bpf_program

            logger.info(f"Loaded eBPF program: {program_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load eBPF program {program_name}: {e}")
            return False

    async def start_cpu_monitoring(self, callback=None) -> bool:
        """Start CPU performance monitoring with eBPF"""
        try:
            if not await self.load_program("cpu_monitor", self.cpu_monitor_program):
                return False

            bpf = self.programs["cpu_monitor"]

            # Attach to CPU cycles perf event
            bpf.attach_perf_event(
                ev_type=BPF.PERF_TYPE_HARDWARE,
                ev_config=BPF.PERF_COUNT_HW_CPU_CYCLES,
                fn_name="trace_cpu_cycles"
            )

            # Start event polling
            def handle_cpu_event(cpu, data, size):
                event = ctypes.cast(data, ctypes.POINTER(CPUEvent)).contents
                if callback:
                    asyncio.create_task(callback("cpu", event))

            bpf["events"].open_perf_buffer(handle_cpu_event)

            self.active_programs["cpu_monitor"] = bpf
            logger.info("CPU monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to start CPU monitoring: {e}")
            return False

    async def start_memory_monitoring(self, callback=None) -> bool:
        """Start memory monitoring with eBPF"""
        try:
            if not await self.load_program("memory_monitor", self.memory_monitor_program):
                return False

            bpf = self.programs["memory_monitor"]

            def handle_memory_event(cpu, data, size):
                event = ctypes.cast(data, ctypes.POINTER(MemoryEvent)).contents
                if callback:
                    asyncio.create_task(callback("memory", event))

            bpf["memory_events"].open_perf_buffer(handle_memory_event)

            self.active_programs["memory_monitor"] = bpf
            logger.info("Memory monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to start memory monitoring: {e}")
            return False

    async def start_io_monitoring(self, callback=None) -> bool:
        """Start I/O monitoring with eBPF"""
        try:
            if not await self.load_program("io_monitor", self.io_monitor_program):
                return False

            bpf = self.programs["io_monitor"]

            def handle_io_event(cpu, data, size):
                event = ctypes.cast(data, ctypes.POINTER(IOEvent)).contents
                if callback:
                    asyncio.create_task(callback("io", event))

            bpf["io_events"].open_perf_buffer(handle_io_event)

            self.active_programs["io_monitor"] = bpf
            logger.info("I/O monitoring started")
            return True

        except Exception as e:
            logger.error(f"Failed to start I/O monitoring: {e}")
            return False

    async def poll_events(self):
        """Poll for eBPF events"""
        try:
            for program_name, bpf in self.active_programs.items():
                try:
                    bpf.perf_buffer_poll(timeout=10)
                except Exception as e:
                    logger.debug(f"Event polling error for {program_name}: {e}")
        except Exception as e:
            logger.error(f"eBPF event polling failed: {e}")

    def cleanup(self):
        """Cleanup eBPF programs"""
        try:
            for program_name, bpf in self.active_programs.items():
                try:
                    bpf.cleanup()
                except Exception as e:
                    logger.error(f"Failed to cleanup {program_name}: {e}")

            self.active_programs.clear()
            self.programs.clear()
            logger.info("eBPF programs cleaned up")

        except Exception as e:
            logger.error(f"eBPF cleanup failed: {e}")


# eBPF event structures
class CPUEvent(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint32),
        ("cpu_time", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
        ("comm", ctypes.c_char * 16)
    ]


class MemoryEvent(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint32),
        ("address", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
        ("type", ctypes.c_uint32)
    ]


class IOEvent(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint32),
        ("dev", ctypes.c_uint32),
        ("sector", ctypes.c_uint64),
        ("len", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint64),
        ("latency", ctypes.c_uint64),
        ("rwbs", ctypes.c_uint32)
    ]


class ProcessOptimizer:
    """Advanced process-level optimizations using kernel interfaces"""

    def __init__(self):
        self.optimized_processes = {}
        self.cgroup_manager = CGroupManager()

    async def optimize_process(self, pid: int, priority: ProcessPriority) -> bool:
        """Apply comprehensive process optimizations"""
        try:
            process_optimizations = []

            # Set process nice value
            if priority.nice != 0:
                os.setpriority(os.PRIO_PROCESS, pid, priority.nice)
                process_optimizations.append(f"nice: {priority.nice}")

            # Set I/O priority using ionice
            try:
                subprocess.run([
                    "ionice", "-c", str(priority.io_class),
                    "-n", str(priority.io_priority),
                    "-p", str(pid)
                ], check=False, stderr=subprocess.DEVNULL)
                process_optimizations.append(f"ionice: class {priority.io_class}, priority {priority.io_priority}")
            except Exception:
                pass

            # Set CPU affinity
            if priority.cpu_affinity and PROCESS_CONTROL_AVAILABLE:
                try:
                    os.sched_setaffinity(pid, priority.cpu_affinity)
                    process_optimizations.append(f"CPU affinity: {priority.cpu_affinity}")
                except Exception as e:
                    logger.debug(f"Failed to set CPU affinity: {e}")

            # Set scheduling policy
            try:
                if priority.scheduler_policy == "SCHED_FIFO":
                    os.sched_setscheduler(pid, os.SCHED_FIFO, os.sched_param(1))
                elif priority.scheduler_policy == "SCHED_RR":
                    os.sched_setscheduler(pid, os.SCHED_RR, os.sched_param(1))
                elif priority.scheduler_policy == "SCHED_BATCH":
                    os.sched_setscheduler(pid, os.SCHED_BATCH, os.sched_param(0))
                process_optimizations.append(f"scheduler: {priority.scheduler_policy}")
            except Exception as e:
                logger.debug(f"Failed to set scheduler policy: {e}")

            # Apply memory optimizations
            await self._optimize_process_memory(pid)
            process_optimizations.append("memory optimized")

            self.optimized_processes[pid] = {
                "priority": priority,
                "optimizations": process_optimizations,
                "timestamp": time.time()
            }

            logger.info(f"Process {pid} optimized: {', '.join(process_optimizations)}")
            return True

        except Exception as e:
            logger.error(f"Process optimization failed for PID {pid}: {e}")
            return False

    async def _optimize_process_memory(self, pid: int):
        """Optimize memory usage for a specific process"""
        try:
            # Set memory advice for the process
            proc_path = Path(f"/proc/{pid}")
            if not proc_path.exists():
                return

            # Read process memory maps
            maps_file = proc_path / "maps"
            if maps_file.exists():
                with open(maps_file, 'r') as f:
                    maps_content = f.read()

                # Analyze memory regions and provide advice
                for line in maps_content.split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            address_range = parts[0]
                            if '-' in address_range:
                                start_addr, end_addr = address_range.split('-')
                                try:
                                    start = int(start_addr, 16)
                                    end = int(end_addr, 16)

                                    # Advise kernel about memory access patterns
                                    # This is a simplified approach
                                    if "heap" in line or "stack" in line:
                                        # Advise sequential access for heap/stack
                                        pass
                                    elif ".so" in line or "lib" in line:
                                        # Advise random access for libraries
                                        pass

                                except ValueError:
                                    continue

        except Exception as e:
            logger.debug(f"Memory optimization failed for PID {pid}: {e}")

    async def create_high_priority_cgroup(self, name: str) -> bool:
        """Create a high-priority cgroup for Terminal Coder processes"""
        try:
            return await self.cgroup_manager.create_cgroup(name, {
                "cpu.weight": "1000",  # High CPU weight
                "memory.high": "4G",   # 4GB memory limit
                "io.weight": "200"     # High I/O weight
            })
        except Exception as e:
            logger.error(f"Failed to create high-priority cgroup: {e}")
            return False

    async def add_process_to_cgroup(self, pid: int, cgroup_name: str) -> bool:
        """Add process to optimized cgroup"""
        try:
            return await self.cgroup_manager.add_process(cgroup_name, pid)
        except Exception as e:
            logger.error(f"Failed to add process {pid} to cgroup {cgroup_name}: {e}")
            return False


class CGroupManager:
    """Linux Control Groups (cgroups) management"""

    def __init__(self):
        self.cgroup_root = Path("/sys/fs/cgroup")
        self.terminal_coder_cgroups = {}

    async def create_cgroup(self, name: str, settings: Dict[str, str]) -> bool:
        """Create a new cgroup with specified settings"""
        try:
            cgroup_path = self.cgroup_root / name

            if cgroup_path.exists():
                logger.info(f"Cgroup {name} already exists")
                return True

            # Create cgroup directory
            cgroup_path.mkdir(parents=True, exist_ok=True)

            # Apply settings
            for setting, value in settings.items():
                setting_file = cgroup_path / setting
                if setting_file.exists():
                    try:
                        with open(setting_file, 'w') as f:
                            f.write(value)
                        logger.debug(f"Set {setting} = {value} for cgroup {name}")
                    except PermissionError:
                        logger.warning(f"Permission denied setting {setting} for cgroup {name}")
                    except Exception as e:
                        logger.warning(f"Failed to set {setting}: {e}")

            self.terminal_coder_cgroups[name] = cgroup_path
            logger.info(f"Created cgroup: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create cgroup {name}: {e}")
            return False

    async def add_process(self, cgroup_name: str, pid: int) -> bool:
        """Add process to cgroup"""
        try:
            if cgroup_name not in self.terminal_coder_cgroups:
                logger.error(f"Cgroup {cgroup_name} not found")
                return False

            cgroup_path = self.terminal_coder_cgroups[cgroup_name]
            procs_file = cgroup_path / "cgroup.procs"

            if procs_file.exists():
                with open(procs_file, 'w') as f:
                    f.write(str(pid))
                logger.info(f"Added process {pid} to cgroup {cgroup_name}")
                return True
            else:
                logger.error(f"cgroup.procs file not found for {cgroup_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to add process {pid} to cgroup {cgroup_name}: {e}")
            return False

    async def get_cgroup_stats(self, cgroup_name: str) -> Dict[str, Any]:
        """Get cgroup resource usage statistics"""
        try:
            if cgroup_name not in self.terminal_coder_cgroups:
                return {}

            cgroup_path = self.terminal_coder_cgroups[cgroup_name]
            stats = {}

            # CPU stats
            cpu_stat_file = cgroup_path / "cpu.stat"
            if cpu_stat_file.exists():
                with open(cpu_stat_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            key, value = line.strip().split(None, 1)
                            stats[f"cpu_{key}"] = value

            # Memory stats
            memory_stat_file = cgroup_path / "memory.stat"
            if memory_stat_file.exists():
                with open(memory_stat_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            key, value = line.strip().split(None, 1)
                            stats[f"memory_{key}"] = value

            # Current memory usage
            memory_current_file = cgroup_path / "memory.current"
            if memory_current_file.exists():
                with open(memory_current_file, 'r') as f:
                    stats["memory_current"] = f.read().strip()

            return stats

        except Exception as e:
            logger.error(f"Failed to get cgroup stats for {cgroup_name}: {e}")
            return {}


class KernelTuner:
    """System kernel parameter tuning for performance"""

    def __init__(self):
        self.original_values = {}
        self.applied_tunings = {}

    async def apply_performance_tuning(self) -> Dict[str, bool]:
        """Apply comprehensive kernel performance tuning"""
        results = {}

        # Network performance tunings
        network_tunings = {
            "net.core.rmem_default": "262144",
            "net.core.rmem_max": "67108864",
            "net.core.wmem_default": "262144",
            "net.core.wmem_max": "67108864",
            "net.core.netdev_max_backlog": "5000",
            "net.core.somaxconn": "65535",
            "net.ipv4.tcp_rmem": "4096 65536 67108864",
            "net.ipv4.tcp_wmem": "4096 65536 67108864",
            "net.ipv4.tcp_congestion_control": "bbr",
        }

        for param, value in network_tunings.items():
            results[f"network_{param}"] = await self._set_sysctl(param, value)

        # Memory management tunings
        memory_tunings = {
            "vm.swappiness": "10",
            "vm.dirty_ratio": "15",
            "vm.dirty_background_ratio": "5",
            "vm.vfs_cache_pressure": "50",
            "vm.min_free_kbytes": "65536",
        }

        for param, value in memory_tunings.items():
            results[f"memory_{param}"] = await self._set_sysctl(param, value)

        # File system tunings
        fs_tunings = {
            "fs.file-max": "1048576",
            "fs.nr_open": "1048576",
        }

        for param, value in fs_tunings.items():
            results[f"fs_{param}"] = await self._set_sysctl(param, value)

        # Kernel tunings
        kernel_tunings = {
            "kernel.sched_autogroup_enabled": "0",
            "kernel.sched_migration_cost_ns": "5000000",
        }

        for param, value in kernel_tunings.items():
            results[f"kernel_{param}"] = await self._set_sysctl(param, value)

        return results

    async def _set_sysctl(self, parameter: str, value: str) -> bool:
        """Set a sysctl parameter"""
        try:
            # Store original value for restoration
            try:
                result = subprocess.run(
                    ["sysctl", "-n", parameter],
                    capture_output=True, text=True, check=True
                )
                self.original_values[parameter] = result.stdout.strip()
            except Exception:
                pass

            # Set new value
            result = subprocess.run(
                ["sysctl", "-w", f"{parameter}={value}"],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                self.applied_tunings[parameter] = value
                logger.debug(f"Set {parameter} = {value}")
                return True
            else:
                logger.warning(f"Failed to set {parameter}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error setting {parameter}: {e}")
            return False

    async def restore_original_values(self):
        """Restore original sysctl values"""
        try:
            for parameter, original_value in self.original_values.items():
                try:
                    subprocess.run(
                        ["sysctl", "-w", f"{parameter}={original_value}"],
                        capture_output=True, check=True
                    )
                    logger.debug(f"Restored {parameter} = {original_value}")
                except Exception as e:
                    logger.warning(f"Failed to restore {parameter}: {e}")

            self.applied_tunings.clear()
            logger.info("Restored original kernel parameters")

        except Exception as e:
            logger.error(f"Failed to restore kernel parameters: {e}")


class KernelOptimizer:
    """Main kernel optimization coordinator"""

    def __init__(self):
        self.ebpf_programs = eBPFPrograms()
        self.process_optimizer = ProcessOptimizer()
        self.kernel_tuner = KernelTuner()
        self.monitoring_active = False
        self.metrics_cache = {}
        self.optimization_thread = None

    async def initialize(self) -> bool:
        """Initialize kernel optimization system"""
        try:
            logger.info("Initializing kernel optimization system...")

            # Check for required capabilities
            capabilities = await self._check_capabilities()
            logger.info(f"Available capabilities: {capabilities}")

            # Apply kernel tuning
            tuning_results = await self.kernel_tuner.apply_performance_tuning()
            successful_tunings = sum(1 for success in tuning_results.values() if success)
            logger.info(f"Applied {successful_tunings}/{len(tuning_results)} kernel tunings")

            # Create optimized cgroup for Terminal Coder
            await self.process_optimizer.create_high_priority_cgroup("terminal_coder")

            # Optimize current process
            current_pid = os.getpid()
            priority = ProcessPriority(
                nice=-5,
                io_class=1,  # Best effort
                io_priority=2,  # High priority
                cpu_affinity=list(range(min(4, os.cpu_count()))),  # Use first 4 CPUs
                scheduler_policy="SCHED_BATCH"
            )

            await self.process_optimizer.optimize_process(current_pid, priority)
            await self.process_optimizer.add_process_to_cgroup(current_pid, "terminal_coder")

            logger.info("Kernel optimization system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize kernel optimizer: {e}")
            return False

    async def _check_capabilities(self) -> Dict[str, bool]:
        """Check available system capabilities"""
        capabilities = {
            "ebpf": EBPF_AVAILABLE,
            "cgroups_v2": Path("/sys/fs/cgroup/cgroup.controllers").exists(),
            "process_control": PROCESS_CONTROL_AVAILABLE,
            "root_access": os.geteuid() == 0,
            "sysctl_access": True  # Will be tested during tuning
        }

        # Test eBPF availability
        if EBPF_AVAILABLE:
            try:
                test_program = BPF(text="int hello(void *ctx) { return 0; }")
                test_program.cleanup()
                capabilities["ebpf_functional"] = True
            except Exception:
                capabilities["ebpf_functional"] = False
        else:
            capabilities["ebpf_functional"] = False

        return capabilities

    async def start_monitoring(self, callback=None) -> bool:
        """Start kernel-level performance monitoring"""
        try:
            if self.monitoring_active:
                return True

            monitoring_tasks = []

            # Start eBPF monitoring if available
            if EBPF_AVAILABLE:
                if await self.ebpf_programs.start_cpu_monitoring(callback):
                    monitoring_tasks.append("cpu_monitoring")

                if await self.ebpf_programs.start_memory_monitoring(callback):
                    monitoring_tasks.append("memory_monitoring")

                if await self.ebpf_programs.start_io_monitoring(callback):
                    monitoring_tasks.append("io_monitoring")

                # Start event polling
                if monitoring_tasks:
                    self.optimization_thread = threading.Thread(
                        target=self._monitoring_loop,
                        daemon=True
                    )
                    self.optimization_thread.start()

            self.monitoring_active = True
            logger.info(f"Kernel monitoring started: {monitoring_tasks}")
            return len(monitoring_tasks) > 0

        except Exception as e:
            logger.error(f"Failed to start kernel monitoring: {e}")
            return False

    def _monitoring_loop(self):
        """Background monitoring loop"""
        try:
            while self.monitoring_active:
                asyncio.run(self.ebpf_programs.poll_events())
                time.sleep(0.1)  # 100ms polling interval
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")

    async def collect_kernel_metrics(self) -> KernelMetrics:
        """Collect comprehensive kernel metrics"""
        try:
            # CPU frequency information
            cpu_freq = {}
            if Path("/sys/devices/system/cpu").exists():
                for cpu_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
                    cpu_id = int(cpu_dir.name[3:])
                    freq_file = cpu_dir / "cpufreq" / "scaling_cur_freq"
                    if freq_file.exists():
                        try:
                            with open(freq_file, 'r') as f:
                                freq_khz = int(f.read().strip())
                                cpu_freq[cpu_id] = freq_khz / 1000  # Convert to MHz
                        except Exception:
                            pass

            # Memory pressure
            memory_pressure = 0.0
            pressure_file = Path("/proc/pressure/memory")
            if pressure_file.exists():
                try:
                    with open(pressure_file, 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if line.startswith('some'):
                                parts = line.split()
                                for part in parts[1:]:
                                    if part.startswith('avg10='):
                                        memory_pressure = float(part.split('=')[1])
                                        break
                except Exception:
                    pass

            # System-wide statistics
            stat_info = {}
            try:
                with open("/proc/stat", 'r') as f:
                    for line in f:
                        if line.startswith('cpu '):
                            continue
                        elif line.startswith('ctxt '):
                            stat_info['context_switches'] = int(line.split()[1])
                        elif line.startswith('intr '):
                            stat_info['interrupts'] = int(line.split()[1])
                        elif line.startswith('processes '):
                            stat_info['processes_created'] = int(line.split()[1])
            except Exception:
                stat_info = {}

            # Network statistics
            net_rx, net_tx = 0, 0
            try:
                with open("/proc/net/dev", 'r') as f:
                    for line in f:
                        if ':' in line and not line.strip().startswith('Inter'):
                            parts = line.split()
                            if len(parts) >= 10:
                                net_rx += int(parts[1])
                                net_tx += int(parts[9])
            except Exception:
                pass

            return KernelMetrics(
                cpu_frequency=cpu_freq,
                memory_pressure=memory_pressure,
                io_wait_time=0.0,  # Would need eBPF for accurate measurement
                context_switches=stat_info.get('context_switches', 0),
                interrupts=stat_info.get('interrupts', 0),
                system_calls=0,  # Would need eBPF
                cache_misses=0,  # Would need perf events
                page_faults=0,  # Would need eBPF
                network_packets=(net_rx, net_tx),
                disk_io=(0, 0)  # Would need eBPF
            )

        except Exception as e:
            logger.error(f"Failed to collect kernel metrics: {e}")
            return KernelMetrics({}, 0.0, 0.0, 0, 0, 0, 0, 0, (0, 0), (0, 0))

    async def optimize_for_ai_workload(self):
        """Apply AI/ML specific optimizations"""
        try:
            optimizations = []

            # Set CPU governor to performance
            try:
                for cpu_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
                    governor_file = cpu_dir / "cpufreq" / "scaling_governor"
                    if governor_file.exists():
                        with open(governor_file, 'w') as f:
                            f.write("performance")
                optimizations.append("CPU governor: performance")
            except Exception as e:
                logger.debug(f"CPU governor optimization failed: {e}")

            # Disable CPU idle states for low latency
            try:
                idle_file = Path("/sys/devices/system/cpu/cpu0/cpuidle/state1/disable")
                if idle_file.exists():
                    with open(idle_file, 'w') as f:
                        f.write("1")
                optimizations.append("CPU idle states: disabled")
            except Exception as e:
                logger.debug(f"CPU idle state optimization failed: {e}")

            # Set I/O scheduler to deadline
            try:
                for block_device in Path("/sys/block").glob("sd*"):
                    scheduler_file = block_device / "queue" / "scheduler"
                    if scheduler_file.exists():
                        with open(scheduler_file, 'w') as f:
                            f.write("deadline")
                        optimizations.append("I/O scheduler: deadline")
                        break  # Only need to set for one device
            except Exception as e:
                logger.debug(f"I/O scheduler optimization failed: {e}")

            logger.info(f"AI workload optimizations applied: {optimizations}")

        except Exception as e:
            logger.error(f"AI workload optimization failed: {e}")

    async def cleanup(self):
        """Cleanup kernel optimization resources"""
        try:
            self.monitoring_active = False

            # Wait for monitoring thread to finish
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=2)

            # Cleanup eBPF programs
            self.ebpf_programs.cleanup()

            # Restore kernel parameters
            await self.kernel_tuner.restore_original_values()

            logger.info("Kernel optimizer cleaned up successfully")

        except Exception as e:
            logger.error(f"Kernel optimizer cleanup failed: {e}")


# Global kernel optimizer instance
kernel_optimizer = None


async def initialize_kernel_optimizations():
    """Initialize global kernel optimization system"""
    global kernel_optimizer
    try:
        kernel_optimizer = KernelOptimizer()
        success = await kernel_optimizer.initialize()

        if success:
            logger.info("Kernel optimization system initialized successfully")
        else:
            logger.warning("Kernel optimization system initialized with limited capabilities")

        return kernel_optimizer
    except Exception as e:
        logger.error(f"Failed to initialize kernel optimizations: {e}")
        return None


def get_kernel_optimizer():
    """Get global kernel optimizer instance"""
    return kernel_optimizer