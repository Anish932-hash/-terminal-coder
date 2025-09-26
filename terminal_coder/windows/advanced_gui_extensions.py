#!/usr/bin/env python3
"""
Advanced GUI Extensions for Terminal Coder Windows
Cutting-edge features with real implementations optimized for Windows
"""

import asyncio
import json
import threading
import time
import numpy as np
import cv2
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import psutil
import sqlite3
import logging
from collections import deque
import subprocess
import os
import sys
import wmi
import win32api
import win32con
import win32gui
import win32process
import win32service
import win32serviceutil
from win32com.client import GetObject
import winreg

# Advanced imports for cutting-edge features
try:
    import torch
    import transformers
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import networkx as nx
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import win32evtlog
    import win32evtlogutil
    WIN32_EVTLOG_AVAILABLE = True
except ImportError:
    WIN32_EVTLOG_AVAILABLE = False


class WindowsRealTimeSystemMonitor:
    """Advanced real-time Windows system monitoring with ML-powered analytics"""

    def __init__(self, gui_parent):
        self.parent = gui_parent
        self.is_monitoring = False
        self.data_history = {
            'cpu': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'disk_io': deque(maxlen=100),
            'network_io': deque(maxlen=100),
            'processes': deque(maxlen=50),
            'timestamps': deque(maxlen=100)
        }
        self.anomaly_detector = None
        self.wmi_conn = None
        self.setup_windows_monitoring()
        self.setup_ml_models()

    def setup_windows_monitoring(self):
        """Setup Windows-specific monitoring components"""
        try:
            self.wmi_conn = wmi.WMI()
        except Exception as e:
            logging.error(f"WMI connection failed: {e}")

    def setup_ml_models(self):
        """Setup ML models for system analysis"""
        if ADVANCED_ML_AVAILABLE:
            # Simple anomaly detection model
            self.anomaly_threshold = 2.0
            self.baseline_metrics = {}

    async def start_monitoring(self, update_callback: Callable):
        """Start advanced real-time monitoring"""
        self.is_monitoring = True

        while self.is_monitoring:
            try:
                # Collect comprehensive system metrics
                metrics = await self._collect_advanced_windows_metrics()

                # Store historical data
                self._store_metrics(metrics)

                # Detect anomalies using ML
                anomalies = self._detect_anomalies(metrics)

                # Update GUI with real-time data
                update_callback(metrics, anomalies)

                # Adaptive monitoring frequency based on system load
                sleep_time = self._calculate_adaptive_sleep(metrics)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                await asyncio.sleep(1.0)

    async def _collect_advanced_windows_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive Windows system metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu': await self._get_cpu_metrics(),
            'memory': await self._get_memory_metrics(),
            'disk': await self._get_disk_metrics(),
            'network': await self._get_network_metrics(),
            'processes': await self._get_process_metrics(),
            'windows': await self._get_windows_specific_metrics()
        }

        # Advanced GPU metrics if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            metrics['gpu'] = [{'id': gpu.id, 'load': gpu.load * 100, 'memory': gpu.memoryUsed / gpu.memoryTotal * 100} for gpu in gpus]
        except ImportError:
            pass

        return metrics

    async def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get Windows CPU metrics using WMI and psutil"""
        cpu_metrics = {
            'percent': psutil.cpu_percent(interval=0.1),
            'per_cpu': psutil.cpu_percent(interval=0.1, percpu=True),
            'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'stats': psutil.cpu_stats()._asdict()
        }

        # Windows-specific CPU information via WMI
        if self.wmi_conn:
            try:
                for cpu in self.wmi_conn.Win32_Processor():
                    cpu_metrics.update({
                        'name': cpu.Name,
                        'manufacturer': cpu.Manufacturer,
                        'architecture': cpu.Architecture,
                        'cores': cpu.NumberOfCores,
                        'threads': cpu.NumberOfLogicalProcessors,
                        'max_clock_speed': cpu.MaxClockSpeed,
                        'current_clock_speed': cpu.CurrentClockSpeed,
                        'temperature': self._get_cpu_temperature()
                    })
                    break  # Take first CPU
            except Exception as e:
                logging.error(f"WMI CPU metrics error: {e}")

        return cpu_metrics

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature using WMI (if available)"""
        try:
            if self.wmi_conn:
                for sensor in self.wmi_conn.MSAcpi_ThermalZoneTemperature():
                    return (sensor.CurrentTemperature / 10.0) - 273.15  # Convert to Celsius
        except Exception:
            pass
        return None

    async def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get Windows memory metrics"""
        memory_metrics = {
            'virtual': psutil.virtual_memory()._asdict(),
            'swap': psutil.swap_memory()._asdict()
        }

        # Windows-specific memory information via WMI
        if self.wmi_conn:
            try:
                for mem in self.wmi_conn.Win32_PhysicalMemory():
                    memory_metrics.setdefault('physical_modules', []).append({
                        'capacity': int(mem.Capacity),
                        'speed': mem.Speed,
                        'manufacturer': mem.Manufacturer,
                        'part_number': mem.PartNumber,
                        'serial_number': mem.SerialNumber
                    })
            except Exception as e:
                logging.error(f"WMI memory metrics error: {e}")

        return memory_metrics

    async def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get Windows disk metrics"""
        disk_metrics = {
            'usage': {},
            'io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
        }

        # Get disk usage for all drives
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_metrics['usage'][partition.device] = usage._asdict()
            except PermissionError:
                continue

        # Windows-specific disk information via WMI
        if self.wmi_conn:
            try:
                disk_metrics['physical_disks'] = []
                for disk in self.wmi_conn.Win32_DiskDrive():
                    disk_metrics['physical_disks'].append({
                        'model': disk.Model,
                        'size': int(disk.Size) if disk.Size else 0,
                        'interface_type': disk.InterfaceType,
                        'media_type': disk.MediaType,
                        'status': disk.Status
                    })
            except Exception as e:
                logging.error(f"WMI disk metrics error: {e}")

        return disk_metrics

    async def _get_network_metrics(self) -> Dict[str, Any]:
        """Get Windows network metrics"""
        network_metrics = {
            'io': psutil.net_io_counters()._asdict(),
            'connections': len(psutil.net_connections()),
            'interfaces': {}
        }

        # Network interface information
        for name, addrs in psutil.net_if_addrs().items():
            network_metrics['interfaces'][name] = [addr._asdict() for addr in addrs]

        # Windows-specific network information via WMI
        if self.wmi_conn:
            try:
                network_metrics['adapters'] = []
                for adapter in self.wmi_conn.Win32_NetworkAdapter():
                    if adapter.PhysicalAdapter:
                        network_metrics['adapters'].append({
                            'name': adapter.Name,
                            'manufacturer': adapter.Manufacturer,
                            'mac_address': adapter.MACAddress,
                            'speed': adapter.Speed,
                            'adapter_type': adapter.AdapterType
                        })
            except Exception as e:
                logging.error(f"WMI network metrics error: {e}")

        return network_metrics

    async def _get_process_metrics(self) -> List[Dict[str, Any]]:
        """Get Windows process metrics"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'create_time', 'status', 'nice', 'num_threads']):
            try:
                proc_info = proc.info
                proc_info['cpu_percent'] = proc.cpu_percent(interval=0)
                proc_info['memory_mb'] = proc.memory_info().rss / 1024 / 1024

                # Windows-specific process information
                try:
                    proc_info['exe'] = proc.exe()
                    proc_info['cmdline'] = ' '.join(proc.cmdline())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

                processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU usage and return top processes
        return sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:20]

    async def _get_windows_specific_metrics(self) -> Dict[str, Any]:
        """Get Windows-specific system metrics"""
        windows_metrics = {
            'boot_time': psutil.boot_time(),
            'users': len(psutil.users()),
            'uptime': time.time() - psutil.boot_time(),
            'windows_version': self._get_windows_version(),
            'services': await self._get_services_status(),
            'event_log_summary': await self._get_event_log_summary()
        }

        # Registry information
        try:
            windows_metrics['registry_info'] = self._get_registry_info()
        except Exception as e:
            logging.error(f"Registry info error: {e}")

        return windows_metrics

    def _get_windows_version(self) -> Dict[str, str]:
        """Get Windows version information"""
        try:
            import platform
            return {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            }
        except Exception:
            return {}

    async def _get_services_status(self) -> Dict[str, int]:
        """Get Windows services status summary"""
        services_status = {'running': 0, 'stopped': 0, 'paused': 0, 'unknown': 0}

        try:
            if self.wmi_conn:
                for service in self.wmi_conn.Win32_Service():
                    state = service.State.lower() if service.State else 'unknown'
                    if state in services_status:
                        services_status[state] += 1
                    else:
                        services_status['unknown'] += 1
        except Exception as e:
            logging.error(f"Services status error: {e}")

        return services_status

    async def _get_event_log_summary(self) -> Dict[str, int]:
        """Get Windows Event Log summary"""
        event_summary = {'error': 0, 'warning': 0, 'information': 0}

        if not WIN32_EVTLOG_AVAILABLE:
            return event_summary

        try:
            # Check System event log for recent entries (last hour)
            server = 'localhost'
            logtype = 'System'
            hand = win32evtlog.OpenEventLog(server, logtype)

            flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
            total = win32evtlog.GetNumberOfEventLogRecords(hand)

            # Read last 100 events
            events_to_read = min(100, total)
            events = win32evtlog.ReadEventLog(hand, flags, events_to_read)

            one_hour_ago = time.time() - 3600

            for event in events:
                if event.TimeGenerated.timestamp() >= one_hour_ago:
                    if event.EventType == win32evtlog.EVENTLOG_ERROR_TYPE:
                        event_summary['error'] += 1
                    elif event.EventType == win32evtlog.EVENTLOG_WARNING_TYPE:
                        event_summary['warning'] += 1
                    elif event.EventType == win32evtlog.EVENTLOG_INFORMATION_TYPE:
                        event_summary['information'] += 1

            win32evtlog.CloseEventLog(hand)
        except Exception as e:
            logging.error(f"Event log summary error: {e}")

        return event_summary

    def _get_registry_info(self) -> Dict[str, Any]:
        """Get relevant Windows Registry information"""
        registry_info = {}

        try:
            # Get Windows version from registry
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                registry_info['windows_version'] = {
                    'product_name': winreg.QueryValueEx(key, "ProductName")[0],
                    'current_build': winreg.QueryValueEx(key, "CurrentBuild")[0],
                    'release_id': winreg.QueryValueEx(key, "ReleaseId")[0] if "ReleaseId" in [winreg.EnumValue(key, i)[0] for i in range(winreg.QueryInfoKey(key)[1])] else "Unknown"
                }
        except Exception as e:
            logging.error(f"Registry Windows version error: {e}")

        try:
            # Get system information
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                registry_info['cpu_info'] = {
                    'identifier': winreg.QueryValueEx(key, "Identifier")[0],
                    'processor_name_string': winreg.QueryValueEx(key, "ProcessorNameString")[0]
                }
        except Exception as e:
            logging.error(f"Registry CPU info error: {e}")

        return registry_info

    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in historical data with SQLite backend"""
        timestamp = metrics['timestamp']

        # Store in memory deques
        self.data_history['timestamps'].append(timestamp)
        self.data_history['cpu'].append(metrics['cpu']['percent'])
        self.data_history['memory'].append(metrics['memory']['virtual']['percent'])

        if metrics['disk']['io']:
            self.data_history['disk_io'].append(metrics['disk']['io'].get('read_bytes', 0) + metrics['disk']['io'].get('write_bytes', 0))

        self.data_history['network_io'].append(metrics['network']['io']['bytes_sent'] + metrics['network']['io']['bytes_recv'])
        self.data_history['processes'].append(len(metrics['processes']))

        # Store in SQLite for persistence (async)
        threading.Thread(target=self._store_in_database, args=(metrics,), daemon=True).start()

    def _store_in_database(self, metrics: Dict[str, Any]):
        """Store metrics in SQLite database"""
        try:
            db_path = Path(os.getenv('APPDATA')) / 'TerminalCoder' / 'metrics.db'
            db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS windows_system_metrics (
                    timestamp TEXT,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_io REAL,
                    network_io REAL,
                    process_count INTEGER,
                    cpu_temp REAL,
                    services_running INTEGER,
                    event_errors INTEGER,
                    event_warnings INTEGER
                )
            ''')

            # Insert metrics
            cursor.execute('''
                INSERT INTO windows_system_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics['timestamp'].isoformat(),
                metrics['cpu']['percent'],
                metrics['memory']['virtual']['percent'],
                sum(metrics['disk']['io'].values()) if metrics['disk']['io'] else 0,
                metrics['network']['io']['bytes_sent'] + metrics['network']['io']['bytes_recv'],
                len(metrics['processes']),
                metrics['cpu'].get('temperature', 0.0) or 0.0,
                metrics['windows']['services'].get('running', 0),
                metrics['windows']['event_log_summary'].get('error', 0),
                metrics['windows']['event_log_summary'].get('warning', 0)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"Database storage error: {e}")

    def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Advanced anomaly detection using statistical methods"""
        anomalies = []

        if len(self.data_history['cpu']) < 10:
            return anomalies

        try:
            # CPU anomaly detection
            cpu_data = list(self.data_history['cpu'])
            cpu_mean = np.mean(cpu_data)
            cpu_std = np.std(cpu_data)
            current_cpu = metrics['cpu']['percent']

            if abs(current_cpu - cpu_mean) > (self.anomaly_threshold * cpu_std):
                anomalies.append({
                    'type': 'cpu',
                    'severity': 'high' if current_cpu > cpu_mean + (3 * cpu_std) else 'medium',
                    'message': f'CPU usage anomaly detected: {current_cpu:.1f}% (normal: {cpu_mean:.1f}±{cpu_std:.1f}%)',
                    'timestamp': metrics['timestamp']
                })

            # Memory anomaly detection
            memory_data = list(self.data_history['memory'])
            memory_mean = np.mean(memory_data)
            memory_std = np.std(memory_data)
            current_memory = metrics['memory']['virtual']['percent']

            if abs(current_memory - memory_mean) > (self.anomaly_threshold * memory_std):
                anomalies.append({
                    'type': 'memory',
                    'severity': 'high' if current_memory > 90 else 'medium',
                    'message': f'Memory usage anomaly detected: {current_memory:.1f}% (normal: {memory_mean:.1f}±{memory_std:.1f}%)',
                    'timestamp': metrics['timestamp']
                })

            # Windows-specific anomaly detection
            error_count = metrics['windows']['event_log_summary'].get('error', 0)
            if error_count > 10:  # More than 10 errors in the last hour
                anomalies.append({
                    'type': 'windows_events',
                    'severity': 'high' if error_count > 20 else 'medium',
                    'message': f'High number of Windows errors detected: {error_count} in the last hour',
                    'timestamp': metrics['timestamp']
                })

            # CPU Temperature anomaly (if available)
            cpu_temp = metrics['cpu'].get('temperature')
            if cpu_temp and cpu_temp > 80:  # Above 80°C
                anomalies.append({
                    'type': 'temperature',
                    'severity': 'critical' if cpu_temp > 90 else 'high',
                    'message': f'High CPU temperature detected: {cpu_temp:.1f}°C',
                    'timestamp': metrics['timestamp']
                })

        except Exception as e:
            logging.error(f"Anomaly detection error: {e}")

        return anomalies

    def _calculate_adaptive_sleep(self, metrics: Dict[str, Any]) -> float:
        """Calculate adaptive monitoring frequency based on system load"""
        cpu_percent = metrics['cpu']['percent']
        memory_percent = metrics['memory']['virtual']['percent']

        # More frequent monitoring under high load
        if cpu_percent > 80 or memory_percent > 90:
            return 0.5  # 500ms
        elif cpu_percent > 50 or memory_percent > 70:
            return 1.0  # 1 second
        else:
            return 2.0  # 2 seconds

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False

    def get_historical_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical data from database"""
        try:
            db_path = Path(os.getenv('APPDATA')) / 'TerminalCoder' / 'metrics.db'
            if not db_path.exists():
                return {}

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            since_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            cursor.execute('''
                SELECT * FROM windows_system_metrics
                WHERE timestamp > ?
                ORDER BY timestamp
            ''', (since_time,))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {}

            # Convert to structured data
            return {
                'timestamps': [row[0] for row in rows],
                'cpu_percent': [row[1] for row in rows],
                'memory_percent': [row[2] for row in rows],
                'disk_io': [row[3] for row in rows],
                'network_io': [row[4] for row in rows],
                'process_count': [row[5] for row in rows],
                'cpu_temp': [row[6] for row in rows],
                'services_running': [row[7] for row in rows],
                'event_errors': [row[8] for row in rows],
                'event_warnings': [row[9] for row in rows]
            }

        except Exception as e:
            logging.error(f"Historical data retrieval error: {e}")
            return {}


class WindowsAdvancedVisualizationPanel:
    """Advanced Windows data visualization with interactive charts"""

    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.figures = {}
        self.canvases = {}
        self.setup_matplotlib_style()
        self.create_visualization_tabs()

    def setup_matplotlib_style(self):
        """Setup modern matplotlib styling for Windows"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def create_visualization_tabs(self):
        """Create tabbed visualization interface"""
        self.viz_notebook = ttk.Notebook(self.parent)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True)

        # Real-time monitoring tab
        self.realtime_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.realtime_frame, text="📊 Real-time")
        self.create_realtime_charts()

        # Historical analysis tab
        self.historical_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.historical_frame, text="📈 Historical")
        self.create_historical_charts()

        # Windows system tab
        self.windows_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.windows_frame, text="🪟 Windows")
        self.create_windows_specific_charts()

        # Performance analysis tab
        self.performance_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.performance_frame, text="⚡ Performance")
        self.create_performance_analysis()

        # ML Analysis tab
        if ADVANCED_ML_AVAILABLE:
            self.ml_frame = ttk.Frame(self.viz_notebook)
            self.viz_notebook.add(self.ml_frame, text="🤖 ML Analysis")
            self.create_ml_analysis()

    def create_realtime_charts(self):
        """Create real-time monitoring charts for Windows"""
        # Create figure with subplots
        self.figures['realtime'] = Figure(figsize=(14, 10), facecolor='#2d3142')

        # CPU and Memory subplot
        self.cpu_memory_ax = self.figures['realtime'].add_subplot(221)
        self.cpu_memory_ax.set_title('CPU & Memory Usage', color='white')
        self.cpu_memory_ax.set_facecolor('#2d3142')
        self.cpu_memory_ax.tick_params(colors='white')

        # Network I/O subplot
        self.network_ax = self.figures['realtime'].add_subplot(222)
        self.network_ax.set_title('Network I/O', color='white')
        self.network_ax.set_facecolor('#2d3142')
        self.network_ax.tick_params(colors='white')

        # Windows Services subplot
        self.services_ax = self.figures['realtime'].add_subplot(223)
        self.services_ax.set_title('Windows Services Status', color='white')
        self.services_ax.set_facecolor('#2d3142')
        self.services_ax.tick_params(colors='white')

        # Temperature and Events subplot
        self.temp_events_ax = self.figures['realtime'].add_subplot(224)
        self.temp_events_ax.set_title('Temperature & Event Log', color='white')
        self.temp_events_ax.set_facecolor('#2d3142')
        self.temp_events_ax.tick_params(colors='white')

        self.figures['realtime'].tight_layout()

        # Create canvas
        self.canvases['realtime'] = FigureCanvasTkAgg(self.figures['realtime'], self.realtime_frame)
        self.canvases['realtime'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_historical_charts(self):
        """Create historical data analysis charts"""
        self.figures['historical'] = Figure(figsize=(14, 10), facecolor='#2d3142')

        # Time series analysis
        self.timeseries_ax = self.figures['historical'].add_subplot(311)
        self.timeseries_ax.set_title('24-Hour Windows System Metrics', color='white')
        self.timeseries_ax.set_facecolor('#2d3142')
        self.timeseries_ax.tick_params(colors='white')

        # Correlation heatmap
        self.correlation_ax = self.figures['historical'].add_subplot(312)
        self.correlation_ax.set_title('Metrics Correlation Matrix', color='white')
        self.correlation_ax.set_facecolor('#2d3142')
        self.correlation_ax.tick_params(colors='white')

        # Windows-specific trends
        self.windows_trends_ax = self.figures['historical'].add_subplot(313)
        self.windows_trends_ax.set_title('Windows-Specific Metrics Trends', color='white')
        self.windows_trends_ax.set_facecolor('#2d3142')
        self.windows_trends_ax.tick_params(colors='white')

        self.figures['historical'].tight_layout()

        self.canvases['historical'] = FigureCanvasTkAgg(self.figures['historical'], self.historical_frame)
        self.canvases['historical'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_windows_specific_charts(self):
        """Create Windows-specific visualization charts"""
        self.figures['windows'] = Figure(figsize=(14, 10), facecolor='#2d3142')

        # Services distribution
        self.services_dist_ax = self.figures['windows'].add_subplot(221)
        self.services_dist_ax.set_title('Windows Services Distribution', color='white')
        self.services_dist_ax.set_facecolor('#2d3142')
        self.services_dist_ax.tick_params(colors='white')

        # Event log analysis
        self.events_ax = self.figures['windows'].add_subplot(222)
        self.events_ax.set_title('Event Log Analysis', color='white')
        self.events_ax.set_facecolor('#2d3142')
        self.events_ax.tick_params(colors='white')

        # Registry monitoring
        self.registry_ax = self.figures['windows'].add_subplot(223)
        self.registry_ax.set_title('System Information', color='white')
        self.registry_ax.set_facecolor('#2d3142')
        self.registry_ax.tick_params(colors='white')

        # Hardware monitoring
        self.hardware_ax = self.figures['windows'].add_subplot(224)
        self.hardware_ax.set_title('Hardware Status', color='white')
        self.hardware_ax.set_facecolor('#2d3142')
        self.hardware_ax.tick_params(colors='white')

        self.figures['windows'].tight_layout()

        self.canvases['windows'] = FigureCanvasTkAgg(self.figures['windows'], self.windows_frame)
        self.canvases['windows'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_performance_analysis(self):
        """Create performance analysis charts"""
        self.figures['performance'] = Figure(figsize=(14, 10), facecolor='#2d3142')

        # CPU performance trends
        self.cpu_perf_ax = self.figures['performance'].add_subplot(221)
        self.cpu_perf_ax.set_title('CPU Performance Analysis', color='white')
        self.cpu_perf_ax.set_facecolor('#2d3142')
        self.cpu_perf_ax.tick_params(colors='white')

        # Memory performance
        self.memory_perf_ax = self.figures['performance'].add_subplot(222)
        self.memory_perf_ax.set_title('Memory Performance Analysis', color='white')
        self.memory_perf_ax.set_facecolor('#2d3142')
        self.memory_perf_ax.tick_params(colors='white')

        # Disk I/O performance
        self.disk_perf_ax = self.figures['performance'].add_subplot(223)
        self.disk_perf_ax.set_title('Disk I/O Performance', color='white')
        self.disk_perf_ax.set_facecolor('#2d3142')
        self.disk_perf_ax.tick_params(colors='white')

        # Network performance
        self.network_perf_ax = self.figures['performance'].add_subplot(224)
        self.network_perf_ax.set_title('Network Performance', color='white')
        self.network_perf_ax.set_facecolor('#2d3142')
        self.network_perf_ax.tick_params(colors='white')

        self.figures['performance'].tight_layout()

        self.canvases['performance'] = FigureCanvasTkAgg(self.figures['performance'], self.performance_frame)
        self.canvases['performance'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_ml_analysis(self):
        """Create ML-based analysis visualizations"""
        if not ADVANCED_ML_AVAILABLE:
            return

        self.figures['ml'] = Figure(figsize=(14, 10), facecolor='#2d3142')

        # Anomaly detection plot
        self.anomaly_ax = self.figures['ml'].add_subplot(221)
        self.anomaly_ax.set_title('Windows System Anomaly Detection', color='white')
        self.anomaly_ax.set_facecolor('#2d3142')
        self.anomaly_ax.tick_params(colors='white')

        # Process behavior clustering
        self.cluster_ax = self.figures['ml'].add_subplot(222)
        self.cluster_ax.set_title('Process Behavior Clustering', color='white')
        self.cluster_ax.set_facecolor('#2d3142')
        self.cluster_ax.tick_params(colors='white')

        # Performance prediction
        self.prediction_ax = self.figures['ml'].add_subplot(223)
        self.prediction_ax.set_title('Windows Performance Prediction', color='white')
        self.prediction_ax.set_facecolor('#2d3142')
        self.prediction_ax.tick_params(colors='white')

        # System optimization suggestions
        self.optimization_ax = self.figures['ml'].add_subplot(224)
        self.optimization_ax.set_title('Windows Optimization Recommendations', color='white')
        self.optimization_ax.set_facecolor('#2d3142')
        self.optimization_ax.tick_params(colors='white')

        self.figures['ml'].tight_layout()

        self.canvases['ml'] = FigureCanvasTkAgg(self.figures['ml'], self.ml_frame)
        self.canvases['ml'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_realtime_charts(self, metrics: Dict[str, Any], anomalies: List[Dict[str, Any]]):
        """Update real-time charts with new Windows data"""
        try:
            # Clear previous plots
            for ax in [self.cpu_memory_ax, self.network_ax, self.services_ax, self.temp_events_ax]:
                ax.clear()

            # CPU & Memory chart
            timestamps = list(range(len(metrics.get('cpu_history', [0]))))
            cpu_data = metrics.get('cpu_history', [0])
            memory_data = metrics.get('memory_history', [0])

            self.cpu_memory_ax.plot(timestamps, cpu_data, label='CPU %', color='#e76f51', linewidth=2)
            self.cpu_memory_ax.plot(timestamps, memory_data, label='Memory %', color='#2a9d8f', linewidth=2)
            self.cpu_memory_ax.set_ylim(0, 100)
            self.cpu_memory_ax.legend()
            self.cpu_memory_ax.grid(True, alpha=0.3)
            self.cpu_memory_ax.set_title('CPU & Memory Usage', color='white')

            # Network I/O chart
            if 'network_history' in metrics:
                network_data = metrics['network_history']
                self.network_ax.bar(timestamps[-len(network_data):], network_data, color='#4f5d75', alpha=0.7)
                self.network_ax.set_ylabel('Bytes/sec', color='white')
                self.network_ax.set_title('Network I/O', color='white')

            # Windows Services pie chart
            if 'services_status' in metrics:
                services = metrics['services_status']
                labels = list(services.keys())
                sizes = list(services.values())
                colors = ['#2a9d8f', '#e76f51', '#e9c46a', '#4f5d75']

                self.services_ax.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%')
                self.services_ax.set_title('Windows Services Status', color='white')

            # Temperature and Events
            temp_data = metrics.get('temperature_history', [])
            error_data = metrics.get('event_error_history', [])

            if temp_data:
                temp_ax = self.temp_events_ax
                temp_line = temp_ax.plot(timestamps[-len(temp_data):], temp_data,
                                       color='#e76f51', linewidth=2, label='CPU Temp (°C)')
                temp_ax.set_ylabel('Temperature (°C)', color='#e76f51')
                temp_ax.tick_params(axis='y', labelcolor='#e76f51')

                # Create second y-axis for errors
                if error_data:
                    error_ax = temp_ax.twinx()
                    error_bars = error_ax.bar(timestamps[-len(error_data):], error_data,
                                            color='#e9c46a', alpha=0.7, label='Event Errors')
                    error_ax.set_ylabel('Event Errors/Hour', color='#e9c46a')
                    error_ax.tick_params(axis='y', labelcolor='#e9c46a')

                self.temp_events_ax.set_title('Temperature & Event Log', color='white')

            # Refresh canvas
            self.canvases['realtime'].draw()

        except Exception as e:
            logging.error(f"Chart update error: {e}")


class WindowsIntelligentCodeAnalyzer:
    """Advanced Windows code analysis with ML-powered insights"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.setup_analysis_models()

    def setup_analysis_models(self):
        """Setup ML models for code analysis"""
        if ADVANCED_ML_AVAILABLE:
            # Initialize TF-IDF vectorizer for code similarity
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 3)
            )

            # Clustering model for code patterns
            self.clustering_model = KMeans(n_clusters=5, random_state=42)

    async def analyze_windows_project(self, project_path: str) -> Dict[str, Any]:
        """Comprehensive Windows project analysis"""
        analysis_results = {
            'summary': {},
            'complexity': {},
            'dependencies': {},
            'security': {},
            'quality': {},
            'patterns': {},
            'windows_specific': {},
            'recommendations': []
        }

        try:
            project_path = Path(project_path)

            # Basic metrics
            analysis_results['summary'] = await self._analyze_basic_metrics(project_path)

            # Code complexity analysis
            analysis_results['complexity'] = await self._analyze_complexity(project_path)

            # Dependency analysis
            analysis_results['dependencies'] = await self._analyze_dependencies(project_path)

            # Security analysis
            analysis_results['security'] = await self._analyze_security(project_path)

            # Code quality metrics
            analysis_results['quality'] = await self._analyze_quality(project_path)

            # Windows-specific analysis
            analysis_results['windows_specific'] = await self._analyze_windows_specific(project_path)

            # Pattern recognition
            if ADVANCED_ML_AVAILABLE:
                analysis_results['patterns'] = await self._analyze_patterns(project_path)

            # Generate recommendations
            analysis_results['recommendations'] = self._generate_windows_recommendations(analysis_results)

        except Exception as e:
            logging.error(f"Windows code analysis error: {e}")
            analysis_results['error'] = str(e)

        return analysis_results

    async def _analyze_windows_specific(self, project_path: Path) -> Dict[str, Any]:
        """Analyze Windows-specific code patterns and issues"""
        windows_analysis = {
            'win32_api_usage': [],
            'registry_operations': [],
            'service_interactions': [],
            'com_objects': [],
            'powershell_usage': [],
            'windows_paths': [],
            'compatibility_issues': []
        }

        # Windows-specific patterns to look for
        windows_patterns = {
            'win32_api': [
                r'win32api\.',
                r'win32con\.',
                r'win32gui\.',
                r'win32service\.',
                r'win32process\.',
                r'win32file\.'
            ],
            'registry': [
                r'winreg\.',
                r'_winreg\.',
                r'OpenKey\(',
                r'QueryValueEx\(',
                r'SetValueEx\(',
                r'HKEY_LOCAL_MACHINE',
                r'HKEY_CURRENT_USER'
            ],
            'com_objects': [
                r'win32com\.client',
                r'GetObject\(',
                r'Dispatch\(',
                r'CoInitialize\(',
                r'CoUninitialize\('
            ],
            'services': [
                r'win32service\.',
                r'StartService\(',
                r'StopService\(',
                r'QueryServiceStatus\('
            ],
            'powershell': [
                r'powershell\.exe',
                r'subprocess.*powershell',
                r'PowerShell\(',
                r'Invoke-Command'
            ]
        }

        import re

        # Analyze Python files for Windows-specific patterns
        for py_file in project_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Check for Windows-specific API usage
                    for category, patterns in windows_patterns.items():
                        for pattern in patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1

                                if category == 'win32_api':
                                    windows_analysis['win32_api_usage'].append({
                                        'file': str(py_file),
                                        'line': line_num,
                                        'pattern': pattern,
                                        'code': match.group(0)
                                    })
                                elif category == 'registry':
                                    windows_analysis['registry_operations'].append({
                                        'file': str(py_file),
                                        'line': line_num,
                                        'pattern': pattern,
                                        'code': match.group(0)
                                    })
                                elif category == 'com_objects':
                                    windows_analysis['com_objects'].append({
                                        'file': str(py_file),
                                        'line': line_num,
                                        'pattern': pattern,
                                        'code': match.group(0)
                                    })
                                elif category == 'services':
                                    windows_analysis['service_interactions'].append({
                                        'file': str(py_file),
                                        'line': line_num,
                                        'pattern': pattern,
                                        'code': match.group(0)
                                    })
                                elif category == 'powershell':
                                    windows_analysis['powershell_usage'].append({
                                        'file': str(py_file),
                                        'line': line_num,
                                        'pattern': pattern,
                                        'code': match.group(0)
                                    })

                    # Check for Windows path usage
                    windows_path_patterns = [
                        r'C:\\\\',
                        r'%APPDATA%',
                        r'%USERPROFILE%',
                        r'%PROGRAMFILES%',
                        r'%TEMP%',
                        r'%WINDIR%'
                    ]

                    for pattern in windows_path_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            windows_analysis['windows_paths'].append({
                                'file': str(py_file),
                                'line': line_num,
                                'path': match.group(0)
                            })

                    # Check for potential compatibility issues
                    compatibility_patterns = [
                        (r'os\.system\(.*["\'].*\.exe.*["\']', 'Direct exe execution may fail on different Windows versions'),
                        (r'subprocess.*shell=True.*["\'].*\.bat.*["\']', 'Batch file execution may have compatibility issues'),
                        (r'import.*pwd', 'pwd module not available on Windows'),
                        (r'import.*grp', 'grp module not available on Windows'),
                        (r'os\.fork\(\)', 'fork() not available on Windows'),
                        (r'/[^/]*/', 'Unix-style paths may not work on Windows')
                    ]

                    for pattern, issue in compatibility_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            windows_analysis['compatibility_issues'].append({
                                'file': str(py_file),
                                'line': line_num,
                                'issue': issue,
                                'code': match.group(0)
                            })

            except Exception:
                continue

        return windows_analysis

    async def _analyze_basic_metrics(self, project_path: Path) -> Dict[str, Any]:
        """Analyze basic code metrics with Windows focus"""
        metrics = {
            'total_files': 0,
            'total_lines': 0,
            'languages': {},
            'file_sizes': [],
            'largest_files': [],
            'windows_specific_files': 0
        }

        language_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.cs': 'C#',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C Header',
            '.css': 'CSS',
            '.html': 'HTML',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.md': 'Markdown',
            '.bat': 'Batch',
            '.cmd': 'Command',
            '.ps1': 'PowerShell',
            '.vbs': 'VBScript',
            '.reg': 'Registry',
            '.ini': 'Configuration'
        }

        windows_specific_extensions = {'.bat', '.cmd', '.ps1', '.vbs', '.reg'}

        for file_path in project_path.rglob('*'):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts[len(project_path.parts):]):
                try:
                    metrics['total_files'] += 1
                    file_size = file_path.stat().st_size
                    metrics['file_sizes'].append(file_size)

                    # Language detection
                    ext = file_path.suffix.lower()
                    if ext in language_extensions:
                        lang = language_extensions[ext]
                        metrics['languages'][lang] = metrics['languages'].get(lang, 0) + 1

                        # Count Windows-specific files
                        if ext in windows_specific_extensions:
                            metrics['windows_specific_files'] += 1

                        # Count lines for text files
                        if ext in ['.py', '.js', '.ts', '.cs', '.cpp', '.c', '.ps1', '.bat', '.cmd']:
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    line_count = len(f.readlines())
                                    metrics['total_lines'] += line_count

                                    # Track largest files
                                    if len(metrics['largest_files']) < 10:
                                        metrics['largest_files'].append((str(file_path), line_count, file_size))
                                    else:
                                        metrics['largest_files'].sort(key=lambda x: x[1], reverse=True)
                                        if line_count > metrics['largest_files'][-1][1]:
                                            metrics['largest_files'][-1] = (str(file_path), line_count, file_size)
                            except Exception:
                                pass

                except Exception:
                    continue

        # Calculate statistics
        if metrics['file_sizes']:
            metrics['avg_file_size'] = np.mean(metrics['file_sizes'])
            metrics['median_file_size'] = np.median(metrics['file_sizes'])
            metrics['max_file_size'] = max(metrics['file_sizes'])

        return metrics

    async def _analyze_complexity(self, project_path: Path) -> Dict[str, Any]:
        """Analyze code complexity with Windows considerations"""
        complexity = {
            'cyclomatic_complexity': [],
            'function_lengths': [],
            'class_counts': {},
            'nesting_levels': [],
            'windows_api_complexity': 0
        }

        # Analyze Python files for complexity
        for py_file in project_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Count control flow statements (rough cyclomatic complexity)
                    control_statements = content.count('if ') + content.count('elif ') + \
                                       content.count('while ') + content.count('for ') + \
                                       content.count('except ') + content.count('with ')

                    complexity['cyclomatic_complexity'].append(control_statements)

                    # Count Windows API calls (adds to complexity)
                    windows_api_calls = content.count('win32') + content.count('winreg') + \
                                      content.count('wmi.') + content.count('COM')
                    complexity['windows_api_complexity'] += windows_api_calls

                    # Analyze functions and classes
                    lines = content.split('\n')
                    current_function_length = 0
                    nesting_level = 0
                    max_nesting = 0

                    for line in lines:
                        stripped = line.lstrip()
                        indent_level = (len(line) - len(stripped)) // 4

                        if stripped.startswith('def '):
                            if current_function_length > 0:
                                complexity['function_lengths'].append(current_function_length)
                            current_function_length = 1
                        elif stripped and not stripped.startswith('#'):
                            if current_function_length > 0:
                                current_function_length += 1

                        # Track nesting
                        if stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'with ', 'def ', 'class ')):
                            nesting_level = indent_level
                            max_nesting = max(max_nesting, nesting_level)

                    complexity['nesting_levels'].append(max_nesting)

                    # Count classes
                    class_count = content.count('class ')
                    if class_count > 0:
                        complexity['class_counts'][str(py_file)] = class_count

            except Exception:
                continue

        # Analyze PowerShell files
        for ps_file in project_path.rglob('*.ps1'):
            try:
                with open(ps_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # PowerShell complexity indicators
                    ps_complexity = content.count('if (') + content.count('while (') + \
                                  content.count('foreach ') + content.count('switch ')
                    complexity['cyclomatic_complexity'].append(ps_complexity)

            except Exception:
                continue

        # Calculate complexity statistics
        if complexity['cyclomatic_complexity']:
            complexity['avg_complexity'] = np.mean(complexity['cyclomatic_complexity'])
            complexity['max_complexity'] = max(complexity['cyclomatic_complexity'])

        if complexity['function_lengths']:
            complexity['avg_function_length'] = np.mean(complexity['function_lengths'])
            complexity['max_function_length'] = max(complexity['function_lengths'])

        return complexity

    async def _analyze_dependencies(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project dependencies with Windows focus"""
        dependencies = {
            'package_files': [],
            'imports': {},
            'dependency_tree': {},
            'windows_dependencies': [],
            'vulnerabilities': []
        }

        # Check for dependency files
        dep_files = ['requirements.txt', 'package.json', 'packages.config', 'project.json', 'Pipfile']

        for dep_file in dep_files:
            file_path = project_path / dep_file
            if file_path.exists():
                dependencies['package_files'].append(dep_file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        dependencies['dependency_tree'][dep_file] = content[:1000]  # First 1000 chars
                except Exception:
                    pass

        # Analyze Python imports with Windows focus
        windows_modules = {
            'win32api', 'win32con', 'win32gui', 'win32service', 'win32process',
            'win32file', 'win32security', 'winreg', '_winreg', 'wmi', 'pythoncom',
            'win32com', 'pywintypes', 'win32clipboard', 'win32pipe'
        }

        for py_file in project_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Extract import statements
                    import_lines = [line.strip() for line in content.split('\n')
                                  if line.strip().startswith(('import ', 'from '))]

                    for import_line in import_lines:
                        # Parse import
                        if import_line.startswith('import '):
                            module = import_line.replace('import ', '').split('.')[0].split(',')[0].strip()
                        elif import_line.startswith('from '):
                            module = import_line.split('from ')[1].split(' import')[0].split('.')[0].strip()
                        else:
                            continue

                        if module:
                            dependencies['imports'][module] = dependencies['imports'].get(module, 0) + 1

                            # Track Windows-specific dependencies
                            if module in windows_modules:
                                dependencies['windows_dependencies'].append({
                                    'module': module,
                                    'file': str(py_file),
                                    'count': dependencies['imports'][module]
                                })

            except Exception:
                continue

        return dependencies

    async def _analyze_security(self, project_path: Path) -> Dict[str, Any]:
        """Analyze security with Windows-specific considerations"""
        security = {
            'potential_issues': [],
            'sensitive_files': [],
            'hardcoded_secrets': [],
            'windows_security_issues': [],
            'security_score': 0
        }

        # Windows-specific security patterns
        windows_security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
            (r'eval\s*\(', 'Use of eval() function'),
            (r'exec\s*\(', 'Use of exec() function'),
            (r'os\.system\s*\(', 'Use of os.system()'),
            (r'subprocess\.call.*shell=True', 'Shell injection risk'),
            (r'win32api\.ShellExecute', 'Potential code execution risk'),
            (r'winreg\.SetValueEx.*HKEY_LOCAL_MACHINE', 'Registry modification with admin rights'),
            (r'win32service\.CreateService', 'Service creation (requires admin)'),
            (r'ctypes\.windll', 'Direct DLL access (security risk)'),
            (r'powershell\.exe.*-ExecutionPolicy\s+Bypass', 'PowerShell execution policy bypass'),
            (r'cmd\.exe.*\/c', 'Command execution through cmd')
        ]

        import re

        # Check for sensitive files
        sensitive_files = ['.env', 'config.ini', 'settings.py', 'secrets.json', 'web.config', '*.pfx', '*.p12']
        for sensitive_file in sensitive_files:
            for file_path in project_path.rglob(sensitive_file):
                security['sensitive_files'].append(str(file_path))

        # Scan code for security issues
        for code_file in project_path.rglob('*.py'):
            try:
                with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    for pattern, description in windows_security_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1

                            issue_data = {
                                'file': str(code_file),
                                'line': line_num,
                                'issue': description,
                                'code': match.group(0)
                            }

                            if 'win32' in description or 'Registry' in description or 'PowerShell' in description or 'cmd' in description:
                                security['windows_security_issues'].append(issue_data)
                            else:
                                security['potential_issues'].append(issue_data)

            except Exception:
                continue

        # Calculate security score
        issue_count = len(security['potential_issues']) + len(security['windows_security_issues'])
        sensitive_count = len(security['sensitive_files'])

        # Higher score is better (0-100), with extra penalty for Windows-specific issues
        windows_penalty = len(security['windows_security_issues']) * 15
        security['security_score'] = max(0, 100 - (issue_count * 10) - (sensitive_count * 5) - windows_penalty)

        return security

    async def _analyze_quality(self, project_path: Path) -> Dict[str, Any]:
        """Analyze code quality with Windows considerations"""
        quality = {
            'documentation_coverage': 0,
            'test_coverage_estimate': 0,
            'code_duplication': [],
            'naming_conventions': {},
            'windows_best_practices': {},
            'quality_score': 0
        }

        total_functions = 0
        documented_functions = 0
        test_files = 0
        total_code_files = 0
        windows_best_practices_score = 0

        # Analyze Python files for quality metrics
        for py_file in project_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    total_code_files += 1

                    # Check if it's a test file
                    if 'test' in py_file.name.lower() or py_file.parent.name.lower() in ['test', 'tests']:
                        test_files += 1

                    # Count functions and documentation
                    lines = content.split('\n')
                    in_function = False
                    function_has_docstring = False

                    for i, line in enumerate(lines):
                        stripped = line.strip()

                        if stripped.startswith('def '):
                            if in_function and function_has_docstring:
                                documented_functions += 1
                            total_functions += 1
                            in_function = True
                            function_has_docstring = False
                        elif in_function and stripped.startswith('"""'):
                            function_has_docstring = True
                        elif stripped.startswith('class ') or (stripped.startswith('def ') and in_function):
                            if in_function and function_has_docstring:
                                documented_functions += 1
                            in_function = stripped.startswith('def ')
                            function_has_docstring = False

                    # Check last function
                    if in_function and function_has_docstring:
                        documented_functions += 1

                    # Check Windows best practices
                    if 'win32' in content or 'winreg' in content or 'wmi' in content:
                        # Check for proper error handling around Windows APIs
                        if 'try:' in content and 'except' in content:
                            windows_best_practices_score += 10

                        # Check for proper resource cleanup
                        if 'finally:' in content or 'with ' in content:
                            windows_best_practices_score += 5

                        # Check for Windows version compatibility checks
                        if 'platform.system()' in content or 'sys.platform' in content:
                            windows_best_practices_score += 15

            except Exception:
                continue

        # Analyze PowerShell files
        for ps_file in project_path.rglob('*.ps1'):
            total_code_files += 1
            try:
                with open(ps_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # PowerShell quality indicators
                    if 'param(' in content.lower():
                        windows_best_practices_score += 10

                    if 'try {' in content and 'catch' in content:
                        windows_best_practices_score += 10

            except Exception:
                continue

        # Calculate metrics
        if total_functions > 0:
            quality['documentation_coverage'] = (documented_functions / total_functions) * 100

        if total_code_files > 0:
            quality['test_coverage_estimate'] = (test_files / total_code_files) * 100

        quality['windows_best_practices'] = {
            'score': windows_best_practices_score,
            'max_score': total_code_files * 40,  # Maximum possible score
            'percentage': (windows_best_practices_score / max(total_code_files * 40, 1)) * 100
        }

        # Calculate overall quality score with Windows considerations
        quality['quality_score'] = (
            quality['documentation_coverage'] * 0.3 +
            quality['test_coverage_estimate'] * 0.4 +
            quality['windows_best_practices']['percentage'] * 0.3
        )

        return quality

    async def _analyze_patterns(self, project_path: Path) -> Dict[str, Any]:
        """Analyze code patterns using ML with Windows focus"""
        if not ADVANCED_ML_AVAILABLE:
            return {}

        patterns = {
            'design_patterns': [],
            'code_clusters': [],
            'similarity_matrix': [],
            'windows_patterns': []
        }

        try:
            # Collect code snippets
            code_snippets = []
            file_paths = []

            for py_file in list(project_path.rglob('*.py'))[:50]:  # Limit for performance
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if len(content.strip()) > 100:  # Skip very small files
                            code_snippets.append(content)
                            file_paths.append(str(py_file))
                except Exception:
                    continue

            # Include PowerShell files
            for ps_file in list(project_path.rglob('*.ps1'))[:20]:
                try:
                    with open(ps_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if len(content.strip()) > 50:
                            code_snippets.append(content)
                            file_paths.append(str(ps_file))
                except Exception:
                    continue

            if len(code_snippets) > 1:
                # Vectorize code
                tfidf_matrix = self.vectorizer.fit_transform(code_snippets)

                # Perform clustering
                if len(code_snippets) >= 5:
                    n_clusters = min(5, len(code_snippets) // 2)
                    self.clustering_model.n_clusters = n_clusters
                    clusters = self.clustering_model.fit_predict(tfidf_matrix)

                    # Group files by cluster
                    for i, cluster in enumerate(clusters):
                        patterns['code_clusters'].append({
                            'file': file_paths[i],
                            'cluster': int(cluster),
                            'is_windows_specific': 'win32' in code_snippets[i] or '.ps1' in file_paths[i]
                        })

                # Calculate similarity matrix (for small datasets)
                if len(code_snippets) <= 10:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    patterns['similarity_matrix'] = similarity_matrix.tolist()

                # Identify Windows-specific patterns
                windows_keywords = ['win32', 'winreg', 'wmi', 'COM', 'powershell', 'registry']
                for i, snippet in enumerate(code_snippets):
                    windows_score = sum(snippet.lower().count(keyword) for keyword in windows_keywords)
                    if windows_score > 5:
                        patterns['windows_patterns'].append({
                            'file': file_paths[i],
                            'windows_api_density': windows_score,
                            'type': 'High Windows API usage'
                        })

        except Exception as e:
            logging.error(f"Pattern analysis error: {e}")

        return patterns

    def _generate_windows_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate intelligent recommendations for Windows development"""
        recommendations = []

        # Code quality recommendations
        quality = analysis_results.get('quality', {})
        if quality.get('documentation_coverage', 0) < 50:
            recommendations.append({
                'type': 'documentation',
                'priority': 'high',
                'title': 'Improve Documentation Coverage',
                'description': f"Only {quality.get('documentation_coverage', 0):.1f}% of functions have docstrings. Consider adding comprehensive documentation.",
                'action': 'Add docstrings to functions and classes'
            })

        # Windows-specific recommendations
        windows_specific = analysis_results.get('windows_specific', {})
        if windows_specific.get('compatibility_issues'):
            recommendations.append({
                'type': 'compatibility',
                'priority': 'high',
                'title': 'Address Windows Compatibility Issues',
                'description': f"Found {len(windows_specific.get('compatibility_issues', []))} potential compatibility issues.",
                'action': 'Review and fix platform-specific code'
            })

        if windows_specific.get('registry_operations'):
            recommendations.append({
                'type': 'security',
                'priority': 'medium',
                'title': 'Registry Operations Detected',
                'description': f"Found {len(windows_specific.get('registry_operations', []))} registry operations. Ensure proper error handling and security.",
                'action': 'Add proper exception handling for registry operations'
            })

        # Security recommendations
        security = analysis_results.get('security', {})
        if security.get('windows_security_issues'):
            recommendations.append({
                'type': 'security',
                'priority': 'critical',
                'title': 'Windows-Specific Security Issues',
                'description': f"Found {len(security.get('windows_security_issues', []))} Windows-specific security issues.",
                'action': 'Review Windows API usage and security implications'
            })

        if security.get('security_score', 100) < 70:
            recommendations.append({
                'type': 'security',
                'priority': 'critical',
                'title': 'Address Security Issues',
                'description': f"Security score is {security.get('security_score', 0)}/100. Found {len(security.get('potential_issues', []))} potential security issues.",
                'action': 'Review and fix security vulnerabilities'
            })

        # Windows best practices
        windows_bp = quality.get('windows_best_practices', {})
        if windows_bp.get('percentage', 0) < 60:
            recommendations.append({
                'type': 'best_practices',
                'priority': 'medium',
                'title': 'Improve Windows Development Best Practices',
                'description': f"Windows best practices score: {windows_bp.get('percentage', 0):.1f}%",
                'action': 'Add proper error handling, resource cleanup, and platform checks'
            })

        # Complexity recommendations
        complexity = analysis_results.get('complexity', {})
        if complexity.get('windows_api_complexity', 0) > 50:
            recommendations.append({
                'type': 'refactoring',
                'priority': 'medium',
                'title': 'High Windows API Complexity',
                'description': f"High number of Windows API calls ({complexity.get('windows_api_complexity', 0)}) may indicate complex code.",
                'action': 'Consider abstracting Windows API calls into separate modules'
            })

        return recommendations


class WindowsSmartContainerOrchestrator:
    """Advanced Windows container orchestration with Docker Desktop integration"""

    def __init__(self):
        self.docker_client = None
        self.setup_clients()

    def setup_clients(self):
        """Setup container orchestration clients for Windows"""
        if DOCKER_AVAILABLE:
            try:
                # Docker client (Docker Desktop on Windows)
                self.docker_client = docker.from_env()

            except Exception as e:
                logging.warning(f"Container client setup failed: {e}")

    async def analyze_windows_container_infrastructure(self) -> Dict[str, Any]:
        """Analyze Windows container infrastructure"""
        analysis = {
            'docker': {
                'containers': [],
                'images': [],
                'networks': [],
                'volumes': [],
                'system_info': {}
            },
            'windows_containers': {
                'windows_containers': [],
                'isolation_modes': {},
                'base_images': []
            },
            'recommendations': []
        }

        if not self.docker_client:
            analysis['error'] = 'Docker client not available. Install Docker Desktop for Windows.'
            return analysis

        try:
            # Docker analysis
            analysis['docker'] = await self._analyze_docker_windows()

            # Windows-specific container analysis
            analysis['windows_containers'] = await self._analyze_windows_containers()

            # Generate recommendations
            analysis['recommendations'] = self._generate_windows_container_recommendations(analysis)

        except Exception as e:
            analysis['error'] = str(e)
            logging.error(f"Windows container analysis error: {e}")

        return analysis

    async def _analyze_docker_windows(self) -> Dict[str, Any]:
        """Analyze Docker on Windows"""
        docker_analysis = {
            'containers': [],
            'images': [],
            'networks': [],
            'volumes': [],
            'system_info': {}
        }

        try:
            # System information
            system_info = self.docker_client.info()
            docker_analysis['system_info'] = {
                'containers_running': system_info.get('ContainersRunning', 0),
                'containers_paused': system_info.get('ContainersPaused', 0),
                'containers_stopped': system_info.get('ContainersStopped', 0),
                'images': system_info.get('Images', 0),
                'server_version': system_info.get('ServerVersion', 'Unknown'),
                'memory_limit': system_info.get('MemTotal', 0),
                'cpu_count': system_info.get('NCPU', 0),
                'operating_system': system_info.get('OperatingSystem', 'Unknown'),
                'os_type': system_info.get('OSType', 'Unknown'),
                'architecture': system_info.get('Architecture', 'Unknown'),
                'isolation': system_info.get('Isolation', 'Unknown')
            }

            # Windows-specific Docker settings
            if 'windows' in system_info.get('OperatingSystem', '').lower():
                docker_analysis['system_info']['windows_version'] = system_info.get('OSVersion', 'Unknown')
                docker_analysis['system_info']['experimental'] = system_info.get('ExperimentalBuild', False)

            # Containers analysis with Windows specifics
            for container in self.docker_client.containers.list(all=True):
                container_info = {
                    'id': container.short_id,
                    'name': container.name,
                    'image': container.image.tags[0] if container.image.tags else 'Unknown',
                    'status': container.status,
                    'created': container.attrs['Created'],
                    'ports': container.ports,
                    'mounts': len(container.attrs.get('Mounts', [])),
                    'network_mode': container.attrs['HostConfig']['NetworkMode']
                }

                # Windows-specific container information
                if 'Platform' in container.attrs:
                    container_info['platform'] = container.attrs['Platform']

                host_config = container.attrs.get('HostConfig', {})
                if 'Isolation' in host_config:
                    container_info['isolation'] = host_config['Isolation']

                # Get resource usage if container is running
                if container.status == 'running':
                    try:
                        stats = container.stats(stream=False)
                        if stats:
                            # Windows containers have different stats structure
                            if 'memory_stats' in stats:
                                memory_stats = stats['memory_stats']
                                if 'privateworkingset' in memory_stats:
                                    # Windows containers use privateworkingset
                                    container_info['memory_usage_mb'] = memory_stats['privateworkingset'] / 1024 / 1024
                                elif 'usage' in memory_stats:
                                    container_info['memory_usage_mb'] = memory_stats['usage'] / 1024 / 1024

                            # CPU stats for Windows containers
                            if 'cpu_stats' in stats and 'cpu_usage' in stats['cpu_stats']:
                                cpu_percent = self._calculate_windows_cpu_percent(stats)
                                container_info['cpu_percent'] = cpu_percent

                    except Exception as e:
                        logging.warning(f"Could not get container stats: {e}")

                docker_analysis['containers'].append(container_info)

            # Images analysis with Windows focus
            for image in self.docker_client.images.list():
                image_info = {
                    'id': image.short_id,
                    'tags': image.tags,
                    'size_mb': image.attrs['Size'] / 1024 / 1024,
                    'created': image.attrs['Created']
                }

                # Detect Windows base images
                if image.tags:
                    for tag in image.tags:
                        if any(windows_tag in tag.lower() for windows_tag in ['windowsservercore', 'nanoserver', 'windows']):
                            image_info['is_windows_image'] = True
                            image_info['windows_type'] = 'Windows Container'
                            break
                    else:
                        image_info['is_windows_image'] = False

                # Get image platform information
                try:
                    image_details = self.docker_client.api.inspect_image(image.id)
                    if 'Os' in image_details:
                        image_info['os'] = image_details['Os']
                    if 'Architecture' in image_details:
                        image_info['architecture'] = image_details['Architecture']
                except Exception:
                    pass

                docker_analysis['images'].append(image_info)

            # Networks analysis
            for network in self.docker_client.networks.list():
                network_info = {
                    'id': network.short_id,
                    'name': network.name,
                    'driver': network.attrs['Driver'],
                    'containers': len(network.containers)
                }

                # Windows network specifics
                if 'Options' in network.attrs:
                    options = network.attrs['Options']
                    if 'com.docker.network.windowsshim.hnsid' in options:
                        network_info['hns_id'] = options['com.docker.network.windowsshim.hnsid']

                docker_analysis['networks'].append(network_info)

            # Volumes analysis
            for volume in self.docker_client.volumes.list():
                volume_info = {
                    'name': volume.name,
                    'driver': volume.attrs['Driver'],
                    'mountpoint': volume.attrs['Mountpoint']
                }

                # Windows volume paths
                if '\\' in volume.attrs['Mountpoint']:
                    volume_info['is_windows_path'] = True

                docker_analysis['volumes'].append(volume_info)

        except Exception as e:
            docker_analysis['error'] = str(e)
            logging.error(f"Docker Windows analysis error: {e}")

        return docker_analysis

    def _calculate_windows_cpu_percent(self, stats) -> float:
        """Calculate CPU percentage for Windows containers"""
        try:
            # Windows containers have different CPU stats structure
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})

            if 'cpu_usage' in cpu_stats and 'cpu_usage' in precpu_stats:
                cpu_usage = cpu_stats['cpu_usage']
                precpu_usage = precpu_stats['cpu_usage']

                if 'total_usage' in cpu_usage and 'total_usage' in precpu_usage:
                    cpu_delta = cpu_usage['total_usage'] - precpu_usage['total_usage']

                    if 'system_cpu_usage' in cpu_stats and 'system_cpu_usage' in precpu_stats:
                        system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']

                        if system_delta > 0:
                            # For Windows, calculate based on number of CPUs
                            num_cpus = len(cpu_usage.get('percpu_usage', [1]))
                            cpu_percent = (cpu_delta / system_delta) * num_cpus * 100
                            return round(cpu_percent, 2)

        except (KeyError, ZeroDivisionError, TypeError):
            pass
        return 0.0

    async def _analyze_windows_containers(self) -> Dict[str, Any]:
        """Analyze Windows-specific container features"""
        windows_analysis = {
            'windows_containers': [],
            'isolation_modes': {'process': 0, 'hyperv': 0, 'default': 0},
            'base_images': {'windowsservercore': 0, 'nanoserver': 0, 'other': 0}
        }

        try:
            if not self.docker_client:
                return windows_analysis

            # Analyze containers for Windows-specific features
            for container in self.docker_client.containers.list(all=True):
                host_config = container.attrs.get('HostConfig', {})

                # Check isolation mode
                isolation = host_config.get('Isolation', 'default')
                if isolation in windows_analysis['isolation_modes']:
                    windows_analysis['isolation_modes'][isolation] += 1
                else:
                    windows_analysis['isolation_modes']['default'] += 1

                # Analyze image for Windows base
                image_name = container.image.tags[0] if container.image.tags else ''

                if 'windowsservercore' in image_name.lower():
                    windows_analysis['base_images']['windowsservercore'] += 1
                elif 'nanoserver' in image_name.lower():
                    windows_analysis['base_images']['nanoserver'] += 1
                elif any(windows_tag in image_name.lower() for windows_tag in ['windows', 'mcr.microsoft.com']):
                    windows_analysis['base_images']['other'] += 1

                # Check for Windows-specific mounts
                mounts = container.attrs.get('Mounts', [])
                windows_mounts = []
                for mount in mounts:
                    if '\\' in mount.get('Source', '') or mount.get('Type') == 'npipe':
                        windows_mounts.append({
                            'type': mount.get('Type'),
                            'source': mount.get('Source'),
                            'destination': mount.get('Destination')
                        })

                if windows_mounts or isolation != 'default':
                    windows_analysis['windows_containers'].append({
                        'name': container.name,
                        'isolation': isolation,
                        'windows_mounts': windows_mounts,
                        'base_image_type': self._detect_base_image_type(image_name)
                    })

        except Exception as e:
            logging.error(f"Windows container analysis error: {e}")

        return windows_analysis

    def _detect_base_image_type(self, image_name: str) -> str:
        """Detect Windows base image type"""
        image_name = image_name.lower()

        if 'windowsservercore' in image_name:
            return 'Windows Server Core'
        elif 'nanoserver' in image_name:
            return 'Nano Server'
        elif 'servercore' in image_name:
            return 'Server Core'
        elif 'windows' in image_name or 'mcr.microsoft.com' in image_name:
            return 'Windows Container'
        else:
            return 'Linux Container'

    def _generate_windows_container_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Windows-specific container optimization recommendations"""
        recommendations = []

        docker_info = analysis.get('docker', {})
        windows_info = analysis.get('windows_containers', {})

        # Docker Desktop recommendations
        system_info = docker_info.get('system_info', {})
        if system_info.get('os_type', '').lower() == 'windows':
            if not system_info.get('experimental', False):
                recommendations.append({
                    'type': 'feature',
                    'priority': 'low',
                    'title': 'Enable Docker Experimental Features',
                    'description': 'Enable experimental features in Docker Desktop for access to latest Windows container features.',
                    'action': 'Enable experimental features in Docker Desktop settings'
                })

        # Windows containers vs Linux containers
        linux_containers = sum(1 for c in docker_info.get('containers', []) if not c.get('isolation'))
        windows_containers = len(windows_info.get('windows_containers', []))

        if linux_containers > 0 and windows_containers > 0:
            recommendations.append({
                'type': 'architecture',
                'priority': 'medium',
                'title': 'Mixed Container Environment Detected',
                'description': f'You have both Linux ({linux_containers}) and Windows ({windows_containers}) containers.',
                'action': 'Consider using separate Docker contexts or hosts for Linux and Windows containers'
            })

        # Isolation mode recommendations
        isolation_modes = windows_info.get('isolation_modes', {})
        if isolation_modes.get('process', 0) > 0:
            recommendations.append({
                'type': 'security',
                'priority': 'medium',
                'title': 'Process Isolation Mode Used',
                'description': f"{isolation_modes['process']} containers using process isolation.",
                'action': 'Consider Hyper-V isolation for better security in production'
            })

        # Base image recommendations
        base_images = windows_info.get('base_images', {})
        if base_images.get('windowsservercore', 0) > base_images.get('nanoserver', 0):
            recommendations.append({
                'type': 'optimization',
                'priority': 'medium',
                'title': 'Consider Nano Server Base Images',
                'description': f"Most containers ({base_images['windowsservercore']}) use Windows Server Core. Nano Server images are smaller.",
                'action': 'Evaluate if applications can run on Nano Server for smaller image sizes'
            })

        # Resource usage recommendations
        high_memory_containers = [c for c in docker_info.get('containers', [])
                                if c.get('memory_usage_mb', 0) > 1000]
        if len(high_memory_containers) > 2:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'High Memory Usage Containers',
                'description': f"{len(high_memory_containers)} containers using >1GB memory.",
                'action': 'Monitor memory usage and optimize Windows container resource allocation'
            })

        # Windows-specific mount recommendations
        windows_mount_containers = [c for c in windows_info.get('windows_containers', [])
                                  if c.get('windows_mounts')]
        if len(windows_mount_containers) > 0:
            recommendations.append({
                'type': 'best_practices',
                'priority': 'medium',
                'title': 'Windows-Specific Mounts Detected',
                'description': f'{len(windows_mount_containers)} containers using Windows-specific mounts.',
                'action': 'Ensure proper permissions and security for Windows container mounts'
            })

        return recommendations


# Export main classes for Windows integration
__all__ = [
    'WindowsRealTimeSystemMonitor',
    'WindowsAdvancedVisualizationPanel',
    'WindowsIntelligentCodeAnalyzer',
    'WindowsSmartContainerOrchestrator'
]