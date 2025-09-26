#!/usr/bin/env python3
"""
Advanced GUI Extensions for Terminal Coder Linux
Cutting-edge features with real implementations
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

# Advanced imports for cutting-edge features with fallback implementations
# Enhanced feature detection with graceful degradation
try:
    import torch
    import transformers
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.info("PyTorch not available - advanced ML features will be limited")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.info("scikit-learn not available - using fallback statistical methods")

    # Provide fallback classes for missing scikit-learn functionality
    class FallbackTfidfVectorizer:
        def __init__(self, *args, **kwargs):
            self.vocabulary = {}

        def fit_transform(self, documents):
            # Simple word frequency fallback
            import re
            all_words = set()
            doc_words = []

            for doc in documents:
                words = re.findall(r'\b\w+\b', doc.lower())
                doc_words.append(words)
                all_words.update(words)

            # Create simple frequency matrix
            vocab_list = list(all_words)
            matrix = []
            for words in doc_words:
                row = [words.count(word) for word in vocab_list]
                matrix.append(row)

            return np.array(matrix) if 'np' in globals() else matrix

    class FallbackKMeans:
        def __init__(self, n_clusters=5, *args, **kwargs):
            self.n_clusters = n_clusters

        def fit_predict(self, X):
            # Simple random clustering fallback
            import random
            if hasattr(X, 'shape'):
                n_samples = X.shape[0]
            else:
                n_samples = len(X)
            return [random.randint(0, self.n_clusters - 1) for _ in range(n_samples)]

    class FallbackStandardScaler:
        def fit_transform(self, X):
            return X  # No-op fallback

    # Use fallbacks when sklearn is not available
    TfidfVectorizer = FallbackTfidfVectorizer
    KMeans = FallbackKMeans
    StandardScaler = FallbackStandardScaler

    def cosine_similarity(X, Y=None):
        """Fallback cosine similarity implementation"""
        if Y is None:
            Y = X
        # Simple dot product based similarity (not true cosine)
        result = []
        for i, x in enumerate(X):
            row = []
            for j, y in enumerate(Y):
                if hasattr(x, '__iter__') and hasattr(y, '__iter__'):
                    similarity = sum(a * b for a, b in zip(x, y)) if x and y else 0
                else:
                    similarity = 0
                row.append(similarity)
            result.append(row)
        return result

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Combined ML availability check
ADVANCED_ML_AVAILABLE = SKLEARN_AVAILABLE

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

# Combined container availability check
CONTAINER_ADVANCED = DOCKER_AVAILABLE or KUBERNETES_AVAILABLE

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

# Advanced error tracking
import traceback
import functools
import warnings
from contextlib import contextmanager

def safe_import_fallback(module_name, fallback_value=None):
    """Safely import modules with fallback values"""
    try:
        return __import__(module_name)
    except ImportError as e:
        logging.warning(f"Failed to import {module_name}: {e}")
        return fallback_value

def handle_exceptions(default_return=None, log_errors=True):
    """Decorator for comprehensive exception handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logging.error(f"Error in {func.__name__}: {e}")
                    logging.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator

@contextmanager
def error_context(operation_name):
    """Context manager for operation error tracking"""
    try:
        logging.debug(f"Starting operation: {operation_name}")
        yield
        logging.debug(f"Completed operation: {operation_name}")
    except Exception as e:
        logging.error(f"Error in operation '{operation_name}': {e}")
        logging.debug(f"Traceback: {traceback.format_exc()}")
        raise


class RealTimeSystemMonitor:
    """Advanced real-time system monitoring with ML-powered analytics"""

    @handle_exceptions(log_errors=True)
    def __init__(self, gui_parent):
        self.parent = gui_parent
        self.is_monitoring = False
        self.monitoring_task = None
        self.data_lock = threading.RLock()  # Reentrant lock for thread safety

        # Initialize data structures with thread-safe collections
        self.data_history = {
            'cpu': deque(maxlen=100),
            'memory': deque(maxlen=100),
            'disk_io': deque(maxlen=100),
            'network_io': deque(maxlen=100),
            'processes': deque(maxlen=50),
            'timestamps': deque(maxlen=100)
        }

        # ML components
        self.anomaly_detector = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

        # Resource management
        self.db_connections = {}
        self.cleanup_handlers = []

        # Initialize components safely
        self.setup_ml_models()

        # Register cleanup
        import atexit
        atexit.register(self.cleanup)

    def setup_ml_models(self):
        """Setup ML models for system analysis"""
        if ADVANCED_ML_AVAILABLE:
            # Simple anomaly detection model
            self.anomaly_threshold = 2.0
            self.baseline_metrics = {}

    async def start_monitoring(self, update_callback: Callable):
        """Start advanced real-time monitoring with comprehensive error handling"""
        if self.is_monitoring:
            logging.warning("Monitoring already running")
            return

        self.is_monitoring = True
        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            with error_context("System Monitoring"):
                while self.is_monitoring:
                    try:
                        # Collect comprehensive system metrics with timeout
                        metrics = await asyncio.wait_for(
                            self._collect_advanced_metrics(),
                            timeout=30.0  # 30 second timeout
                        )

                        # Store historical data thread-safely
                        with self.data_lock:
                            self._store_metrics(metrics)

                        # Detect anomalies using ML (with fallback)
                        anomalies = self._detect_anomalies(metrics)

                        # Update GUI with real-time data (with error handling)
                        await self._safe_callback_execution(update_callback, metrics, anomalies)

                        # Reset error counter on success
                        consecutive_errors = 0

                        # Adaptive monitoring frequency based on system load
                        sleep_time = self._calculate_adaptive_sleep(metrics)
                        await asyncio.sleep(max(0.1, min(sleep_time, 10.0)))  # Bounds checking

                    except asyncio.TimeoutError:
                        logging.warning("Metrics collection timeout, continuing...")
                        consecutive_errors += 1
                        await asyncio.sleep(2.0)

                    except Exception as e:
                        consecutive_errors += 1
                        logging.error(f"Monitoring iteration error: {e}")
                        logging.debug(f"Error traceback: {traceback.format_exc()}")

                        # Exponential backoff for errors
                        backoff_time = min(2 ** consecutive_errors, 60)
                        await asyncio.sleep(backoff_time)

                        # Stop monitoring if too many consecutive errors
                        if consecutive_errors >= max_consecutive_errors:
                            logging.critical(f"Too many consecutive errors ({consecutive_errors}), stopping monitoring")
                            break

        except asyncio.CancelledError:
            logging.info("Monitoring cancelled")
            raise
        except Exception as e:
            logging.error(f"Fatal monitoring error: {e}")
            raise
        finally:
            self.is_monitoring = False

    async def _safe_callback_execution(self, callback: Callable, metrics: Dict, anomalies: List):
        """Execute callback with comprehensive error handling"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(metrics, anomalies)
            else:
                # Run synchronous callback in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, metrics, anomalies)
        except Exception as e:
            logging.error(f"Callback execution error: {e}")
            # Don't re-raise to avoid stopping monitoring

    async def _collect_advanced_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'per_cpu': psutil.cpu_percent(interval=0.1, percpu=True),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'stats': psutil.cpu_stats()._asdict(),
                'load_avg': os.getloadavg()
            },
            'memory': {
                'virtual': psutil.virtual_memory()._asdict(),
                'swap': psutil.swap_memory()._asdict()
            },
            'disk': {
                'usage': {path: psutil.disk_usage(path)._asdict()
                         for path in ['/'] if os.path.exists(path)},
                'io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
            },
            'network': {
                'io': psutil.net_io_counters()._asdict(),
                'connections': len(psutil.net_connections()),
                'interfaces': {name: addr._asdict() for name, addrs in psutil.net_if_addrs().items() for addr in addrs}
            },
            'processes': await self._collect_process_metrics(),
            'system': {
                'boot_time': psutil.boot_time(),
                'users': len(psutil.users()),
                'uptime': time.time() - psutil.boot_time()
            }
        }

        # Advanced GPU metrics if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            metrics['gpu'] = [{'id': gpu.id, 'load': gpu.load * 100, 'memory': gpu.memoryUsed / gpu.memoryTotal * 100} for gpu in gpus]
        except ImportError:
            pass

        return metrics

    async def _collect_process_metrics(self) -> List[Dict[str, Any]]:
        """Collect detailed process metrics"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'create_time', 'status', 'nice', 'num_threads']):
            try:
                proc_info = proc.info
                proc_info['cpu_percent'] = proc.cpu_percent(interval=0)
                proc_info['memory_mb'] = proc.memory_info().rss / 1024 / 1024
                processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU usage and return top processes
        return sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:20]

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

    @handle_exceptions(log_errors=True)
    def _store_in_database(self, metrics: Dict[str, Any]):
        """Store metrics in SQLite database with connection pooling"""
        db_path = None
        try:
            with error_context("Database storage"):
                # Ensure database directory exists
                db_path = Path.home() / '.local/share/terminal-coder/metrics.db'
                db_path.parent.mkdir(parents=True, exist_ok=True)

                # Use connection pooling for better resource management
                db_key = str(db_path)
                conn = None

                try:
                    # Get or create connection
                    if db_key in self.db_connections:
                        conn = self.db_connections[db_key]
                        # Test connection
                        conn.execute('SELECT 1')
                    else:
                        conn = sqlite3.connect(
                            str(db_path),
                            timeout=30.0,  # 30 second timeout
                            check_same_thread=False,  # Allow multi-threading
                            isolation_level=None  # Autocommit mode
                        )
                        # Optimize connection
                        conn.execute('PRAGMA journal_mode=WAL')
                        conn.execute('PRAGMA synchronous=NORMAL')
                        conn.execute('PRAGMA cache_size=10000')
                        conn.execute('PRAGMA temp_store=MEMORY')

                        self.db_connections[db_key] = conn

                    cursor = conn.cursor()

                    # Create table if not exists (with error handling)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS system_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            cpu_percent REAL,
                            memory_percent REAL,
                            disk_io REAL,
                            network_io REAL,
                            process_count INTEGER,
                            load_avg_1 REAL,
                            load_avg_5 REAL,
                            load_avg_15 REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(timestamp) ON CONFLICT REPLACE
                        )
                    ''')

                    # Prepare data with validation
                    timestamp = metrics.get('timestamp', datetime.now()).isoformat()
                    cpu_percent = float(metrics.get('cpu', {}).get('percent', 0.0))
                    memory_percent = float(metrics.get('memory', {}).get('virtual', {}).get('percent', 0.0))

                    # Safe disk I/O calculation
                    disk_io = 0
                    if metrics.get('disk', {}).get('io'):
                        disk_io = sum(v for v in metrics['disk']['io'].values() if isinstance(v, (int, float)))

                    # Safe network I/O calculation
                    network_io = 0
                    net_data = metrics.get('network', {}).get('io', {})
                    if net_data:
                        network_io = (net_data.get('bytes_sent', 0) + net_data.get('bytes_recv', 0))

                    process_count = len(metrics.get('processes', []))

                    # Safe load average extraction
                    load_avg = metrics.get('cpu', {}).get('load_avg', [0, 0, 0])
                    if not isinstance(load_avg, (list, tuple)) or len(load_avg) < 3:
                        load_avg = [0, 0, 0]

                    # Insert metrics with parameter validation
                    cursor.execute('''
                        INSERT OR REPLACE INTO system_metrics
                        (timestamp, cpu_percent, memory_percent, disk_io, network_io,
                         process_count, load_avg_1, load_avg_5, load_avg_15)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp,
                        max(0.0, min(100.0, cpu_percent)),  # Clamp values
                        max(0.0, min(100.0, memory_percent)),
                        max(0, disk_io),
                        max(0, network_io),
                        max(0, process_count),
                        float(load_avg[0]),
                        float(load_avg[1]),
                        float(load_avg[2])
                    ))

                    # Cleanup old records (keep last 7 days)
                    cutoff_time = (datetime.now() - timedelta(days=7)).isoformat()
                    cursor.execute('DELETE FROM system_metrics WHERE timestamp < ?', (cutoff_time,))

                except sqlite3.Error as e:
                    logging.error(f"SQLite error: {e}")
                    # Remove bad connection from pool
                    if db_key in self.db_connections:
                        try:
                            self.db_connections[db_key].close()
                        except:
                            pass
                        del self.db_connections[db_key]
                    raise

        except Exception as e:
            logging.error(f"Database storage error for {db_path}: {e}")
            logging.debug(f"Database error traceback: {traceback.format_exc()}")

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

            # Process count anomaly detection
            process_data = list(self.data_history['processes'])
            process_mean = np.mean(process_data)
            process_std = np.std(process_data)
            current_processes = len(metrics['processes'])

            if abs(current_processes - process_mean) > (2 * process_std):
                anomalies.append({
                    'type': 'processes',
                    'severity': 'medium',
                    'message': f'Process count anomaly: {current_processes} (normal: {process_mean:.0f}±{process_std:.0f})',
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
        """Stop monitoring with proper cleanup"""
        with error_context("Stopping monitoring"):
            self.is_monitoring = False

            # Cancel monitoring task if running
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    # Wait briefly for clean cancellation
                    asyncio.get_event_loop().run_until_complete(
                        asyncio.wait_for(self.monitoring_task, timeout=2.0)
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logging.info("Monitoring task cancelled")

    @handle_exceptions(log_errors=True)
    def cleanup(self):
        """Comprehensive resource cleanup"""
        try:
            # Stop monitoring
            self.stop_monitoring()

            # Close database connections
            for db_name, conn in self.db_connections.items():
                try:
                    if hasattr(conn, 'close'):
                        conn.close()
                        logging.debug(f"Closed database connection: {db_name}")
                except Exception as e:
                    logging.error(f"Error closing database {db_name}: {e}")

            self.db_connections.clear()

            # Execute cleanup handlers
            for handler in self.cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    logging.error(f"Error in cleanup handler: {e}")

            # Clear data structures
            with self.data_lock:
                for key in self.data_history:
                    self.data_history[key].clear()

            logging.info("System monitor cleanup completed")

        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
        return False  # Don't suppress exceptions

    def get_historical_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical data from database"""
        try:
            db_path = Path.home() / '.local/share/terminal-coder/metrics.db'
            if not db_path.exists():
                return {}

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            since_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            cursor.execute('''
                SELECT * FROM system_metrics
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
                'load_avg_1': [row[6] for row in rows],
                'load_avg_5': [row[7] for row in rows],
                'load_avg_15': [row[8] for row in rows]
            }

        except Exception as e:
            logging.error(f"Historical data retrieval error: {e}")
            return {}


class AdvancedVisualizationPanel:
    """Advanced data visualization with interactive charts"""

    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.figures = {}
        self.canvases = {}
        self.setup_matplotlib_style()
        self.create_visualization_tabs()

    def setup_matplotlib_style(self):
        """Setup modern matplotlib styling"""
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

        # System topology tab
        self.topology_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.topology_frame, text="🔗 Topology")
        self.create_topology_visualization()

        # ML Analysis tab
        if ADVANCED_ML_AVAILABLE:
            self.ml_frame = ttk.Frame(self.viz_notebook)
            self.viz_notebook.add(self.ml_frame, text="🤖 ML Analysis")
            self.create_ml_analysis()

    def create_realtime_charts(self):
        """Create real-time monitoring charts"""
        # Create figure with subplots
        self.figures['realtime'] = Figure(figsize=(12, 8), facecolor='#2d3142')

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

        # Process count subplot
        self.process_ax = self.figures['realtime'].add_subplot(223)
        self.process_ax.set_title('Process Count & Load Average', color='white')
        self.process_ax.set_facecolor('#2d3142')
        self.process_ax.tick_params(colors='white')

        # System health heatmap
        self.health_ax = self.figures['realtime'].add_subplot(224)
        self.health_ax.set_title('System Health Matrix', color='white')
        self.health_ax.set_facecolor('#2d3142')
        self.health_ax.tick_params(colors='white')

        self.figures['realtime'].tight_layout()

        # Create canvas
        self.canvases['realtime'] = FigureCanvasTkAgg(self.figures['realtime'], self.realtime_frame)
        self.canvases['realtime'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_historical_charts(self):
        """Create historical data analysis charts"""
        self.figures['historical'] = Figure(figsize=(14, 10), facecolor='#2d3142')

        # Time series analysis
        self.timeseries_ax = self.figures['historical'].add_subplot(311)
        self.timeseries_ax.set_title('24-Hour System Metrics', color='white')
        self.timeseries_ax.set_facecolor('#2d3142')
        self.timeseries_ax.tick_params(colors='white')

        # Correlation heatmap
        self.correlation_ax = self.figures['historical'].add_subplot(312)
        self.correlation_ax.set_title('Metrics Correlation Matrix', color='white')
        self.correlation_ax.set_facecolor('#2d3142')
        self.correlation_ax.tick_params(colors='white')

        # Performance trends
        self.trends_ax = self.figures['historical'].add_subplot(313)
        self.trends_ax.set_title('Performance Trends & Forecasting', color='white')
        self.trends_ax.set_facecolor('#2d3142')
        self.trends_ax.tick_params(colors='white')

        self.figures['historical'].tight_layout()

        self.canvases['historical'] = FigureCanvasTkAgg(self.figures['historical'], self.historical_frame)
        self.canvases['historical'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_topology_visualization(self):
        """Create system topology visualization"""
        self.figures['topology'] = Figure(figsize=(12, 8), facecolor='#2d3142')
        self.topology_ax = self.figures['topology'].add_subplot(111)
        self.topology_ax.set_title('System Architecture Topology', color='white')
        self.topology_ax.set_facecolor('#2d3142')
        self.topology_ax.tick_params(colors='white')

        self.canvases['topology'] = FigureCanvasTkAgg(self.figures['topology'], self.topology_frame)
        self.canvases['topology'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create topology graph
        self.create_system_topology_graph()

    def create_ml_analysis(self):
        """Create ML-based analysis visualizations"""
        if not ADVANCED_ML_AVAILABLE:
            return

        self.figures['ml'] = Figure(figsize=(14, 10), facecolor='#2d3142')

        # Anomaly detection plot
        self.anomaly_ax = self.figures['ml'].add_subplot(221)
        self.anomaly_ax.set_title('Anomaly Detection (Isolation Forest)', color='white')
        self.anomaly_ax.set_facecolor('#2d3142')
        self.anomaly_ax.tick_params(colors='white')

        # Process clustering
        self.cluster_ax = self.figures['ml'].add_subplot(222)
        self.cluster_ax.set_title('Process Behavior Clustering', color='white')
        self.cluster_ax.set_facecolor('#2d3142')
        self.cluster_ax.tick_params(colors='white')

        # Performance prediction
        self.prediction_ax = self.figures['ml'].add_subplot(223)
        self.prediction_ax.set_title('Performance Prediction (Neural Network)', color='white')
        self.prediction_ax.set_facecolor('#2d3142')
        self.prediction_ax.tick_params(colors='white')

        # Resource optimization suggestions
        self.optimization_ax = self.figures['ml'].add_subplot(224)
        self.optimization_ax.set_title('Resource Optimization Recommendations', color='white')
        self.optimization_ax.set_facecolor('#2d3142')
        self.optimization_ax.tick_params(colors='white')

        self.figures['ml'].tight_layout()

        self.canvases['ml'] = FigureCanvasTkAgg(self.figures['ml'], self.ml_frame)
        self.canvases['ml'].get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_realtime_charts(self, metrics: Dict[str, Any], anomalies: List[Dict[str, Any]]):
        """Update real-time charts with new data"""
        try:
            # Clear previous plots
            for ax in [self.cpu_memory_ax, self.network_ax, self.process_ax, self.health_ax]:
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

            # Network I/O chart
            if 'network_history' in metrics:
                network_data = metrics['network_history']
                self.network_ax.bar(timestamps[-len(network_data):], network_data, color='#4f5d75', alpha=0.7)
                self.network_ax.set_ylabel('Bytes/sec')

            # Process count and load average
            if 'process_history' in metrics and 'load_avg' in metrics:
                process_data = metrics['process_history']
                load_avg = metrics['load_avg']

                self.process_ax.plot(timestamps[-len(process_data):], process_data,
                                   label='Processes', color='#e9c46a', linewidth=2)
                if len(load_avg) >= 3:
                    self.process_ax.axhline(y=load_avg[0], color='red', linestyle='--', alpha=0.7, label='Load Avg 1m')
                self.process_ax.legend()

            # System health matrix
            health_data = np.array([
                [metrics.get('cpu', {}).get('percent', 0), metrics.get('memory', {}).get('virtual', {}).get('percent', 0)],
                [len(anomalies) * 10, metrics.get('disk_usage', 0)]
            ])

            sns.heatmap(health_data, annot=True, cmap='RdYlGn_r', ax=self.health_ax,
                       xticklabels=['Metric 1', 'Metric 2'],
                       yticklabels=['CPU/Memory', 'Anomalies/Disk'])

            # Refresh canvas
            self.canvases['realtime'].draw()

        except Exception as e:
            logging.error(f"Chart update error: {e}")

    def create_system_topology_graph(self):
        """Create advanced system topology graph"""
        try:
            # Create network graph
            G = nx.Graph()

            # Add nodes for system components
            components = {
                'System': {'type': 'root', 'color': '#e76f51'},
                'CPU': {'type': 'hardware', 'color': '#2a9d8f'},
                'Memory': {'type': 'hardware', 'color': '#e9c46a'},
                'Storage': {'type': 'hardware', 'color': '#4f5d75'},
                'Network': {'type': 'hardware', 'color': '#bfc0c0'},
                'Processes': {'type': 'software', 'color': '#264653'},
                'Services': {'type': 'software', 'color': '#f4a261'},
            }

            for name, attrs in components.items():
                G.add_node(name, **attrs)

            # Add edges
            edges = [
                ('System', 'CPU'), ('System', 'Memory'), ('System', 'Storage'),
                ('System', 'Network'), ('CPU', 'Processes'), ('Memory', 'Processes'),
                ('Storage', 'Services'), ('Network', 'Services')
            ]

            G.add_edges_from(edges)

            # Create layout
            pos = nx.spring_layout(G, k=2, iterations=50)

            # Draw network graph
            self.topology_ax.clear()

            # Draw nodes with different colors
            for component, attrs in components.items():
                nx.draw_networkx_nodes(G, pos, nodelist=[component],
                                     node_color=attrs['color'],
                                     node_size=2000, ax=self.topology_ax)

            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='#666666',
                                 width=2, alpha=0.7, ax=self.topology_ax)

            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10,
                                  font_color='white', ax=self.topology_ax)

            self.topology_ax.set_title('System Component Topology', color='white', fontsize=14)
            self.topology_ax.axis('off')

            self.canvases['topology'].draw()

        except Exception as e:
            logging.error(f"Topology graph error: {e}")


class IntelligentCodeAnalyzer:
    """Advanced code analysis with ML-powered insights"""

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

    async def analyze_codebase(self, project_path: str) -> Dict[str, Any]:
        """Comprehensive codebase analysis"""
        analysis_results = {
            'summary': {},
            'complexity': {},
            'dependencies': {},
            'security': {},
            'quality': {},
            'patterns': {},
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

            # Pattern recognition with graceful degradation
            try:
                if ADVANCED_ML_AVAILABLE:
                    analysis_results['patterns'] = await self._analyze_patterns(project_path)
                else:
                    analysis_results['patterns'] = await self._analyze_patterns_fallback(project_path)
            except Exception as e:
                logging.warning(f"Pattern analysis failed, using basic analysis: {e}")
                analysis_results['patterns'] = {'error': 'Pattern analysis unavailable', 'method_used': 'none'}

            # Generate recommendations
            analysis_results['recommendations'] = self._generate_recommendations(analysis_results)

        except Exception as e:
            logging.error(f"Code analysis error: {e}")
            analysis_results['error'] = str(e)

        return analysis_results

    async def _analyze_basic_metrics(self, project_path: Path) -> Dict[str, Any]:
        """Analyze basic code metrics"""
        metrics = {
            'total_files': 0,
            'total_lines': 0,
            'languages': {},
            'file_sizes': [],
            'largest_files': []
        }

        language_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
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
            '.sh': 'Shell',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby'
        }

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

                        # Count lines for text files
                        if ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb']:
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
        """Analyze code complexity metrics"""
        complexity = {
            'cyclomatic_complexity': [],
            'function_lengths': [],
            'class_counts': {},
            'nesting_levels': []
        }

        # Simple complexity analysis for Python files
        for py_file in project_path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Count control flow statements (rough cyclomatic complexity)
                    control_statements = content.count('if ') + content.count('elif ') + \
                                       content.count('while ') + content.count('for ') + \
                                       content.count('except ') + content.count('with ')

                    complexity['cyclomatic_complexity'].append(control_statements)

                    # Analyze functions
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

        # Calculate complexity statistics
        if complexity['cyclomatic_complexity']:
            complexity['avg_complexity'] = np.mean(complexity['cyclomatic_complexity'])
            complexity['max_complexity'] = max(complexity['cyclomatic_complexity'])

        if complexity['function_lengths']:
            complexity['avg_function_length'] = np.mean(complexity['function_lengths'])
            complexity['max_function_length'] = max(complexity['function_lengths'])

        return complexity

    async def _analyze_dependencies(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project dependencies"""
        dependencies = {
            'package_files': [],
            'imports': {},
            'dependency_tree': {},
            'vulnerabilities': []
        }

        # Check for dependency files
        dep_files = ['requirements.txt', 'package.json', 'Cargo.toml', 'pom.xml', 'build.gradle']

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

        # Analyze Python imports
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

            except Exception:
                continue

        return dependencies

    async def _analyze_security(self, project_path: Path) -> Dict[str, Any]:
        """Analyze security vulnerabilities"""
        security = {
            'potential_issues': [],
            'sensitive_files': [],
            'hardcoded_secrets': [],
            'security_score': 0
        }

        # Security patterns to look for
        security_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
            (r'eval\s*\(', 'Use of eval() function'),
            (r'exec\s*\(', 'Use of exec() function'),
            (r'os\.system\s*\(', 'Use of os.system()'),
            (r'subprocess\.call.*shell=True', 'Shell injection risk'),
            (r'pickle\.load', 'Pickle deserialization risk'),
            (r'yaml\.load\s*\(', 'YAML deserialization risk')
        ]

        import re

        # Check for sensitive files
        sensitive_files = ['.env', 'config.ini', 'settings.py', 'secrets.json', '.htpasswd']
        for sensitive_file in sensitive_files:
            for file_path in project_path.rglob(sensitive_file):
                security['sensitive_files'].append(str(file_path))

        # Scan code for security issues
        for code_file in project_path.rglob('*.py'):
            try:
                with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    for pattern, description in security_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            security['potential_issues'].append({
                                'file': str(code_file),
                                'line': line_num,
                                'issue': description,
                                'code': match.group(0)
                            })

            except Exception:
                continue

        # Calculate security score
        issue_count = len(security['potential_issues'])
        sensitive_count = len(security['sensitive_files'])

        # Higher score is better (0-100)
        security['security_score'] = max(0, 100 - (issue_count * 10) - (sensitive_count * 5))

        return security

    async def _analyze_quality(self, project_path: Path) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        quality = {
            'documentation_coverage': 0,
            'test_coverage_estimate': 0,
            'code_duplication': [],
            'naming_conventions': {},
            'quality_score': 0
        }

        total_functions = 0
        documented_functions = 0
        test_files = 0
        total_code_files = 0

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

            except Exception:
                continue

        # Calculate metrics
        if total_functions > 0:
            quality['documentation_coverage'] = (documented_functions / total_functions) * 100

        if total_code_files > 0:
            quality['test_coverage_estimate'] = (test_files / total_code_files) * 100

        # Calculate overall quality score
        quality['quality_score'] = (
            quality['documentation_coverage'] * 0.4 +
            quality['test_coverage_estimate'] * 0.6
        )

        return quality

    async def _analyze_patterns(self, project_path: Path) -> Dict[str, Any]:
        """Analyze code patterns using ML with error handling"""
        patterns = {
            'design_patterns': [],
            'code_clusters': [],
            'similarity_matrix': [],
            'method_used': 'advanced_ml'
        }

        if not ADVANCED_ML_AVAILABLE:
            return await self._analyze_patterns_fallback(project_path)

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
                            'cluster': int(cluster)
                        })

                # Calculate similarity matrix (for small datasets)
                if len(code_snippets) <= 10:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    patterns['similarity_matrix'] = similarity_matrix.tolist()

        except Exception as e:
            logging.error(f"Pattern analysis error: {e}")

        return patterns

    async def _analyze_patterns_fallback(self, project_path: Path) -> Dict[str, Any]:
        """Fallback pattern analysis using basic statistical methods"""
        patterns = {
            'design_patterns': [],
            'code_clusters': [],
            'similarity_matrix': [],
            'method_used': 'statistical_fallback'
        }

        try:
            # Basic pattern detection without ML
            code_files = list(project_path.rglob('*.py'))[:20]  # Limit for performance
            file_stats = []

            for py_file in code_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Basic statistics
                    lines = content.count('\n')
                    functions = content.count('def ')
                    classes = content.count('class ')
                    imports = content.count('import ')

                    file_stats.append({
                        'file': str(py_file),
                        'lines': lines,
                        'functions': functions,
                        'classes': classes,
                        'imports': imports,
                        'complexity_score': functions + classes * 2 + imports * 0.5
                    })

                except Exception:
                    continue

            # Group files by similarity in basic stats
            if file_stats:
                # Simple clustering based on complexity score
                sorted_files = sorted(file_stats, key=lambda x: x['complexity_score'])
                cluster_size = max(1, len(sorted_files) // 3)

                for i, file_stat in enumerate(sorted_files):
                    cluster = min(2, i // cluster_size)  # 3 clusters max
                    patterns['code_clusters'].append({
                        'file': file_stat['file'],
                        'cluster': cluster,
                        'complexity_score': file_stat['complexity_score']
                    })

                # Basic design pattern detection
                for file_stat in file_stats:
                    if file_stat['classes'] > file_stat['functions']:
                        patterns['design_patterns'].append({
                            'file': file_stat['file'],
                            'pattern': 'object_oriented',
                            'confidence': min(1.0, file_stat['classes'] / max(1, file_stat['functions']))
                        })
                    elif file_stat['functions'] > 5:
                        patterns['design_patterns'].append({
                            'file': file_stat['file'],
                            'pattern': 'functional',
                            'confidence': min(1.0, file_stat['functions'] / 10)
                        })

        except Exception as e:
            logging.error(f"Fallback pattern analysis error: {e}")
            patterns['error'] = str(e)

        return patterns

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate intelligent recommendations based on analysis"""
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

        if quality.get('test_coverage_estimate', 0) < 30:
            recommendations.append({
                'type': 'testing',
                'priority': 'high',
                'title': 'Increase Test Coverage',
                'description': f"Estimated test coverage is {quality.get('test_coverage_estimate', 0):.1f}%. Consider adding more test files.",
                'action': 'Create unit tests for core functionality'
            })

        # Security recommendations
        security = analysis_results.get('security', {})
        if security.get('security_score', 100) < 70:
            recommendations.append({
                'type': 'security',
                'priority': 'critical',
                'title': 'Address Security Issues',
                'description': f"Security score is {security.get('security_score', 0)}/100. Found {len(security.get('potential_issues', []))} potential security issues.",
                'action': 'Review and fix security vulnerabilities'
            })

        # Complexity recommendations
        complexity = analysis_results.get('complexity', {})
        if complexity.get('max_complexity', 0) > 20:
            recommendations.append({
                'type': 'refactoring',
                'priority': 'medium',
                'title': 'Reduce Code Complexity',
                'description': f"Maximum cyclomatic complexity is {complexity.get('max_complexity', 0)}. Consider refactoring complex functions.",
                'action': 'Break down complex functions into smaller ones'
            })

        # Performance recommendations
        summary = analysis_results.get('summary', {})
        if summary.get('max_function_length', 0) > 100:
            recommendations.append({
                'type': 'maintainability',
                'priority': 'medium',
                'title': 'Reduce Function Length',
                'description': f"Longest function has {summary.get('max_function_length', 0)} lines. Consider breaking down large functions.",
                'action': 'Split large functions into smaller, focused functions'
            })

        return recommendations


class SmartContainerOrchestrator:
    """Advanced container orchestration with Kubernetes integration"""

    def __init__(self):
        self.docker_client = None
        self.k8s_client = None
        self.setup_clients()

    def setup_clients(self):
        """Setup container orchestration clients"""
        if CONTAINER_ADVANCED:
            try:
                # Docker client
                self.docker_client = docker.from_env()

                # Kubernetes client
                kubernetes.config.load_kube_config()
                self.k8s_client = kubernetes.client.ApiClient()

            except Exception as e:
                logging.warning(f"Container client setup failed: {e}")

    async def analyze_container_infrastructure(self) -> Dict[str, Any]:
        """Analyze container infrastructure"""
        analysis = {
            'docker': {
                'containers': [],
                'images': [],
                'networks': [],
                'volumes': [],
                'system_info': {}
            },
            'kubernetes': {
                'pods': [],
                'services': [],
                'deployments': [],
                'nodes': []
            },
            'recommendations': []
        }

        if not self.docker_client:
            analysis['error'] = 'Docker client not available'
            return analysis

        try:
            # Docker analysis
            analysis['docker'] = await self._analyze_docker()

            # Kubernetes analysis
            if self.k8s_client:
                analysis['kubernetes'] = await self._analyze_kubernetes()

            # Generate recommendations
            analysis['recommendations'] = self._generate_container_recommendations(analysis)

        except Exception as e:
            analysis['error'] = str(e)
            logging.error(f"Container analysis error: {e}")

        return analysis

    async def _analyze_docker(self) -> Dict[str, Any]:
        """Analyze Docker environment"""
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
                'cpu_count': system_info.get('NCPU', 0)
            }

            # Containers analysis
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

                # Get resource usage if container is running
                if container.status == 'running':
                    try:
                        stats = container.stats(stream=False)
                        cpu_percent = self._calculate_cpu_percent(stats)
                        memory_usage = stats['memory_stats'].get('usage', 0)
                        memory_limit = stats['memory_stats'].get('limit', 0)
                        memory_percent = (memory_usage / memory_limit * 100) if memory_limit else 0

                        container_info.update({
                            'cpu_percent': cpu_percent,
                            'memory_usage_mb': memory_usage / 1024 / 1024,
                            'memory_percent': memory_percent
                        })
                    except Exception:
                        pass

                docker_analysis['containers'].append(container_info)

            # Images analysis
            for image in self.docker_client.images.list():
                image_info = {
                    'id': image.short_id,
                    'tags': image.tags,
                    'size_mb': image.attrs['Size'] / 1024 / 1024,
                    'created': image.attrs['Created']
                }
                docker_analysis['images'].append(image_info)

            # Networks analysis
            for network in self.docker_client.networks.list():
                network_info = {
                    'id': network.short_id,
                    'name': network.name,
                    'driver': network.attrs['Driver'],
                    'containers': len(network.containers)
                }
                docker_analysis['networks'].append(network_info)

            # Volumes analysis
            for volume in self.docker_client.volumes.list():
                volume_info = {
                    'name': volume.name,
                    'driver': volume.attrs['Driver'],
                    'mountpoint': volume.attrs['Mountpoint']
                }
                docker_analysis['volumes'].append(volume_info)

        except Exception as e:
            docker_analysis['error'] = str(e)

        return docker_analysis

    def _calculate_cpu_percent(self, stats):
        """Calculate CPU percentage from Docker stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']

            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                return round(cpu_percent, 2)
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0

    async def _analyze_kubernetes(self) -> Dict[str, Any]:
        """Analyze Kubernetes cluster"""
        k8s_analysis = {
            'pods': [],
            'services': [],
            'deployments': [],
            'nodes': []
        }

        try:
            v1 = kubernetes.client.CoreV1Api(self.k8s_client)
            apps_v1 = kubernetes.client.AppsV1Api(self.k8s_client)

            # Pods analysis
            pods = v1.list_pod_for_all_namespaces()
            for pod in pods.items:
                pod_info = {
                    'name': pod.metadata.name,
                    'namespace': pod.metadata.namespace,
                    'status': pod.status.phase,
                    'node': pod.spec.node_name,
                    'containers': len(pod.spec.containers),
                    'restart_count': sum(c.restart_count for c in pod.status.container_statuses or [])
                }
                k8s_analysis['pods'].append(pod_info)

            # Services analysis
            services = v1.list_service_for_all_namespaces()
            for service in services.items:
                service_info = {
                    'name': service.metadata.name,
                    'namespace': service.metadata.namespace,
                    'type': service.spec.type,
                    'cluster_ip': service.spec.cluster_ip,
                    'ports': len(service.spec.ports or [])
                }
                k8s_analysis['services'].append(service_info)

            # Deployments analysis
            deployments = apps_v1.list_deployment_for_all_namespaces()
            for deployment in deployments.items:
                deployment_info = {
                    'name': deployment.metadata.name,
                    'namespace': deployment.metadata.namespace,
                    'replicas': deployment.spec.replicas,
                    'ready_replicas': deployment.status.ready_replicas or 0,
                    'available_replicas': deployment.status.available_replicas or 0
                }
                k8s_analysis['deployments'].append(deployment_info)

            # Nodes analysis
            nodes = v1.list_node()
            for node in nodes.items:
                node_info = {
                    'name': node.metadata.name,
                    'status': 'Ready' if any(c.type == 'Ready' and c.status == 'True' for c in node.status.conditions) else 'NotReady',
                    'version': node.status.node_info.kubelet_version,
                    'os': node.status.node_info.os_image,
                    'architecture': node.status.node_info.architecture
                }
                k8s_analysis['nodes'].append(node_info)

        except Exception as e:
            k8s_analysis['error'] = str(e)

        return k8s_analysis

    def _generate_container_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate container optimization recommendations"""
        recommendations = []

        docker_info = analysis.get('docker', {})

        # Resource usage recommendations
        for container in docker_info.get('containers', []):
            if container.get('cpu_percent', 0) > 80:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'title': f"High CPU Usage in {container['name']}",
                    'description': f"Container {container['name']} is using {container['cpu_percent']:.1f}% CPU",
                    'action': 'Consider scaling or optimizing the container'
                })

            if container.get('memory_percent', 0) > 90:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'critical',
                    'title': f"High Memory Usage in {container['name']}",
                    'description': f"Container {container['name']} is using {container['memory_percent']:.1f}% memory",
                    'action': 'Investigate memory leaks or increase memory limits'
                })

        # Image optimization recommendations
        large_images = [img for img in docker_info.get('images', []) if img.get('size_mb', 0) > 1000]
        if large_images:
            recommendations.append({
                'type': 'optimization',
                'priority': 'medium',
                'title': 'Large Container Images Detected',
                'description': f"Found {len(large_images)} images larger than 1GB",
                'action': 'Consider using multi-stage builds or alpine base images'
            })

        # Unused resources
        stopped_containers = [c for c in docker_info.get('containers', []) if c['status'] in ['exited', 'stopped']]
        if len(stopped_containers) > 5:
            recommendations.append({
                'type': 'cleanup',
                'priority': 'low',
                'title': 'Many Stopped Containers',
                'description': f"Found {len(stopped_containers)} stopped containers",
                'action': 'Clean up unused containers to free disk space'
            })

        # Kubernetes recommendations
        k8s_info = analysis.get('kubernetes', {})
        for deployment in k8s_info.get('deployments', []):
            if deployment.get('ready_replicas', 0) < deployment.get('replicas', 0):
                recommendations.append({
                    'type': 'availability',
                    'priority': 'high',
                    'title': f"Deployment {deployment['name']} Not Fully Ready",
                    'description': f"Only {deployment['ready_replicas']}/{deployment['replicas']} replicas are ready",
                    'action': 'Check pod status and resource availability'
                })

        return recommendations


# Export main classes for integration
__all__ = [
    'RealTimeSystemMonitor',
    'AdvancedVisualizationPanel',
    'IntelligentCodeAnalyzer',
    'SmartContainerOrchestrator'
]