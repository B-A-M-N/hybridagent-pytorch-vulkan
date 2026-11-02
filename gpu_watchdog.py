#!/usr/bin/env python3
"""
GPU Safety Watchdog — Independent monitoring process for AMD RX 6600 XT (RADV)
Monitors temperature, VRAM, GPU faults, and inference process health.
Fail-closed design: stops GPU work on threshold violations.
"""

import os
import sys
import time
import json
import signal
import subprocess
import threading
import psutil
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('/tmp/gpu_watchdog.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('gpu_watchdog')

# Safety thresholds (millicelsius)
HOTSPOT_WARN = 85_000      # 85°C
HOTSPOT_CRITICAL = 90_000  # 90°C - stop GPU work
EDGE_WARN = 75_000         # 75°C
EDGE_CRITICAL = 80_000     # 80°C - stop GPU work
VRAM_MIN_AVAILABLE = 1.5 * 1024**3  # 1.5 GiB minimum
VRAM_MAX_ALLOC_RATIO = 0.70         # Max 70% of initially free VRAM
VRAM_GROWTH_WARN = 0.10             # 10% rapid growth warning

class WatchdogState(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    STOPPED = "stopped"

@dataclass
class GPUMetrics:
    timestamp: float
    edge_temp: int          # millicelsius
    junction_temp: int      # millicelsius (hotspot)
    mem_temp: int           # millicelsius
    vram_used: int          # bytes
    vram_total: int         # bytes
    vram_available: int     # bytes
    gtt_used: int           # bytes
    gpu_utilization: int    # percent
    power_draw: int         # microwatts
    vram_growth_rate: float = 0.0  # bytes/sec

@dataclass
class SafetyEvent:
    timestamp: str
    level: str
    message: str
    metrics: Optional[GPUMetrics] = None

class GPUSysfsMonitor:
    """Reads GPU metrics from sysfs/hwmon"""
    
    def __init__(self, card_path: str = "/sys/class/drm/card0/device"):
        self.card_path = pathlib.Path(card_path)
        self.hwmon_path = self._find_hwmon()
        self.mem_info_path = self.card_path
        self._initial_vram_free = None
        self._last_vram_used = None
        self._last_timestamp = None
    
    def _find_hwmon(self) -> Optional[pathlib.Path]:
        for p in self.card_path.glob("hwmon/hwmon*"):
            return p
        return None
    
    def read_metrics(self) -> GPUMetrics:
        now = time.time()
        
        # Read temperatures from hwmon
        edge_temp = junction_temp = mem_temp = 0
        if self.hwmon_path:
            try:
                edge_temp = int((self.hwmon_path / "temp1_input").read_text().strip())
                junction_temp = int((self.hwmon_path / "temp2_input").read_text().strip())
                mem_temp = int((self.hwmon_path / "temp3_input").read_text().strip())
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to read temperatures: {e}")
        
        # Read VRAM info
        vram_used = vram_total = vram_available = gtt_used = 0
        try:
            vram_used = int((self.mem_info_path / "mem_info_vram_used").read_text().strip())
            vram_total = int((self.mem_info_path / "mem_info_vram_total").read_text().strip())
            gtt_used = int((self.mem_info_path / "mem_info_gtt_used").read_text().strip())
            vram_available = vram_total - vram_used
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to read VRAM info: {e}")
        
        # GPU utilization
        gpu_util = 0
        try:
            gpu_util = int((self.card_path / "gpu_busy_percent").read_text().strip())
        except (FileNotFoundError, ValueError):
            pass
        
        # Power draw
        power_draw = 0
        hwmon = self.hwmon_path
        if hwmon:
            try:
                power_draw = int((hwmon / "power1_average").read_text().strip())
            except (FileNotFoundError, ValueError):
                pass
        
        # Calculate VRAM growth rate
        vram_growth_rate = 0.0
        if self._last_vram_used is not None and self._last_timestamp is not None:
            dt = time.time() - self._last_timestamp
            if dt > 0:
                vram_growth_rate = (vram_used - self._last_vram_used) / dt
        
        self._last_vram_used = vram_used
        self._last_timestamp = time.time()
        
        if self._initial_vram_free is None:
            self._initial_vram_free = vram_total - vram_used
        
        return GPUMetrics(
            timestamp=time.time(),
            edge_temp=edge_temp,
            junction_temp=junction_temp,
            mem_temp=mem_temp,
            vram_used=vram_used,
            vram_total=vram_total,
            vram_available=vram_available,
            gtt_used=gtt_used,
            gpu_utilization=gpu_util,
            power_draw=power_draw,
            vram_growth_rate=vram_growth_rate
        )
    
    def check_thresholds(self, metrics: GPUMetrics) -> List[SafetyEvent]:
        events = []
        
        # Temperature checks
        if metrics.junction_temp >= HOTSPOT_CRITICAL:
            events.append(SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                message=f"Hotspot temperature {metrics.junction_temp/1000}°C >= {HOTSPOT_CRITICAL/1000}°C — STOPPING GPU WORK",
                metrics=metrics
            ))
        elif metrics.junction_temp >= HOTSPOT_WARN:
            events.append(SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="WARNING",
                message=f"Hotspot temperature {metrics.junction_temp/1000}°C >= {HOTSPOT_WARN/1000}°C",
                metrics=metrics
            ))
        
        if metrics.edge_temp >= EDGE_CRITICAL:
            events.append(SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                message=f"Edge temperature {metrics.edge_temp/1000}°C >= {EDGE_CRITICAL/1000}°C — STOPPING GPU WORK",
                metrics=metrics
            ))
        elif metrics.edge_temp >= EDGE_WARN:
            events.append(SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="WARNING",
                message=f"Edge temperature {metrics.edge_temp/1000}°C >= {EDGE_WARN/1000}°C",
                metrics=metrics
            ))
        
        # VRAM checks
        if metrics.vram_available < VRAM_MIN_AVAILABLE:
            events.append(SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                message=f"VRAM available {metrics.vram_available/1024**3:.2f} GiB < {VRAM_MIN_AVAILABLE/1024**3:.1f} GiB minimum",
                metrics=metrics
            ))
        
        # VRAM growth rate check
        if metrics.vram_growth_rate > 100 * 1024**2:  # >100 MB/s sustained
            events.append(SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="WARNING",
                message=f"Rapid VRAM growth: {metrics.vram_growth_rate/1024**2:.1f} MB/s",
                metrics=metrics
            ))
        
        return events

class DmesgMonitor:
    """Monitors kernel logs for AMDGPU faults and resets"""
    
    def __init__(self):
        self.last_check = time.time()
        self.seen_lines = set()
    
    def check(self) -> List[SafetyEvent]:
        events = []
        try:
            result = subprocess.run(
                ["dmesg", "-T", "--since", f"{int(self.last_check)}s ago"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if line and 'amdgpu' in line.lower() and line not in self.seen_lines:
                    self.seen_lines.add(line)
                    level = "INFO"
                    if any(kw in line.lower() for kw in ['reset', 'fault', 'timeout', 'error', 'hang']):
                        level = "CRITICAL"
                    elif 'warning' in line.lower():
                        level = "WARNING"
                    events.append(SafetyEvent(
                        timestamp=datetime.now().isoformat(),
                        level=level,
                        message=f"dmesg: {line.strip()}",
                        metrics=None
                    ))
            self.last_check = time.time()
        except Exception as e:
            logger.warning(f"dmesg check failed: {e}")
        return events

class ProcessHealthMonitor:
    """Monitors inference worker process health"""
    
    def __init__(self, pid: Optional[int] = None, cmd_pattern: Optional[str] = None):
        self.pid = pid
        self.cmd_pattern = cmd_pattern
        self.process: Optional[psutil.Process] = None
        self._find_process()
    
    def _find_process(self):
        if self.pid:
            try:
                self.process = psutil.Process(self.pid)
                return
            except psutil.NoSuchProcess:
                pass
        if self.cmd_pattern:
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if self.cmd_pattern in cmdline:
                        self.process = proc
                        self.pid = proc.pid
                        return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    
    def check(self) -> List[SafetyEvent]:
        events = []
        if not self.process:
            # No process to monitor - this is OK for independent watchdog operation
            return events
        
        try:
            if not self.process.is_running():
                events.append(SafetyEvent(
                    timestamp=datetime.now().isoformat(),
                    level="CRITICAL",
                    message=f"Inference worker PID {self.pid} has exited",
                    metrics=None
                ))
            elif self.process.status() == psutil.STATUS_ZOMBIE:
                events.append(SafetyEvent(
                    timestamp=datetime.now().isoformat(),
                    level="CRITICAL",
                    message=f"Inference worker PID {self.pid} is zombie",
                    metrics=None
                ))
        except psutil.NoSuchProcess:
            events.append(SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                message=f"Inference worker PID {self.pid} no longer exists",
                metrics=None
            ))
        except psutil.AccessDenied:
            events.append(SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="WARNING",
                message=f"Cannot access inference worker PID {self.pid}",
                metrics=None
            ))
        
        return events

class GPUWatchdog:
    """Main watchdog coordinating all monitoring"""
    
    def __init__(self, worker_pid: Optional[int] = None, worker_cmd: Optional[str] = None):
        self.state = WatchdogState.HEALTHY
        self.sysfs = GPUSysfsMonitor()
        self.dmesg = DmesgMonitor()
        self.proc_health = ProcessHealthMonitor(worker_pid, worker_cmd)
        self.events: List[SafetyEvent] = []
        self._stop_flag = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_gpu_work = False
        self._lock = threading.Lock()
        
        # Control interface (for commands)
        self.control_pipe = "/tmp/gpu_watchdog_control"
        # Status pipe (for responses)
        self.status_pipe = "/tmp/gpu_watchdog_status"
        self._setup_control_pipe()
        self._setup_status_pipe()
    
    def _setup_control_pipe(self):
        try:
            os.mkfifo(self.control_pipe, 0o600)
        except FileExistsError:
            pass
    
    def _setup_status_pipe(self):
        try:
            os.mkfifo(self.status_pipe, 0o600)
        except FileExistsError:
            pass
    
    @property
    def stop_gpu_work(self) -> bool:
        with self._lock:
            return self._stop_gpu_work
    
    def _set_stop_gpu_work(self, value: bool):
        with self._lock:
            self._stop_gpu_work = value
    
    def record_event(self, event: SafetyEvent):
        self.events.append(event)
        logger.log(
            logging.CRITICAL if event.level == "CRITICAL" else 
            logging.WARNING if event.level == "WARNING" else 
            logging.INFO,
            f"[{event.level}] {event.message}"
        )
        
        # Update state
        if event.level == "CRITICAL":
            self.state = WatchdogState.CRITICAL
            self._set_stop_gpu_work(True)
        elif event.level == "WARNING" and self.state == WatchdogState.HEALTHY:
            self.state = WatchdogState.WARNING
    
    def _monitor_loop(self):
        logger.info("GPU Watchdog started")
        while not self._stop_flag:
            try:
                # Read metrics
                metrics = self.sysfs.read_metrics()
                
                # Check sysfs thresholds
                for event in self.sysfs.check_thresholds(metrics):
                    self.record_event(event)
                
                # Check dmesg
                for event in self.dmesg.check():
                    self.record_event(event)
                
                # Check process health
                for event in self.proc_health.check():
                    self.record_event(event)
                
                # Log summary every 10 seconds
                if int(time.time()) % 10 == 0:
                    logger.info(
                        f"GPU: edge={metrics.edge_temp/1000:.1f}°C "
                        f"junction={metrics.junction_temp/1000:.1f}°C "
                        f"VRAM={metrics.vram_used/1024**3:.2f}/{metrics.vram_total/1024**3:.2f} GiB "
                        f"util={metrics.gpu_utilization}% "
                        f"state={self.state.value}"
                    )
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            time.sleep(1.0)
        
        logger.info("GPU Watchdog stopped")
    
    def start(self):
        self._stop_flag = False
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self):
        self._stop_flag = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state.value,
                "stop_gpu_work": self._stop_gpu_work,
                "events_count": len(self.events),
                "recent_events": [e.__dict__ for e in self.events[-10:]],
                "metrics": self.sysfs.read_metrics().__dict__ if self.sysfs else None
            }
    
    def simulate_failure(self, failure_type: str):
        """Inject simulated failures for safety gate testing"""
        if failure_type == "temperature":
            event = SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                message="SIMULATED: Hotspot temperature 95°C >= 90°C — STOPPING GPU WORK",
                metrics=None
            )
            self.record_event(event)
        elif failure_type == "vram":
            event = SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                message="SIMULATED: VRAM available 0.5 GiB < 1.5 GiB minimum",
                metrics=None
            )
            self.record_event(event)
        elif failure_type == "timeout":
            event = SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                message="SIMULATED: GPU execution timeout",
                metrics=None
            )
            self.record_event(event)
        elif failure_type == "device_lost":
            event = SafetyEvent(
                timestamp=datetime.now().isoformat(),
                level="CRITICAL",
                message="SIMULATED: VK_ERROR_DEVICE_LOST",
                metrics=None
            )
            self.record_event(event)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU Safety Watchdog")
    parser.add_argument("--pid", type=int, help="Worker process PID to monitor")
    parser.add_argument("--cmd", help="Worker command pattern to monitor")
    parser.add_argument("--simulate", choices=["temperature", "vram", "timeout", "device_lost"],
                        help="Inject simulated failure for safety gate testing")
    parser.add_argument("--status", action="store_true", help="Print status and exit")
    args = parser.parse_args()
    
    watchdog = GPUWatchdog(worker_pid=args.pid, worker_cmd=args.cmd)
    
    if args.simulate:
        watchdog.simulate_failure(args.simulate)
        print(json.dumps(watchdog.get_status(), indent=2))
        return
    
    if args.status:
        print(json.dumps(watchdog.get_status(), indent=2))
        return
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        watchdog.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    watchdog.start()
    
    # Control pipe listener
    try:
        while True:
            with open(watchdog.control_pipe, 'r') as f:
                cmd = f.read().strip()
            # Write response to status pipe
            with open(watchdog.status_pipe, 'w') as f:
                if cmd == "status":
                    f.write(json.dumps(watchdog.get_status()) + '\n')
                elif cmd == "stop":
                    f.write("OK\n")
                    watchdog.stop()
                    break
                elif cmd.startswith("simulate "):
                    watchdog.simulate_failure(cmd.split()[1])
                    f.write("OK\n")
                else:
                    f.write("UNKNOWN\n")
    except KeyboardInterrupt:
        pass
    finally:
        watchdog.stop()

if __name__ == "__main__":
    main()