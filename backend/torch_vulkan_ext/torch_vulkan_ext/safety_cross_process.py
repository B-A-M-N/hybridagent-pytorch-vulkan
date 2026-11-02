#!/usr/bin/env python3
"""
Cross-Process Vulkan Safety Enforcement (P0)
Uses the GPUWatchdog control pipe for proper process-separated safety enforcement.
"""

import os
import ctypes
import time
import json
import logging
import threading
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

try:
    from .torch_vulkan_ext import _load, Allocation, BufferView
except ImportError:
    import sys
    sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan')
    from backend.torch_vulkan_ext.torch_vulkan_ext import _load, Allocation, BufferView

logger = logging.getLogger("vulkan_safety_cross_process")

class SafetyError(Exception):
    pass

class ValidationError(SafetyError):
    pass

class DeviceLostError(SafetyError):
    pass

class TimeoutError(SafetyError):
    pass

class WatchdogState(Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    STOPPED = "stopped"
    UNREACHABLE = "unreachable"

@dataclass
class DispatchSpec:
    kernel_name: str
    buffers: List[BufferView]
    push_constants: bytes
    workgroup_x: int
    workgroup_y: int
    workgroup_z: int
    fence_timeout_ms: int = 5000

class CrossProcessWatchdogClient:
    """
    Client for communicating with the GPUWatchdog daemon via named pipe.
    Ensures worker cannot submit without healthy watchdog heartbeat.
    """
    
    def __init__(self, pipe_path: str = "/tmp/gpu_watchdog_control", 
                 status_pipe_path: str = "/tmp/gpu_watchdog_status"):
        self.pipe_path = pipe_path
        self.status_pipe_path = status_pipe_path
        self._lock = threading.Lock()
        self._cached_state = WatchdogState.UNKNOWN
        self._last_status_time = 0
        self._status_cache_ttl = 0.5  # Cache status for 500ms
        
    def _send_command(self, cmd: str) -> Optional[str]:
        """Send command to watchdog control pipe with timeout."""
        try:
            # Use os.open with O_NONBLOCK to avoid blocking if watchdog isn't running
            # Write command
            fd_write = os.open(self.pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd_write, (cmd + '\n').encode())
            finally:
                os.close(fd_write)
            
            # Read response from status pipe with timeout
            fd_read = os.open(self.status_pipe_path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                # Poll for data with timeout
                import select
                ready, _, _ = select.select([fd_read], [], [], 1.0)  # 1 second timeout
                if not ready:
                    return None
                response = os.read(fd_read, 4096).decode().strip()
            finally:
                os.close(fd_read)
            return response
        except (FileNotFoundError, IOError, OSError) as e:
            logger.warning(f"Watchdog communication failed: {e}")
            return None
    
    def get_status(self) -> dict:
        """Get watchdog status via control pipe."""
        with self._lock:
            now = time.time()
            if now - self._last_status_time < self._status_cache_ttl:
                return self._cached_status
            
            response = self._send_command("status")
            if response is None:
                self._cached_state = WatchdogState.UNREACHABLE
                self._cached_status = {
                    "state": self._cached_state.value,
                    "stop_gpu_work": True,  # FAIL CLOSED: unreachable = stop
                    "events_count": 0,
                    "recent_events": [],
                    "metrics": None
                }
            else:
                try:
                    status = json.loads(response)
                    self._cached_state = WatchdogState(status.get("state", "unknown"))
                    self._cached_status = status
                except json.JSONDecodeError:
                    self._cached_state = WatchdogState.UNREACHABLE
                    self._cached_status = {
                        "state": "unreachable",
                        "stop_gpu_work": True,
                        "events_count": 0,
                        "recent_events": [],
                        "metrics": None
                    }
            
            self._last_status_time = now
            return self._cached_status
    
    def is_healthy(self) -> bool:
        """Check if watchdog is healthy and allows GPU work."""
        status = self.get_status()
        return (
            status.get("state") == "healthy" and 
            not status.get("stop_gpu_work", True)
        )
    
    def check_and_raise(self) -> None:
        """Check watchdog state, raise SafetyError if not healthy."""
        status = self.get_status()
        if status.get("stop_gpu_work", True):
            state = status.get("state", "unreachable")
            raise SafetyError(f"Watchdog state={state}: GPU work halted")
        if status.get("state") != "healthy":
            raise SafetyError(f"Watchdog unhealthy: {status.get('state')}")

class VulkanSafetyValidator:
    """Validates all Vulkan operations before execution."""
    
    # Physical device limits for RX 6600 XT (RADV NAVI23)
    MAX_WORKGROUP_COUNT = (65535, 65535, 65535)
    MAX_WORKGROUP_SIZE = 1024
    MAX_WORKGROUP_INVOCATIONS = 1024
    MAX_PUSH_CONSTANT_SIZE = 128  # bytes
    MAX_BUFFER_SIZE = 1 << 30  # 1 GiB per buffer
    MAX_DESCRIPTOR_SETS = 8
    
    def __init__(self, watchdog_client: Optional[CrossProcessWatchdogClient] = None):
        self.watchdog = watchdog_client
        self._lib = _load()
        self._in_flight_submissions = 0
        self._max_in_flight = 1
        self._lock = threading.Lock()
        self._validation_layers_enabled = True
        
    def validate_buffer(self, view: BufferView, expected_size: int, 
                        expected_offset: int = 0) -> None:
        if view.size != expected_size:
            raise ValidationError(
                f"Buffer size mismatch: expected {expected_size}, got {view.size}"
            )
        if view.offset != expected_offset:
            raise ValidationError(
                f"Buffer offset mismatch: expected {expected_offset}, got {view.offset}"
            )
        if view.size > self.MAX_BUFFER_SIZE:
            raise ValidationError(f"Buffer size {view.size} exceeds max")
        if view.offset + view.size > view._alloc.size:
            raise ValidationError(
                f"Buffer view extends past allocation: offset={view.offset}, size={view.size}, alloc_size={view._alloc.size}"
            )
        if view.offset % 4 != 0:
            raise ValidationError(f"Buffer offset {view.offset} not 4-byte aligned")
        if view.size % 4 != 0:
            raise ValidationError(f"Buffer size {view.size} not 4-byte aligned")
    
    def validate_dispatch(self, spec: DispatchSpec) -> None:
        if spec.workgroup_x <= 0 or spec.workgroup_x > self.MAX_WORKGROUP_COUNT[0]:
            raise ValidationError(f"workgroup_x {spec.workgroup_x} out of range")
        if spec.workgroup_y <= 0 or spec.workgroup_y > self.MAX_WORKGROUP_COUNT[1]:
            raise ValidationError(f"workgroup_y {spec.workgroup_y} out of range")
        if spec.workgroup_z <= 0 or spec.workgroup_z > self.MAX_WORKGROUP_COUNT[2]:
            raise ValidationError(f"workgroup_z {spec.workgroup_z} out of range")
        if len(spec.push_constants) > self.MAX_PUSH_CONSTANT_SIZE:
            raise ValidationError(f"Push constants {len(spec.push_constants)} bytes > max {self.MAX_PUSH_CONSTANT_SIZE}")
        if spec.fence_timeout_ms <= 0 or spec.fence_timeout_ms > 30000:
            raise ValidationError(f"Fence timeout {spec.fence_timeout_ms}ms out of range")
        for buf in spec.buffers:
            if buf.size > self.MAX_BUFFER_SIZE:
                raise ValidationError(f"Buffer size {buf.size} exceeds max")
    
    def validate_shader_bounds(self, kernel_name: str, total_elements: int,
                               workgroup_size: int, workgroup_count: int) -> None:
        if total_elements <= 0:
            raise ValidationError(f"Total elements {total_elements} must be positive")
        if workgroup_size <= 0 or workgroup_size > self.MAX_WORKGROUP_SIZE:
            raise ValidationError(f"Workgroup size {workgroup_size} out of range")
        if workgroup_count <= 0:
            raise ValidationError(f"Workgroup count {workgroup_count} must be positive")
        expected = workgroup_size * workgroup_count
        if abs(total_elements - expected) > workgroup_size:
            raise ValidationError(
                f"Work quantity mismatch: {total_elements} elements vs {expected}"
            )
    
    def check_vulkan_result(self, result: int, operation: str) -> None:
        if result == 0:
            return
        error_map = {
            -1: "OUT_OF_HOST_MEMORY",
            -2: "OUT_OF_DEVICE_MEMORY",
            -3: "INITIALIZATION_FAILED",
            -4: "DEVICE_LOST",
            -5: "MEMORY_MAP_FAILED",
        }
        error_name = error_map.get(result, f"UNKNOWN_ERROR_{result}")
        if result == -4:
            raise DeviceLostError(f"{operation}: {error_name}")
        elif result == -2:
            raise SafetyError(f"{operation}: {error_name} - VRAM exhausted")
        elif result == -1:
            raise SafetyError(f"{operation}: {error_name} - System RAM exhausted")
        else:
            raise SafetyError(f"{operation}: {error_name} (code={result})")

class SafeVulkanExecutor:
    """Thread-safe Vulkan executor with cross-process watchdog enforcement."""
    
    def __init__(self, watchdog_client: Optional[CrossProcessWatchdogClient] = None):
        self.validator = VulkanSafetyValidator(watchdog_client)
        self._submission_lock = threading.Lock()
        self._last_fence_time = 0
        
    def _check_watchdog(self) -> None:
        """Check watchdog state before GPU work - FAIL CLOSED if unreachable."""
        if self.validator.watchdog:
            self.validator.watchdog.check_and_raise()
        else:
            # No watchdog configured - FAIL CLOSED for experimental execution
            raise SafetyError("No watchdog configured: GPU work halted")
    
    def _wait_for_slot(self) -> None:
        with self._submission_lock:
            while self.validator._in_flight_submissions >= self.validator._max_in_flight:
                waited = 0
                while self.validator._in_flight_submissions >= self.validator._max_in_flight:
                    if waited > 5000:
                        raise TimeoutError("No submission slot available within deadline")
                    time.sleep(0.001)
                    waited += 1
    
    def dispatch(self, kernel_name: str, buffers: List[BufferView], 
                 push_constants: bytes, workgroup_x: int, workgroup_y: int, 
                 workgroup_z: int, sync: bool = False, 
                 fence_timeout_ms: int = 5000) -> int:
        
        # Pre-submission checks
        self._check_watchdog()
        self._wait_for_slot()
        
        # Build and validate spec
        spec = DispatchSpec(
            kernel_name=kernel_name,
            buffers=buffers,
            push_constants=push_constants,
            workgroup_x=workgroup_x,
            workgroup_y=workgroup_y,
            workgroup_z=workgroup_z,
            fence_timeout_ms=fence_timeout_ms
        )
        
        self.validator.validate_dispatch(spec)
        for buf in buffers:
            self.validator.validate_buffer(buf, buf.size, buf.offset)
        
        # Submit with validation
        with self._submission_lock:
            self.validator._in_flight_submissions += 1
        
        try:
            lib = _load()
            buf_handles = (ctypes.c_ulonglong * len(buffers))(*[b._h for b in buffers])
            
            self._check_watchdog()
            
            result = lib.tvk_dispatch(
                kernel_name.encode('utf-8'),
                buf_handles, len(buffers),
                push_constants, len(push_constants),
                workgroup_x, workgroup_y, workgroup_z,
                1 if sync else 0
            )
            
            self.validator.check_vulkan_result(result, f"tvk_dispatch({kernel_name})")
            
            if sync:
                wait_result = lib.tvk_submit()
                self.validator.check_vulkan_result(wait_result, f"tvk_submit({kernel_name})")
            
            return result
            
        finally:
            with self._submission_lock:
                self.validator._in_flight_submissions -= 1
    
    def dispatch_async(self, kernel_name: str, buffers: List[BufferView], 
                       push_constants: bytes, workgroup_x: int, workgroup_y: int, 
                       workgroup_z: int, fence_timeout_ms: int = 5000) -> int:
        return self.dispatch(
            kernel_name, buffers, push_constants,
            workgroup_x, workgroup_y, workgroup_z,
            sync=False, fence_timeout_ms=fence_timeout_ms
        )
    
    def submit(self, timeout_ms: int = 5000) -> int:
        self._check_watchdog()
        lib = _load()
        result = lib.tvk_submit()
        self.validator.check_vulkan_result(result, "tvk_submit")
        return result
    
    def copy_to_device(self, src: BufferView, dst: BufferView) -> int:
        self._check_watchdog()
        self.validator.validate_buffer(src, src.size)
        self.validator.validate_buffer(dst, src.size)
        lib = _load()
        result = lib.tvk_copy_to_device(dst._alloc._h, src._alloc._h)
        self.validator.check_vulkan_result(result, "tvk_copy_to_device")
        return result
    
    def copy_from_device(self, src: BufferView, dst: BufferView) -> int:
        self._check_watchdog()
        self.validator.validate_buffer(src, src.size)
        self.validator.validate_buffer(dst, src.size)
        lib = _load()
        result = lib.tvk_copy_from_device(dst._alloc._h, src._alloc._h)
        self.validator.check_vulkan_result(result, "tvk_copy_from_device")
        return result

# Convenience functions
_global_executor: Optional[SafeVulkanExecutor] = None
_executor_lock = threading.Lock()

def get_safe_executor(watchdog_client=None) -> SafeVulkanExecutor:
    global _global_executor
    with _executor_lock:
        if _global_executor is None:
            _global_executor = SafeVulkanExecutor(watchdog_client)
        return _global_executor

def safe_dispatch(kernel_name: str, buffers: List[BufferView], 
                  push_constants: bytes, workgroup_x: int, workgroup_y: int, 
                  workgroup_z: int, sync: bool = False, 
                  fence_timeout_ms: int = 5000) -> int:
    executor = get_safe_executor()
    return executor.dispatch(
        kernel_name, buffers, push_constants,
        workgroup_x, workgroup_y, workgroup_z,
        sync, fence_timeout_ms
    )

def safe_dispatch_async(kernel_name: str, buffers: List[BufferView], 
                        push_constants: bytes, workgroup_x: int, workgroup_y: int, 
                        workgroup_z: int, fence_timeout_ms: int = 5000) -> int:
    executor = get_safe_executor()
    return executor.dispatch_async(
        kernel_name, buffers, push_constants,
        workgroup_x, workgroup_y, workgroup_z,
        fence_timeout_ms
    )

def safe_submit(timeout_ms: int = 5000) -> int:
    executor = get_safe_executor()
    return executor.submit(timeout_ms)

def safe_copy_to_device(src: BufferView, dst: BufferView) -> int:
    executor = get_safe_executor()
    return executor.copy_to_device(src, dst)

def safe_copy_from_device(src: BufferView, dst: BufferView) -> int:
    executor = get_safe_executor()
    return executor.copy_from_device(src, dst)

__all__ = [
    'SafetyError', 'ValidationError', 'DeviceLostError', 'TimeoutError',
    'WatchdogState', 'DispatchSpec', 'CrossProcessWatchdogClient',
    'VulkanSafetyValidator', 'SafeVulkanExecutor',
    'get_safe_executor', 'safe_dispatch', 'safe_dispatch_async',
    'safe_submit', 'safe_copy_to_device', 'safe_copy_from_device',
]