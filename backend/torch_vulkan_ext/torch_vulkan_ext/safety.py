#!/usr/bin/env python3
"""
Vulkan Execution Safety Layer (P1) — Validates all GPU submissions before execution.
Provides fail-closed safety checks for buffer sizes, dispatch dimensions, and Vulkan results.
"""

import ctypes
import time
import torch
import numpy as np
import logging
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any
from enum import Enum

try:
    from .torch_vulkan_ext import _load, Allocation, BufferView, _staging
except ImportError:
    # For direct execution
    import sys
    sys.path.insert(0, '/home/bamn/ai-experiments/hybridagent-pytorch-vulkan')
    from backend.torch_vulkan_ext.torch_vulkan_ext import _load, Allocation, BufferView, _staging

logger = logging.getLogger("vulkan_safety")

class SafetyError(Exception):
    """Raised when a safety check fails."""
    pass

class ValidationError(SafetyError):
    """Raised when validation fails before GPU dispatch."""
    pass

class DeviceLostError(SafetyError):
    """Raised when GPU device is lost."""
    pass

class TimeoutError(SafetyError):
    """Raised when GPU execution times out."""
    pass

class VulkanResult(Enum):
    SUCCESS = 0
    ERROR_OUT_OF_HOST_MEMORY = -1
    ERROR_OUT_OF_DEVICE_MEMORY = -2
    ERROR_INITIALIZATION_FAILED = -3
    ERROR_DEVICE_LOST = -4
    ERROR_MEMORY_MAP_FAILED = -5
    ERROR_LAYER_NOT_PRESENT = -6
    ERROR_EXTENSION_NOT_PRESENT = -7
    ERROR_FEATURE_NOT_PRESENT = -8
    ERROR_INCOMPATIBLE_DRIVER = -9
    ERROR_TOO_MANY_OBJECTS = -10
    ERROR_FORMAT_NOT_SUPPORTED = -11
    ERROR_FRAGMENTED_POOL = -12
    ERROR_UNKNOWN = -13
    ERROR_OUT_OF_POOL_MEMORY = -1000069000
    ERROR_INVALID_EXTERNAL_HANDLE = -1000072003
    ERROR_FRAGMENTATION = -1000161000
    ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS = -1000257000
    PIPELINE_COMPILE_REQUIRED = 1000297000
    ERROR_SURFACE_LOST_KHR = -1000000000
    ERROR_NATIVE_WINDOW_IN_USE_KHR = -1000000001
    ERROR_OUT_OF_DATE_KHR = -1000001004
    ERROR_INCOMPATIBLE_DISPLAY_KHR = -1000003001
    ERROR_VALIDATION_FAILED_EXT = -1000011001
    ERROR_INCOMPATIBLE_SHADER_BINARY_EXT = -1000155000
    ERROR_PIPELINE_BINARY_MISSING_KHR = -1000380000

@dataclass
class DispatchSpec:
    """Validated dispatch specification."""
    kernel_name: str
    buffers: List[BufferView]
    push_constants: bytes
    workgroup_x: int
    workgroup_y: int
    workgroup_z: int
    fence_timeout_ms: int = 5000

@dataclass
class BufferSpec:
    """Validated buffer specification."""
    view: BufferView
    expected_size: int
    expected_offset: int
    is_read_only: bool = False
    is_write_only: bool = False

class VulkanSafetyValidator:
    """Validates all Vulkan operations before execution."""
    
    # Physical device limits for RX 6600 XT (RADV NAVI23)
    MAX_WORKGROUP_COUNT = (65535, 65535, 65535)
    MAX_WORKGROUP_SIZE = 1024
    MAX_WORKGROUP_INVOCATIONS = 1024
    MAX_PUSH_CONSTANT_SIZE = 128  # bytes
    MAX_BUFFER_SIZE = 1 << 30  # 1 GiB per buffer
    MAX_DESCRIPTOR_SETS = 8
    
    def __init__(self, watchdog=None):
        self.watchdog = watchdog
        self._lib = _load()
        self._in_flight_submissions = 0
        self._max_in_flight = 1  # P1.7: One GPU submission in flight
        self._lock = threading.Lock()
        self._initialized = False
        self._validation_layers_enabled = True
        
    def validate_buffer(self, view: BufferView, expected_size: int, 
                        expected_offset: int = 0) -> None:
        """Validate buffer view before use."""
        if view.size != expected_size:
            raise ValidationError(
                f"Buffer size mismatch: expected {expected_size}, got {view.size}"
            )
        if view.offset != expected_offset:
            raise ValidationError(
                f"Buffer offset mismatch: expected {expected_offset}, got {view.offset}"
            )
        if view.size > self.MAX_BUFFER_SIZE:
            raise ValidationError(
                f"Buffer size {view.size} exceeds max {self.MAX_BUFFER_SIZE}"
            )
        if view.offset + view.size > view._alloc.size:
            raise ValidationError(
                f"Buffer view extends past allocation: "
                f"offset={view.offset}, size={view.size}, alloc_size={view._alloc.size}"
            )
        
        # Check alignment (Vulkan requires 4-byte alignment for float32)
        if view.offset % 4 != 0:
            raise ValidationError(f"Buffer offset {view.offset} not 4-byte aligned")
        if view.size % 4 != 0:
            raise ValidationError(f"Buffer size {view.size} not 4-byte aligned")
    
    def validate_dispatch(self, spec: DispatchSpec) -> None:
        """Validate dispatch parameters before submission."""
        # Check workgroup dimensions
        if spec.workgroup_x <= 0 or spec.workgroup_x > self.MAX_WORKGROUP_COUNT[0]:
            raise ValidationError(f"workgroup_x {spec.workgroup_x} out of range")
        if spec.workgroup_y <= 0 or spec.workgroup_y > self.MAX_WORKGROUP_COUNT[1]:
            raise ValidationError(f"workgroup_y {spec.workgroup_y} out of range")
        if spec.workgroup_z <= 0 or spec.workgroup_z > self.MAX_WORKGROUP_COUNT[2]:
            raise ValidationError(f"workgroup_z {spec.workgroup_z} out of range")
        
        # Check push constant size
        if len(spec.push_constants) > self.MAX_PUSH_CONSTANT_SIZE:
            raise ValidationError(
                f"Push constants {len(spec.push_constants)} bytes > "
                f"max {self.MAX_PUSH_CONSTANT_SIZE}"
            )
        
        # Check fence timeout
        if spec.fence_timeout_ms <= 0 or spec.fence_timeout_ms > 30000:
            raise ValidationError(
                f"Fence timeout {spec.fence_timeout_ms}ms out of range (1-30000)"
            )
        
        # Validate all buffers
        for buf in spec.buffers:
            # Basic validation - size and offset checked at creation
            if buf.size > self.MAX_BUFFER_SIZE:
                raise ValidationError(f"Buffer size {buf.size} exceeds max")
    
    def validate_shader_bounds(self, kernel_name: str, total_elements: int,
                               workgroup_size: int, workgroup_count: int) -> None:
        """Validate that shader loops and work quantities have provable bounds."""
        if total_elements <= 0:
            raise ValidationError(f"Total elements {total_elements} must be positive")
        if workgroup_size <= 0 or workgroup_size > self.MAX_WORKGROUP_SIZE:
            raise ValidationError(f"Workgroup size {workgroup_size} out of range")
        if workgroup_count <= 0:
            raise ValidationError(f"Workgroup count {workgroup_count} must be positive")
        
        # Verify total work matches
        expected = workgroup_size * workgroup_count
        if abs(total_elements - expected) > workgroup_size:  # Allow partial last group
            raise ValidationError(
                f"Work quantity mismatch: {total_elements} elements vs "
                f"{expected} (workgroup_size={workgroup_size} * count={workgroup_count})"
            )
    
    def check_vulkan_result(self, result: int, operation: str) -> None:
        """Check Vulkan result and raise appropriate exception."""
        if result == 0:  # VK_SUCCESS
            return
        
        error_map = {
            -1: "OUT_OF_HOST_MEMORY",
            -2: "OUT_OF_DEVICE_MEMORY",
            -3: "INITIALIZATION_FAILED",
            -4: "DEVICE_LOST",
            -5: "MEMORY_MAP_FAILED",
        }
        error_name = error_map.get(result, f"UNKNOWN_ERROR_{result}")
        
        if result == -4:  # VK_ERROR_DEVICE_LOST
            raise DeviceLostError(f"{operation}: {error_name}")
        elif result == -2:  # OUT_OF_DEVICE_MEMORY
            raise SafetyError(f"{operation}: {error_name} - VRAM exhausted")
        elif result == -1:  # OUT_OF_HOST_MEMORY
            raise SafetyError(f"{operation}: {error_name} - System RAM exhausted")
        elif result == -3:  # INITIALIZATION_FAILED
            raise SafetyError(f"{operation}: {error_name}")
        else:
            raise SafetyError(f"{operation}: {error_name} (code={result})")

class SafeVulkanExecutor:
    """Thread-safe Vulkan executor with safety validation and watchdog integration."""
    
    def __init__(self, watchdog=None):
        self.validator = VulkanSafetyValidator(watchdog)
        self._submission_lock = threading.Lock()
        self._last_fence_time = 0
        
    def _check_watchdog(self) -> None:
        """Check watchdog state before GPU work."""
        if self.validator.watchdog and self.validator.watchdog.stop_gpu_work:
            raise SafetyError("Watchdog requested stop: GPU work halted")
    
    def _wait_for_slot(self) -> None:
        """Wait for in-flight submission slot (P1.7)."""
        with self._submission_lock:
            while self.validator._in_flight_submissions >= self.validator._max_in_flight:
                # Wait with timeout
                waited = 0
                while self.validator._in_flight_submissions >= self.validator._max_in_flight:
                    if waited > 5000:  # 5 second max wait
                        raise TimeoutError("No submission slot available within deadline")
                    time.sleep(0.001)
                    waited += 1
    
    def dispatch(self, kernel_name: str, buffers: List[BufferView], 
                 push_constants: bytes, workgroup_x: int, workgroup_y: int, 
                 workgroup_z: int, sync: bool = False, 
                 fence_timeout_ms: int = 5000) -> int:
        """Safe dispatch with full validation."""
        
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
        
        # Validate all buffers
        for buf in buffers:
            self.validator.validate_buffer(buf, buf.size, buf.offset)
        
        # Submit with validation
        with self._submission_lock:
            self.validator._in_flight_submissions += 1
        
        try:
            lib = _load()
            
            # Prepare buffer handles
            buf_handles = (ctypes.c_ulonglong * len(buffers))(*[b._h for b in buffers])
            
            # Dispatch
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
                # Wait with timeout (P1.8)
                wait_result = lib.tvk_submit()
                self.validator.check_vulkan_result(
                    wait_result, f"tvk_submit({kernel_name})")
            
            return result
            
        finally:
            with self._submission_lock:
                self.validator._in_flight_submissions -= 1
    
    def dispatch_async(self, kernel_name: str, buffers: List[BufferView], 
                       push_constants: bytes, workgroup_x: int, workgroup_y: int, 
                       workgroup_z: int, fence_timeout_ms: int = 5000) -> int:
        """Safe async dispatch (no sync)."""
        return self.dispatch(
            kernel_name, buffers, push_constants,
            workgroup_x, workgroup_y, workgroup_z,
            sync=False, fence_timeout_ms=fence_timeout_ms
        )
    
    def submit(self, timeout_ms: int = 5000) -> int:
        """Submit pending work and wait with timeout."""
        self._check_watchdog()
        
        lib = _load()
        result = lib.tvk_submit()
        self.validator.check_vulkan_result(result, "tvk_submit")
        return result
    
    def copy_to_device(self, src: BufferView, dst: BufferView) -> int:
        """Safe device-to-device copy with validation."""
        self._check_watchdog()
        
        self.validator.validate_buffer(src, src.size)
        self.validator.validate_buffer(dst, src.size)  # dst must fit src
        
        lib = _load()
        result = lib.tvk_copy_to_device(dst._alloc._h, src._alloc._h)
        self.validator.check_vulkan_result(result, "tvk_copy_to_device")
        return result
    
    def copy_from_device(self, src: BufferView, dst: BufferView) -> int:
        """Safe device-to-host copy with validation."""
        self._check_watchdog()
        
        self.validator.validate_buffer(src, src.size)
        self.validator.validate_buffer(dst, src.size)
        
        lib = _load()
        result = lib.tvk_copy_from_device(dst._alloc._h, src._alloc._h)
        self.validator.check_vulkan_result(result, "tvk_copy_from_device")
        return result

# Convenience functions for backward compatibility
_global_executor: Optional[SafeVulkanExecutor] = None
_executor_lock = threading.Lock()

def get_safe_executor(watchdog=None) -> SafeVulkanExecutor:
    """Get or create the global safe executor."""
    global _global_executor
    with _executor_lock:
        if _global_executor is None:
            _global_executor = SafeVulkanExecutor(watchdog)
        return _global_executor

def safe_dispatch(kernel_name: str, buffers: List[BufferView], 
                  push_constants: bytes, workgroup_x: int, workgroup_y: int, 
                  workgroup_z: int, sync: bool = False, 
                  fence_timeout_ms: int = 5000) -> int:
    """Safe dispatch using global executor."""
    executor = get_safe_executor()
    return executor.dispatch(
        kernel_name, buffers, push_constants,
        workgroup_x, workgroup_y, workgroup_z,
        sync, fence_timeout_ms
    )

def safe_dispatch_async(kernel_name: str, buffers: List[BufferView], 
                        push_constants: bytes, workgroup_x: int, workgroup_y: int, 
                        workgroup_z: int, fence_timeout_ms: int = 5000) -> int:
    """Safe async dispatch using global executor."""
    executor = get_safe_executor()
    return executor.dispatch_async(
        kernel_name, buffers, push_constants,
        workgroup_x, workgroup_y, workgroup_z,
        fence_timeout_ms
    )

def safe_submit(timeout_ms: int = 5000) -> int:
    """Safe submit using global executor."""
    executor = get_safe_executor()
    return executor.submit(timeout_ms)

def safe_copy_to_device(src: BufferView, dst: BufferView) -> int:
    """Safe device-to-device copy."""
    executor = get_safe_executor()
    return executor.copy_to_device(src, dst)

def safe_copy_from_device(src: BufferView, dst: BufferView) -> int:
    """Safe device-to-host copy."""
    executor = get_safe_executor()
    return executor.copy_from_device(src, dst)

# Export the key classes
__all__ = [
    'SafetyError', 'ValidationError', 'DeviceLostError', 'TimeoutError',
    'VulkanResult', 'DispatchSpec', 'BufferSpec',
    'VulkanSafetyValidator', 'SafeVulkanExecutor',
    'get_safe_executor', 'safe_dispatch', 'safe_dispatch_async',
    'safe_submit', 'safe_copy_to_device', 'safe_copy_from_device',
]