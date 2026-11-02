"""Memory budget checker for sweep configurations."""
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryBudget:
    """Memory budget for a sweep configuration."""
    total_ram_gb: float
    available_ram_gb: float
    swap_total_gb: float
    swap_free_gb: float
    
    # Estimated requirements
    bf16_model_gb: float = 16.4
    int8_model_gb: float = 9.4
    vulkan_buffer_gb: float = 2.0
    kv_cache_gb: float = 0.5
    safety_margin_gb: float = 2.0
    
    @property
    def cpu_only_gb(self) -> float:
        """Memory for CPU-only INT8 inference."""
        return self.int8_model_gb + self.kv_cache_gb + self.safety_margin_gb
    
    @property
    def hybrid_1gpu_gb(self) -> float:
        """Memory for INT8 CPU + 1 GPU layer."""
        # Vulkan layer weights stay in VRAM, CPU layers are INT8
        return self.int8_model_gb + self.vulkan_buffer_gb + self.kv_cache_gb + self.safety_margin_gb
    
    @property
    def hybrid_6gpu_gb(self) -> float:
        """Memory for INT8 CPU + 6 GPU layers."""
        return self.int8_model_gb + 6 * self.vulkan_buffer_gb + self.kv_cache_gb + self.safety_margin_gb


def get_memory_info() -> dict:
    """Get system memory info from /proc/meminfo."""
    meminfo = {}
    with open('/proc/meminfo') as f:
        for line in f:
            if ':' in line:
                key, val = line.split(':', 1)
                val = val.strip().split()[0]
                try:
                    meminfo[key] = int(val)
                except ValueError:
                    pass
    return meminfo


def get_memory_budget() -> MemoryBudget:
    """Get current memory budget."""
    meminfo = get_memory_info()
    total_kb = meminfo.get('MemTotal', 0)
    available_kb = meminfo.get('MemAvailable', 0)
    swap_total_kb = meminfo.get('SwapTotal', 0)
    swap_free_kb = meminfo.get('SwapFree', 0)
    
    return MemoryBudget(
        total_ram_gb=total_kb / 1024 / 1024,
        available_ram_gb=available_kb / 1024 / 1024,
        swap_total_gb=swap_total_kb / 1024 / 1024,
        swap_free_gb=swap_free_kb / 1024 / 1024,
    )


def check_memory_budget(config_name: str, required_gb: float) -> tuple[bool, str]:
    """
    Check if configuration fits in memory budget.
    
    Returns:
        (can_run, status_message)
    """
    budget = get_memory_budget()
    available = budget.available_ram_gb
    
    if required_gb > available * 0.85:  # 15% headroom
        return False, f"INSUFFICIENT_MEMORY: {config_name} needs ~{required_gb:.1f}GB, available={available:.1f}GB"
    
    return True, f"OK: {config_name} needs ~{required_gb:.1f}GB, available={available:.1f}GB"


def estimate_config_memory(gpu_layers: int) -> float:
    """Estimate memory for a configuration."""
    budget = get_memory_budget()
    if gpu_layers == 0:
        return budget.cpu_only_gb
    elif gpu_layers <= 2:
        return budget.hybrid_1gpu_gb
    else:
        return budget.hybrid_6gpu_gb


def check_cgroup_oom() -> int:
    """Check cgroup OOM kill count."""
    try:
        with open('/sys/fs/cgroup/memory.events', 'r') as f:
            for line in f:
                if line.startswith('oom_kill'):
                    return int(line.split()[1])
    except:
        pass
    return -1