#!/usr/bin/env python3
"""Minimal CPU-only canonical smoke example.

Runs the canonical heterogeneous harness (test_100_latent_full.py) with
zero Vulkan layers for a short trajectory. Uses the existing public
runtime interfaces only — no second inference engine.

Usage:
    .venv/bin/python examples/cpu_only_smoke.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = dict(os.environ)
env["HYBRID_GPU_LAYERS"] = "0"
env["HYBRID_STEPS"] = "5"
proc = subprocess.run(
    [sys.executable, "-u", os.path.join(ROOT, "test_100_latent_full.py")],
    cwd=ROOT, env=env,
)
sys.exit(proc.returncode)
