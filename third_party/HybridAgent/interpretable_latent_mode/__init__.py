"""Interpretable Latent Mode: a plug-and-play way to read what HybridAgent thinks in latent
mode. Every latent step is mapped to its nearest token embeddings and rendered as pseudo-text
next to the text reasoning, tool calls and observations, as an HTML report or in the terminal.
"""
from .render import render_html, render_text, trace_html
from .trace import GENERATION_DEFAULTS, HybridTrace, LatentRecordingEngine, LatentStep, Segment, inspect, parse_segments, trace_from_record

__all__ = [
    "GENERATION_DEFAULTS", "HybridTrace", "LatentRecordingEngine", "LatentStep", "Segment",
    "inspect", "parse_segments", "render_html", "render_text", "trace_from_record", "trace_html",
]
