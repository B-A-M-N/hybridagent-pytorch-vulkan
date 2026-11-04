"""Renderers for hybrid traces: a self-contained HTML report that shows the latent thoughts
as pseudo-text inline with the text reasoning, tool calls and observations (token-level
readability colouring, per-step nearest-token candidates, light and dark themes), and a
coloured terminal view.
"""
from __future__ import annotations

import html
import shutil
import textwrap
from datetime import datetime

from .trace import HybridTrace, Segment

_KIND_LABEL = {
    "text": "text mode",
    "latent": "latent mode",
    "tool_call": "tool call",
    "tool_result": "observation",
    "answer": "answer",
}

_CSS = """
:root {
  --bg: #f6f7fb; --card: #ffffff; --ink: #1b1f2a; --muted: #6b7280; --line: #e6e8ef;
  --text-bg: #f3f4f8; --latent-bg: #f1ecff; --latent-ink: #3b1d8a; --latent-line: #b9a6ff; --latent-tok: 124, 58, 237;
  --tool-bg: #e9f3ff; --tool-ink: #0b4f8a; --obs-bg: #f1f5f3; --obs-ink: #334155;
  --answer-bg: #e8f8ee; --answer-ink: #14532d; --chip: #eef0f5; --accent: #7c3aed; --accent2: #2563eb;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1016; --card: #161923; --ink: #e7e9f0; --muted: #98a2b3; --line: #262b38;
    --text-bg: #1b1f2b; --latent-bg: #22193f; --latent-ink: #d9ccff; --latent-line: #5b3fb0; --latent-tok: 167, 139, 250;
    --tool-bg: #11263c; --tool-ink: #9ccbff; --obs-bg: #18211d; --obs-ink: #c9d4cd;
    --answer-bg: #12301f; --answer-ink: #b8f1c8; --chip: #222836; --accent: #a78bfa; --accent2: #60a5fa;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1016; --card: #161923; --ink: #e7e9f0; --muted: #98a2b3; --line: #262b38;
  --text-bg: #1b1f2b; --latent-bg: #22193f; --latent-ink: #d9ccff; --latent-line: #5b3fb0; --latent-tok: 167, 139, 250;
  --tool-bg: #11263c; --tool-ink: #9ccbff; --obs-bg: #18211d; --obs-ink: #c9d4cd;
  --answer-bg: #12301f; --answer-ink: #b8f1c8; --chip: #222836; --accent: #a78bfa; --accent2: #60a5fa;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.55; }
main { max-width: 980px; margin: 0 auto; padding: 28px 16px 64px; }
.hero { border-radius: 16px; padding: 22px 24px; color: #fff; margin-bottom: 18px;
        background: linear-gradient(120deg, #4c1d95 0%, #6d28d9 45%, #2563eb 100%); }
.hero h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: 0.2px; }
.hero .sub { opacity: 0.9; font-size: 14px; }
.hero .meta { margin-top: 10px; font-size: 12px; opacity: 0.85; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 18px; }
.toolbar button { font: inherit; font-size: 12px; padding: 5px 11px; border-radius: 999px; border: 1px solid var(--line);
                  background: var(--card); color: var(--ink); cursor: pointer; }
.toolbar button:hover { border-color: var(--accent); }
.legend { display: flex; flex-wrap: wrap; gap: 6px; margin-left: auto; }
.legend span, .chip { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; background: var(--chip); }
.legend .l-latent { background: var(--latent-bg); color: var(--latent-ink); }
.legend .l-text { background: var(--text-bg); }
.legend .l-tool { background: var(--tool-bg); color: var(--tool-ink); }
.legend .l-obs { background: var(--obs-bg); color: var(--obs-ink); }
.legend .l-answer { background: var(--answer-bg); color: var(--answer-ink); }
.trace { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 20px 22px; margin-bottom: 24px;
         box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.trace h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); margin: 0 0 6px; }
.trace .question { font-size: 16px; font-weight: 600; margin: 0 0 10px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.chips .chip b { font-weight: 600; }
.chip.pred { background: var(--answer-bg); color: var(--answer-ink); }
.modebar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 4px 0 16px; background: var(--text-bg); }
.modebar .lat { background: var(--accent); }
.modebar .txt { background: var(--accent2); opacity: 0.55; }
.modelabel { font-size: 11px; color: var(--muted); margin-top: -12px; margin-bottom: 14px; }
.seg { border-radius: 12px; padding: 10px 14px; margin: 8px 0; font-size: 14px; white-space: pre-wrap; word-break: break-word; }
.seg .tag { display: flex; justify-content: space-between; gap: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px;
            color: var(--muted); margin-bottom: 5px; white-space: normal; }
.seg.text { background: var(--text-bg); }
.seg.latent { background: var(--latent-bg); color: var(--latent-ink); border-left: 4px solid var(--latent-line); }
.seg.latent .pseudo { font-family: var(--mono); font-size: 13px; line-height: 1.8; }
.seg.latent .tag { color: var(--latent-ink); opacity: 0.8; }
.tok { border-radius: 3px; padding: 1px 0; background: rgba(var(--latent-tok), calc(var(--c, 0) * 0.35)); cursor: help; }
.seg.tool_call { background: var(--tool-bg); color: var(--tool-ink); font-family: var(--mono); font-size: 13px; }
.seg.tool_result { background: var(--obs-bg); color: var(--obs-ink); font-family: var(--mono); font-size: 12.5px; }
.seg.tool_result details summary { color: var(--muted); font-family: var(--sans); font-size: 12px; cursor: pointer; }
.seg.answer { background: var(--answer-bg); color: var(--answer-ink); font-weight: 500; }
details.steps { margin-top: 8px; }
details.steps summary { cursor: pointer; font-size: 12px; color: var(--muted); }
table.steps { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; font-family: var(--mono); }
table.steps th, table.steps td { text-align: left; padding: 3px 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.steps th { color: var(--muted); font-weight: 500; }
.bar { display: inline-block; height: 8px; background: var(--accent); border-radius: 4px; vertical-align: middle; margin-right: 6px; opacity: 0.85; }
.alts { color: var(--muted); }
footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
"""

_JS = """
function setOpen(open) { document.querySelectorAll('details.steps').forEach(d => d.open = open); }
function toggleTheme() {
  const r = document.documentElement; const cur = r.getAttribute('data-theme');
  const dark = cur ? cur === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  r.setAttribute('data-theme', dark ? 'light' : 'dark');
}
"""


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _chip(label: str, value, cls: str = "") -> str:
    return f'<span class="chip {cls}">{_esc(label)} <b>{_esc(str(value))}</b></span>'


def _readability(seg: Segment) -> float | None:
    if not seg.steps:
        return None
    return sum(s.cosine for s in seg.steps) / len(seg.steps)


def _latent_html(seg: Segment, index: int) -> str:
    read = _readability(seg)
    head_right = []
    if seg.steps:
        head_right.append(f"{len(seg.steps)} steps")
    if read is not None:
        head_right.append(f"readability {read:.2f}")
    if seg.exit:
        head_right.append("exit anchor" if seg.exit == "anchor" else f"closed: {seg.exit}")
    head = (f'<span class="tag"><span>latent span {index}</span>'
            f'<span>{" · ".join(_esc(h) for h in head_right)}</span></span>')
    if seg.steps:
        toks = []
        for step in seg.steps:
            alts = ", ".join(f"{t} ({c:.2f})" for t, _, c in step.candidates[1:])
            title = f"step {step.index} · cos {step.cosine:.3f}" + (f" · also: {alts}" if alts else "")
            toks.append(f'<span class="tok" style="--c:{max(min(step.cosine, 1.0), 0.0):.3f}" title="{_esc(title)}">{_esc(step.token)}</span>')
        pseudo = "".join(toks)
    else:
        pseudo = _esc(seg.pseudo_text) or "&nbsp;"
    parts = [head, f'<span class="pseudo">{pseudo}</span>']
    if seg.steps and seg.steps[0].candidates:
        rows = []
        for step in seg.steps:
            width = max(2, int(60 * max(min(step.cosine, 1.0), 0.0)))
            alts = ", ".join(f"{_esc(t)} ({c:.2f})" for t, _, c in step.candidates[1:])
            rows.append(f"<tr><td>{step.index}</td><td>{_esc(step.token)}</td>"
                        f'<td><span class="bar" style="width:{width}px"></span>{step.cosine:.3f}</td>'
                        f'<td class="alts">{alts}</td></tr>')
        parts.append('<details class="steps"><summary>per-step nearest tokens</summary>'
                     '<table class="steps"><tr><th>step</th><th>nearest token</th><th>cosine</th><th>alternatives</th></tr>'
                     + "".join(rows) + "</table></details>")
    return f'<div class="seg latent">{"".join(parts)}</div>'


def _segment_html(seg: Segment, latent_counter: list[int]) -> str:
    if seg.kind == "latent":
        latent_counter[0] += 1
        return _latent_html(seg, latent_counter[0])
    label = _KIND_LABEL[seg.kind] + (f" · {seg.tool}" if seg.tool else "")
    body = seg.text.strip("\n")
    if seg.kind == "tool_result" and len(body) > 700:
        body_html = f"{_esc(body[:700])}…<details><summary>show the full observation ({len(body):,} characters)</summary>{_esc(body)}</details>"
    else:
        body_html = _esc(body)
    return f'<div class="seg {seg.kind}"><span class="tag"><span>{_esc(label)}</span></span>{body_html}</div>'


def trace_html(trace: HybridTrace, index: int = 1) -> str:
    chips = [_chip("answer", trace.prediction or "—", "pred")]
    for key in ("latent_spans", "latent_steps", "text_tokens", "search_calls", "python_calls", "end_reason"):
        if key in trace.stats:
            chips.append(_chip(key.replace("_", " "), trace.stats[key]))
    latent = int(trace.stats.get("latent_steps", 0) or 0)
    text = int(trace.stats.get("text_tokens", 0) or 0)
    total = max(latent + text, 1)
    modebar = (f'<div class="modebar"><span class="lat" style="width:{100 * latent / total:.1f}%"></span>'
               f'<span class="txt" style="width:{100 * text / total:.1f}%"></span></div>'
               f'<div class="modelabel">{100 * latent / total:.0f}% of the generation steps in latent mode</div>')
    counter = [0]
    segments = "".join(_segment_html(s, counter) for s in trace.segments)
    title = f"Trace {index}" + (f" · sample {trace.sample_idx}" if trace.sample_idx else "")
    return (f'<section class="trace"><h2>{_esc(title)}</h2><p class="question">{_esc(trace.question)}</p>'
            f'<div class="chips">{"".join(chips)}</div>{modebar}{segments}</section>')


def render_html(traces: list[HybridTrace], title: str = "HybridAgent · Interpretable Latent Mode") -> str:
    model = next((t.model for t in traces if t.model), "")
    n_spans = sum(len(t.latent_segments) for t in traces)
    reads = [r for t in traces for s in t.latent_segments if (r := _readability(s)) is not None]
    meta = [f"{len(traces)} trace(s)", f"{n_spans} latent span(s)"]
    if reads:
        meta.append(f"mean readability {sum(reads) / len(reads):.2f}")
    if model:
        meta.append(f"model {model}")
    hero = (f'<div class="hero"><h1>{_esc(title)}</h1>'
            '<div class="sub">Latent thoughts rendered as pseudo-text, inline with the text reasoning, tool calls and observations. '
            'Hover a latent token to see its nearest-token cosine and alternatives.</div>'
            f'<div class="meta">{" · ".join(_esc(m) for m in meta)}</div></div>')
    toolbar = ('<div class="toolbar"><button onclick="setOpen(true)">expand all steps</button>'
               '<button onclick="setOpen(false)">collapse all</button><button onclick="toggleTheme()">light / dark</button>'
               '<div class="legend"><span class="l-text">text mode</span><span class="l-latent">latent mode</span>'
               '<span class="l-tool">tool call</span><span class="l-obs">observation</span><span class="l-answer">answer</span></div></div>')
    body = "".join(trace_html(t, i + 1) for i, t in enumerate(traces))
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{_esc(title)}</title><style>{_CSS}</style><script>{_JS}</script></head><body><main>"
            f"{hero}{toolbar}{body}<footer>generated {stamp}</footer></main></body></html>")


# ----------------------------------------------------------------------------- terminal

_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "text": "\033[37m", "latent": "\033[95m", "latent_bg": "\033[45;97m", "tool_call": "\033[94m",
    "tool_result": "\033[90m", "answer": "\033[92m", "title": "\033[1;95m", "key": "\033[36m",
}
_GUTTER = {"text": "text   ", "latent": "latent ", "tool_call": "tool   ", "tool_result": "result ", "answer": "answer "}


def render_text(trace: HybridTrace, color: bool = True, show_steps: bool = False, width: int | None = None) -> str:
    """Terminal view with a coloured gutter per segment; latent spans show their pseudo-text,
    step count and readability (`show_steps` adds the per-step nearest tokens)."""
    def c(name: str, text: str) -> str:
        return f"{_C[name]}{text}{_C['reset']}" if color else text

    width = width or min(shutil.get_terminal_size((100, 20)).columns, 120)
    inner = max(40, width - 10)
    lines: list[str] = []
    bar = "─" * (width - 2)
    lines.append(c("title", f"┌{bar}┐"))
    lines.append(c("title", "│ ") + c("bold", "HybridAgent · Interpretable Latent Mode".ljust(width - 4)) + c("title", " │"))
    lines.append(c("title", f"└{bar}┘"))
    lines.append(c("key", "question ") + c("bold", trace.question))
    stats = "  ".join(f"{c('dim', k.replace('_', ' '))} {v}" for k, v in trace.stats.items())
    lines.append(stats)
    lines.append("")

    def block(kind: str, body: str, wrap: bool = True) -> None:
        gutter = _GUTTER[kind]
        paragraphs = body.strip("\n").split("\n") if body.strip() else [""]
        first = True
        for para in paragraphs:
            chunks = textwrap.wrap(para, inner, break_long_words=False, replace_whitespace=False) if wrap else [para]
            for chunk in chunks or [""]:
                prefix = c(kind, gutter if first else " " * len(gutter)) + c(kind, "│ ")
                lines.append(prefix + c(kind, chunk))
                first = False

    counter = 0
    for seg in trace.segments:
        if seg.kind == "latent":
            counter += 1
            read = _readability(seg)
            head = f"⟨latent span {counter}"
            if seg.steps:
                head += f" · {len(seg.steps)} steps"
            if read is not None:
                head += f" · readability {read:.2f}"
            head += "⟩"
            pseudo = seg.pseudo_text or "".join(s.token for s in seg.steps)
            block("latent", head + "\n" + pseudo)
            if show_steps and seg.steps:
                for step in seg.steps:
                    alts = ", ".join(f"{t}({cs:.2f})" for t, _, cs in step.candidates[1:])
                    lines.append(c("dim", f"       │   {step.index:3d}  {step.cosine:.3f}  {step.token!r}" + (f"  {alts}" if alts else "")))
        elif seg.kind == "tool_call":
            block("tool_call", f"<{seg.tool}>\n{seg.text.strip()}\n</{seg.tool}>", wrap=False)
        elif seg.kind == "tool_result":
            body = seg.text.strip()
            if len(body) > 600:
                body = body[:600] + f"… [{len(seg.text):,} characters]"
            block("tool_result", body)
        elif seg.kind == "answer":
            block("answer", seg.text.strip())
        else:
            block("text", seg.text)
    lines.append("")
    lines.append(c("key", "answer   ") + c("bold", trace.prediction or "—"))
    return "\n".join(lines)
