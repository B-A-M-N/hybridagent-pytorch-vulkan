"""Trajectory reward r(tau) = r_correct + r_format + r_tool: answer
correctness (Math-Verify for math, token-level F1 for search), a format penalty for an
invalid hybrid trajectory, and a bonus for correct answers that used both tools.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from hybrid_agent.constants import THINK_OPEN
from hybrid_rl.config import MAX_ANSWER_CHARS, R_FORMAT_FAIL, R_TOOL

_PUNCT = set(string.punctuation)
_EXACT_ONLY = ("yes", "no", "noanswer")


@dataclass
class Reward:
    total: float
    r_correct: float
    r_format: float
    r_tool: float
    pred: str | None
    format_reason: str = ""

    @property
    def correct(self) -> bool:
        return self.r_correct > 0.0


def validate_format(text: str) -> tuple[bool, str]:
    """Hybrid trajectory grammar: paired <think> and <latent> tags, one <answer> block,
    every tool call followed by its <result>, and a \\boxed{} final answer."""
    if text.count("<think>") != text.count("</think>"):
        return False, "<think> </think> not paired"
    if text.count("<think>") == 0 or text.count("</think>") == 0:
        return False, "<think> or </think> not found"
    if text.count("<latent>") != text.count("</latent>"):
        return False, "<latent> </latent> not paired"
    if text.count("<latent>") == 0 or text.count("</latent>") == 0:
        return False, "<latent> or </latent> not found"
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return False, "<answer> or </answer> not found"

    for open_tag, close_tag in (("<search>", "</search>"), ("<python>", "</python>")):
        current_pos = 0
        while True:
            open_pos = text.find(open_tag, current_pos)
            if open_pos == -1:
                break
            result_pos = text.find("<result>", open_pos)
            open_end_pos = text.find(close_tag, open_pos)
            result_end_pos = text.find("</result>", result_pos)
            if -1 in (result_pos, open_end_pos, result_end_pos):
                return False, f"{open_tag[1:-1]}/result tags are incomplete"
            if not (open_pos < open_end_pos < result_pos < result_end_pos):
                return False, f"{open_tag[1:-1]}/result tags are nested in the wrong order"
            current_pos = result_end_pos

    answer_start = text.find("<answer>")
    answer_end = text.find("</answer>")
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if "\\boxed{" not in answer_content or "}" not in answer_content:
        return False, "answer is missing \\boxed{} format"
    return True, "format is correct"


def extract_boxed_answer(text: str) -> str | None:
    """Content of the last balanced \\boxed{...}, inside <answer> when present."""
    answer_start = text.find("<answer>")
    answer_end = text.find("</answer>")
    scope = text[answer_start:answer_end] if 0 <= answer_start < answer_end else text
    idx = scope.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(scope) and depth > 0:
        c = scope[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip()


def math_verify_equal(pred: str, gold: str) -> bool:
    from math_verify import parse, verify

    pred = pred[:MAX_ANSWER_CHARS]
    gold = str(gold)[:MAX_ANSWER_CHARS]
    try:
        return bool(verify(parse(f"\\boxed{{{gold}}}"), parse(f"\\boxed{{{pred}}}")))
    except Exception:
        return False


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, golds: str | list[str]) -> float:
    """Token-level F1 against the best-matching gold answer (exact match for yes/no)."""
    if isinstance(golds, str):
        golds = [golds]
    pred = _normalize(prediction)
    best = 0.0
    for gold in golds:
        gold = _normalize(str(gold))
        if (pred in _EXACT_ONLY or gold in _EXACT_ONLY) and pred != gold:
            continue
        pred_tokens, gold_tokens = pred.split(), gold.split()
        common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def compute_reward(output: str, gold, task: str, *, prompt_ends_in_think: bool = False) -> Reward:
    # the format check sees the full response, including a <think> prefilled by the template
    checked = (THINK_OPEN + "\n" + output) if prompt_ends_in_think else output
    ok, reason = validate_format(checked)
    r_format = 0.0 if ok else R_FORMAT_FAIL

    pred = extract_boxed_answer(output)
    if pred is None:
        r_correct = 0.0
    elif task == "math":
        r_correct = 1.0 if math_verify_equal(pred, str(gold)) else 0.0
    else:
        r_correct = float(token_f1(pred, gold))

    r_tool = R_TOOL if (r_correct > 0.0 and "</search>" in output and "</python>" in output) else 0.0
    return Reward(
        total=r_correct + r_format + r_tool,
        r_correct=r_correct,
        r_format=r_format,
        r_tool=r_tool,
        pred=pred,
        format_reason="" if ok else reason,
    )
