"""HybridRL prompts: a parquet file (local path or `hf://<dataset repo>/<file>`) with
`extra_info.question`, `reward_model.ground_truth` and `ability` columns, streamed as
deterministic shuffled batches of unique prompts.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Prompt:
    idx: int
    question: str
    gold: str

    def to_json(self) -> str:
        return json.dumps({"idx": self.idx, "question": self.question, "gold": self.gold})


def resolve_data_path(spec: str) -> Path:
    """Local file as is; `hf://<repo_id>/<filename>` downloads from a dataset repo."""
    if spec.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo_id, _, filename = spec[len("hf://"):].partition("/")
        owner, _, rest = filename.partition("/")
        if rest:  # repo ids have the form <owner>/<name>
            repo_id, filename = f"{repo_id}/{owner}", rest
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))
    return Path(spec)


def _coerce(v):
    if isinstance(v, dict):
        return v
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def load_prompts(path: str | Path, ability: str) -> list[Prompt]:
    """All rows of the expected ability; malformed rows are skipped."""
    import pandas as pd

    df = pd.read_parquet(path)
    missing = {"ability", "reward_model", "extra_info"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    out: list[Prompt] = []
    for i, row in df.iterrows():
        if str(row["ability"]) != ability:
            raise ValueError(f"row {i}: ability {row['ability']!r} does not match the task ({ability!r})")
        gold = (_coerce(row["reward_model"]) or {}).get("ground_truth")
        question = (_coerce(row["extra_info"]) or {}).get("question")
        if not question or gold is None:
            continue
        out.append(Prompt(idx=int(i), question=str(question), gold=str(gold)))
    if not out:
        raise ValueError(f"no usable rows in {path}")
    return out


def epoch_batches(prompts: list[Prompt], prompts_per_step: int, seed: int) -> Iterator[list[Prompt]]:
    """Endless stream of complete prompt batches, reshuffled every epoch."""
    if prompts_per_step < 1:
        raise ValueError("prompts_per_step must be positive")
    if len(prompts) < prompts_per_step:
        raise ValueError(f"need at least {prompts_per_step} prompts for one step, got {len(prompts)}")
    rng = random.Random(seed)
    order = list(range(len(prompts)))
    while True:
        rng.shuffle(order)
        for start in range(0, len(order) - prompts_per_step + 1, prompts_per_step):
            yield [prompts[index] for index in order[start : start + prompts_per_step]]
