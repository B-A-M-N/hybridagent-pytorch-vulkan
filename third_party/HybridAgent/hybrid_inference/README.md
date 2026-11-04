# Batch Hybrid Inference

`hybrid_inference.run_batch` runs HybridAgent on a JSONL file of questions with the Python
and search tools and writes one record per sample.

```bash
python -m hybrid_inference.run_batch --model <checkpoint dir or HF id> \
    --input examples/questions.jsonl --output outputs/results.jsonl --num_samples 4
```

Input records need a `question` field; an `id` and any other fields are carried through.
Samples of the same question are generated together and share one prefill of the working
memory. On several GPUs:

```bash
bash scripts/run_batch_inference.sh 8 --model <ckpt> --input questions.jsonl --output outputs/results.jsonl
```

which starts one process per GPU, shards the questions round-robin and merges the shards.
`--resume` skips samples already present in the output file.

## Configs

| Option | Default | Meaning |
|---|---|---|
| `--temperature / --top_p / --top_k` | 0.6 / 0.95 / 20 | sampling in text mode |
| `--max_response_tokens` | 4096 | per-sample budget of text tokens + latent steps (`--count_tool_tokens` also counts observations) |
| `--max_tool_calls`, `--max_search_calls`, `--max_python_calls` | 5 / 5 / 5 | tool-call caps; the sample ends at the cap unless `--continue_on_tool_cap` |
| `--min_latent_steps / --max_latent_steps` | 4 / 200 | bounds on each latent span |
| `--repetition_penalty` | 1.1 | penalty on repeated tokens (1.0 disables it) |
| `--search_max_results / --search_result_length` | 10 / 500 | snippets per search and characters per snippet |

For long agentic tasks use a larger budget and search cap, e.g. `--max_response_tokens 24000 --max_search_calls 10 --max_python_calls 5 --max_tool_calls 15 --continue_on_tool_cap --top_k 0`.

## Output record

```json
{"id": "q1", "sample_idx": 0, "question": "...",
 "output": "<think>\n...<latent></latent>...\n</think><python>...</python><result>...</result>...<answer>\\boxed{24}</answer>",
 "output_rendered": "... <latent>pseudo-text of the span</latent> ...",
 "prediction": "24",
 "latent_readouts": [{"steps": 37, "exit": "anchor", "pseudo_text": "..."}],
 "num_tool_calls": 1, "num_search_calls": 0, "num_python_calls": 1,
 "generation_steps": 412, "latent_steps": 37, "latent_spans": 1,
 "end_reason": "eos", "wall_time_s": 9.8}
```

`output` is the trajectory as generated (latent spans decode to empty anchors); `output_rendered` fills each span with its pseudo-text readout. `end_reason` is `eos`, `budget`, `tool_cap` or `max_turns`.
