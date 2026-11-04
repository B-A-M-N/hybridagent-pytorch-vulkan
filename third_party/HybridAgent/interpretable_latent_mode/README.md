# Interpretable Latent Mode

HybridAgent's latent mode is decoding-free yet directly inspectable: mapping each latent
step produced by HybridLink to its nearest token embedding renders a latent span as
readable pseudo-text that stays aligned with the surrounding text reasoning and tool calls.
This package turns that readout into a plug-and-play inspector.

```bash
# answer a question and write an HTML report of the hybrid trace
python -m interpretable_latent_mode --model <checkpoint dir or HF id> \
    --question "How many trailing zeros does 100! have?" --output report.html

# several questions, several samples each, with the top-3 candidates per latent step
python -m interpretable_latent_mode --model <ckpt> --input examples/questions.jsonl \
    --num_samples 2 --readout_top_k 3 --output report.html

# render the records written by hybrid_inference.run_batch
python -m interpretable_latent_mode --input outputs/results.jsonl --output report.html

# terminal view instead of a report (--show_steps lists the nearest token of every latent step)
python -m interpretable_latent_mode --model <ckpt> --question "..." --show_steps
```

`--open` opens the HTML report in a browser once it is written.

The HTML report is self-contained (no external assets, light and dark themes). For every
trace it shows the question, the extracted answer, the usage summary and the share of
generation steps spent in latent mode, then the trajectory as a sequence of segments:

- **text mode** segments as written by the agent, including the `<think>` reasoning;
- **latent mode** segments as pseudo-text: every token is shaded by its nearest-token cosine similarity (the readability of that step) and shows its alternative candidates on hover; the span header gives the number of latent steps, the mean readability and how the span ended, and a collapsible table lists every step;
- **tool calls**, **observations** (long ones collapsed) and the final **answer**.

## Example

A trace of HybridAgent-14B-Agentic on the first question of `examples/questions.jsonl`, as printed by the
terminal view (abbreviated: `...` marks omitted segments). Latent spans are the agent's own thoughts
rendered as pseudo-text; each is tens of forward passes without a single decoded token.

```text
QUESTION  At the speed of sound in air at 20 degrees Celsius, how many seconds does sound take
          to travel 10 kilometers? Round to the nearest second.
TRACE     7 latent spans · 138 latent steps · 782 text tokens · 1 search call · 2 python calls
          · 4 turns · end reason eos

TEXT      <think>
          Okay, so I need to figure out how long it takes for sound to travel 10 kilometers at
          the speed of sound in air at 20 degrees Celsius. Hmm, first off, I remember that the
          speed of sound varies depending on the medium and temperature. The question specifies
          20°C in air, so I should look up the exact speed under those conditions.
LATENT    [span 1 · 42 steps · readability 0.84]
          Wait, I me the the speed of speed in air at 20° is about 343 meters per second. But I
          should verify that. Let me do a search search to be sure.
TEXT      So I'll start by performing a search query to get the precise value.
          </think>
SEARCH    speed of sound in air at 20 degrees celsius meters per second
RESULT    Page 1: The speed of sound in dry air at 20 °C is 343 meters per second. Generally,
          sound waves travel most quickly through solids, followed by liquids, and then by
          Page 2: The speed of sound is the distance travelled per unit of time by a sound wave
          as it propagates through an elastic
          ...

TEXT      <think>
          Okay, so the user asked how long it takes for sound to travel 10 kilometers at the
          speed of sound in air at 20°C. From my initial search results, I found that the speed
          of sound in dry air at 20°C is indeed 343 meters per second. Some sources mentioned
          344 m/s, but 343 seems more consistent across multiple references.
LATENT    [span 2 · 22 steps · readability 0.88]
          Now, I need to calculate the time it takes for speed to travel 10 km at this speed.
TEXT      First, convert 10 kilometers to meters because the speed is given in m/s. 10
          kilometers is 10,000 meters. Then, use the formula time = distance / speed.
LATENT    [span 3 · 21 steps · readability 0.96]
          So, time = 10,000 meters / 343 meters per second.
TEXT      Let me perform this calculation using Python to ensure accuracy.
          </think>
PYTHON    # Calculate the time it takes for sound to travel 10 kilometers at 343 m/s
          distance_meters = 10 * 1000  # Convert 10 kilometers to meters
          speed_ms = 343  # Speed of sound in air at 20°C in meters per second
          time_seconds = distance_meters / speed_ms
          print(f"Time taken: {time_seconds} seconds")
          # Rounding to the nearest second
          rounded_time = round(time_seconds)
          print(f"Rounded to the nearest second: {rounded_time} seconds")
RESULT    Time taken: 29.154518950437318 seconds
          Rounded to the nearest second: 29 seconds

TEXT      <think>
          Okay, let me check the calculation again to make sure there isn't an error. The
          problem states the speed of sound in air at 20°C is approximately 343 meters per
          second, and the distance is 10 kilometers, which converts to 10,000 meters.
LATENT    [span 4 · 9 steps · readability 0.90]
          The the formula time = distance / speed:
TEXT      10,000 ÷ 343 ≈ 29.154...
LATENT    [span 5 · 13 steps · readability 0.89]
          R to the nearest second, that's 29 seconds.
          ...

LATENT    [span 7 · 10 steps · readability 0.92]
          So, the answer is 29 seconds.
TEXT      The speed of sound in air at 20°C is approximately **343 meters per second**.
          Converting 10 kilometers to meters (10,000 meters) and calculating the time using
          the formula \( \text{time} = \frac{\text{distance}}{\text{speed}} \):
          </think>
ANSWER    \boxed{29}  ->  29
```

In the HTML report the same spans are shown token by token, shaded by readability, with the
alternative nearest tokens of every step on hover.

## Python API

```python
import torch
from hybrid_agent import build_tool_executor, load_hybrid_agent
from interpretable_latent_mode import inspect, render_html, render_text

model, tokenizer = load_hybrid_agent("togethercomputer/HybridAgent-14B-Agentic", device="cuda")
traces = inspect(model, tokenizer, "How many trailing zeros does 100! have?",
                 device=torch.device("cuda"), tool_executor=build_tool_executor(), readout_top_k=5)
print(render_text(traces[0]))
open("report.html", "w").write(render_html(traces))
for segment in traces[0].latent_segments:
    print(segment.pseudo_text, [(s.token, round(s.cosine, 2)) for s in segment.steps[:5]])
```

`inspect` runs `LatentRecordingEngine`, the generation engine with a hook that keeps every
latent input, and returns one `HybridTrace` per sample: a list of `Segment`s (`text`,
`latent`, `tool_call`, `tool_result`, `answer`) where latent segments carry their
`LatentStep`s (nearest token, cosine, candidates). `trace_from_record` builds the same
structure from a batch-inference record (span pseudo-text only).
