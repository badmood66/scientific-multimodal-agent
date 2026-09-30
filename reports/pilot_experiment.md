# Pilot experiment: ChartQA-50

Run date: 2026-10-01  
Model: `Qwen/Qwen3-VL-2B-Instruct` (frozen, deterministic decoding)  
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM  
Dataset: 50 seeded, balanced samples from the ChartQA test split  
Seed: 42

## Research questions

1. Does evidence-grounded agent orchestration improve chart-question accuracy over direct prompting?
2. Does explicit memory, tool execution, and verification improve confidence calibration?
3. Which question types remain difficult for a compact local vision-language model?

## Results

| Method | Accuracy | ECE (lower is better) | Mean latency | Verified rate |
|---|---:|---:|---:|---:|
| Direct VLM | 74.0% | 0.240 | 0.36 s | — |
| Chain-of-thought | 76.0% | 0.260 | 2.97 s | — |
| Full agent | **76.0%** | **0.087** | 4.67 s | 86.0% |

The full agent improves accuracy by 2 percentage points over direct prompting and matches the
chain-of-thought baseline. Its clearer benefit is reliability: expected calibration error falls by
63.8% relative to the direct baseline, while 43 of 50 responses are linked to extracted evidence or
a successful tool result. The 50-sample pilot is a development result, not a claim about the full
ChartQA test set.

## Full-agent results by planned reasoning type

| Reasoning type | Correct / total | Accuracy |
|---|---:|---:|
| Count | 11 / 11 | 100.0% |
| Comparison | 5 / 5 | 100.0% |
| Lookup | 14 / 16 | 87.5% |
| Trend | 1 / 2 | 50.0% |
| Arithmetic | 7 / 16 | 43.8% |

The gap is concentrated in arithmetic questions. Manual trace review identified four recurring
failure modes: dense multi-series legend grounding, ratio operand ordering, reading small or
negative changes, and extracting the correct subset for averages. These failures motivate chart
region grounding and uncertainty-aware tool invocation rather than more permissive tool use.

## Reliability design

- The planner uses deterministic question routing so ablations are reproducible.
- Structured memory is deliberately question-focused and stores no more than two relevant values.
- The calculator evaluates an allow-listed arithmetic abstract syntax tree; code execution, names,
  attributes, calls, and indexing are rejected.
- Tool outputs only overwrite a numeric candidate when they agree within 10%; otherwise they are
  retained as disagreement evidence. This conservative rule was introduced after an early pilot
  showed that unconditional tool use can amplify perception errors.
- Each JSONL trace stores the plan, memory, evidence, tool calls, confidence, latency, and errors.

## Limitations

- Fifty samples are sufficient for pipeline debugging but not for a statistically stable benchmark.
- The verifier checks evidence consistency, not factual truth; correlated perception errors can pass.
- ChartQA is dominated by business and public-statistics charts. Broader scientific figures require
  domain-specific evaluation, including plots, diagrams, tables, and multi-panel figures.
- No model fine-tuning was performed, so results isolate orchestration rather than training gains.

