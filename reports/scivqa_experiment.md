# Scientific-figure experiment: SciVQA-100

Run date: 2026-10-01  
Model: `Qwen/Qwen3-VL-2B-Instruct` (frozen, deterministic decoding)  
Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM  
Dataset: 100 distinct figures from the SciVQA validation split  
Seed: 42

## Sampling and evaluation

The sample contains one question per figure and is balanced across three explicitly visual QA
types: 34 binary questions, 33 finite non-binary questions, and 33 open-answer questions. Images,
captions, and answer options are supplied to the model. Non-visual questions are excluded.

Exact/relaxed accuracy is reported alongside Token-F1 and ROUGE-L F1 because scientific answers
may use equivalent surface forms such as `Sue, Bill` and `Sue and Bill`. Generated traces preserve
all failures without retries.

## Main results

| Method | Accuracy | Token-F1 | ROUGE-L F1 | Mean latency |
|---|---:|---:|---:|---:|
| Direct VLM | 51.0% | 0.597 | 0.594 | 0.28 s |
| Chain-of-thought | 51.0% | 0.570 | 0.570 | 2.92 s |
| Full agent | **57.0%** | **0.643** | **0.640** | 4.81 s |

The full agent gains 6 percentage points in exact/relaxed accuracy over both baselines. Free-form
chain-of-thought does not improve accuracy and reduces overlap scores because complex multi-panel
questions often produce long, truncated explanations.

## Results by visual QA type

| QA type | Samples | Direct | Full agent | Change |
|---|---:|---:|---:|---:|
| Binary visual | 34 | 82.4% | 82.4% | 0.0 pp |
| Finite non-binary visual | 33 | 30.3% | 48.5% | **+18.2 pp** |
| Open-answer visual | 33 | 39.4% | 39.4% | 0.0 pp |

The gain is concentrated in finite-answer visual questions. Structured memory constrains the model
to valid option letters and corrects several cases where direct prompting repeats an option, emits
its description, or selects an inconsistent label.

## Component ablations

Ablations replay the same frozen visual-memory outputs. This controls model variation and isolates
downstream component effects.

| Variant | Accuracy | Token-F1 | ROUGE-L F1 | Interpretation |
|---|---:|---:|---:|---|
| Full agent | 57.0% | 0.643 | 0.640 | All components |
| Without structured memory | 51.0% | 0.597 | 0.594 | Reverts to direct candidate |
| Without arithmetic tools | 57.0% | 0.643 | 0.640 | No accuracy effect on this subset |
| Without verifier | 57.0% | 0.643 | 0.640 | Same answers; removes reliability signal |

The arithmetic tool is useful for ChartQA but contributes no answer changes on this SciVQA sample.
This is a negative result rather than evidence that the tool is universally unnecessary: the
selected scientific questions are dominated by spatial grounding, curve identification, and
multiple-choice mapping rather than explicit arithmetic.

## Consensus verification

Direct, chain-of-thought, and full-agent answers are treated as independent reasoning routes. The
answer is marked verified only when all three normalized predictions agree; the full-agent answer
itself is not changed.

- Coverage: 50 / 100 samples.
- Selective accuracy on verified samples: **72.0%**.
- Overall accuracy remains 57.0%.
- Exploratory ECE with heuristic agreement confidences: 0.034.

This verifier is more informative than the initial evidence-string heuristic, which was not
predictive under domain shift. Confidence values should be calibrated on a separate development
split before deployment; selective accuracy and coverage are the primary results here.

## Error analysis

The remaining failures cluster around:

1. multi-panel spatial grounding (e.g., top-right versus bottom-right plots);
2. visually similar line styles, markers, and legend colors;
3. combination answers such as `A,B,C`, where one option is omitted;
4. approximate ranges whose valid textual forms differ from the reference;
5. architecture diagrams requiring fine-grained edge or component localization.

These errors mainly reflect visual grounding limits of a frozen 2B model. Agent orchestration can
constrain outputs and expose uncertainty, but it cannot recover details the visual encoder failed
to perceive.

## Limitations

- The experiment uses 100 of 1,680 validation questions and one deterministic seed.
- Captions are supplied, so the setting measures image-plus-caption reasoning rather than image-only
  performance.
- Token-F1 and ROUGE-L are lightweight local metrics; the official shared task also uses semantic
  metrics and human assessment.
- Consensus verification requires three inference routes and raises mean latency to about 8 seconds.

