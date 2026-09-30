from __future__ import annotations

import json
from pathlib import Path

from scimultimodal_agent.agent.pipeline import ScientificChartAgent
from scimultimodal_agent.evaluation.metrics import summarize
from scimultimodal_agent.schemas import ChartSample


def run_evaluation(
    agent: ScientificChartAgent,
    samples: list[ChartSample],
    mode: str,
    output_path: Path,
    tolerance: float = 0.05,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            trace = agent.run(sample, mode)
            traces.append(trace)
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            print(
                f"[{index}/{len(samples)}] {sample.sample_id}: "
                f"pred={trace.prediction!r} gold={trace.gold_answer!r} "
                f"latency={trace.latency_seconds:.1f}s"
            )
    metrics = summarize(traces, tolerance)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

