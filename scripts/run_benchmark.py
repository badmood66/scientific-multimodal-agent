from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scimultimodal_agent.agent.pipeline import ScientificChartAgent
from scimultimodal_agent.data.chartqa import download_chartqa, materialize_samples
from scimultimodal_agent.evaluation.runner import run_evaluation
from scimultimodal_agent.models.qwen_vl import QwenVLBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched ChartQA experiments.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["direct", "cot", "full"])
    parser.add_argument("--cache-dir", type=Path, default=Path("D:/ResearchData/hf_cache"))
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(args.cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(args.cache_dir / "hub"))
    parquet = download_chartqa("test", args.cache_dir / "hub")
    samples = materialize_samples(
        parquet,
        Path("data/processed/chartqa_images"),
        max_samples=args.samples,
        seed=args.seed,
    )
    agent = ScientificChartAgent(QwenVLBackend())
    all_metrics = {}
    for mode in args.modes:
        all_metrics[mode] = run_evaluation(
            agent,
            samples,
            mode=mode,
            output_path=Path("reports") / f"chartqa_{mode}_n{len(samples)}.jsonl",
        )
    output = Path("reports") / f"benchmark_n{len(samples)}.json"
    output.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()

