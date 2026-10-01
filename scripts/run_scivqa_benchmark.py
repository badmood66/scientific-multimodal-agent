from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scimultimodal_agent.agent.pipeline import ScientificChartAgent
from scimultimodal_agent.data.scivqa import download_scivqa, materialize_scivqa_samples
from scimultimodal_agent.evaluation.runner import run_evaluation
from scimultimodal_agent.models.qwen_vl import QwenVLBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched SciVQA experiments.")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["direct", "cot", "full"])
    parser.add_argument("--cache-dir", type=Path, default=Path("D:/ResearchData/hf_cache"))
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(args.cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(args.cache_dir / "hub"))
    metadata, images = download_scivqa("validation", args.cache_dir / "hub")
    samples = materialize_scivqa_samples(
        metadata,
        images,
        Path("data/processed/scivqa_validation"),
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
            output_path=Path("reports") / f"scivqa_{mode}_n{len(samples)}.jsonl",
        )
    output = Path("reports") / f"scivqa_benchmark_n{len(samples)}.json"
    output.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()

