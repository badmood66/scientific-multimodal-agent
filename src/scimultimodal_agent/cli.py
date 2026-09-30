from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scimultimodal_agent.agent.pipeline import ScientificChartAgent
from scimultimodal_agent.data.chartqa import download_chartqa, materialize_samples
from scimultimodal_agent.evaluation.runner import run_evaluation
from scimultimodal_agent.models.qwen_vl import QwenVLBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a multimodal chart reasoning agent.")
    parser.add_argument("--mode", default="full", choices=sorted(ScientificChartAgent.MODES))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("D:/ResearchData/hf_cache"))
    parser.add_argument("--output", type=Path, default=Path("reports/full.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("HF_HOME", str(args.cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(args.cache_dir / "hub"))
    parquet = download_chartqa("test", args.cache_dir / "hub")
    samples = materialize_samples(
        parquet,
        Path("data/processed/chartqa_images"),
        max_samples=args.samples,
        seed=args.seed,
    )
    backend = QwenVLBackend(model_id=args.model, device=args.device)
    metrics = run_evaluation(
        ScientificChartAgent(backend),
        samples,
        mode=args.mode,
        output_path=args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

