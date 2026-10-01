from __future__ import annotations

import argparse
import json
from pathlib import Path

from scimultimodal_agent.agent.pipeline import ScientificChartAgent, expected_answer_type
from scimultimodal_agent.agent.verifier import EvidenceVerifier
from scimultimodal_agent.evaluation.metrics import summarize
from scimultimodal_agent.schemas import AgentTrace


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def option_ids(metadata_path: Path | None) -> set[str]:
    if metadata_path is None:
        return set()
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        f"scivqa-{record['instance_id']}"
        for record in records
        if record.get("answer_options")
    }


def to_trace(record: dict, mode: str, direct_latency: float, has_options: bool) -> AgentTrace:
    memory = dict(record.get("memory", {}))
    direct_candidate = str(memory.get("direct_candidate", record.get("prediction", "")))
    tools = list(record.get("tool_calls", []))
    verifier = EvidenceVerifier()

    if mode == "no_memory":
        memory = {"direct_candidate": direct_candidate}
        tools = []
        prediction = direct_candidate
        verified, confidence = False, 0.5
        latency = direct_latency
    elif mode == "no_tools":
        tools = []
        prediction = ScientificChartAgent._resolve_answer(
            direct_candidate,
            memory,
            tools,
            expected_answer_type(record["question"]),
            record["question"],
            has_options,
        )
        result = verifier.verify(prediction, memory, tools)
        verified, confidence = result.accepted, result.confidence
        latency = float(record.get("latency_seconds", 0.0))
    elif mode == "no_verifier":
        prediction = record["prediction"]
        verified, confidence = False, 0.5
        latency = float(record.get("latency_seconds", 0.0))
    else:
        raise ValueError(mode)

    return AgentTrace(
        sample_id=record["sample_id"],
        question=record["question"],
        gold_answer=record["gold_answer"],
        prediction=prediction,
        mode=mode,
        memory=memory,
        tool_calls=tools,
        verified=verified,
        confidence=confidence,
        latency_seconds=latency,
        error=record.get("error"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay deterministic agent ablations.")
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    full_records = load_jsonl(args.full)
    direct_records = {record["sample_id"]: record for record in load_jsonl(args.direct)}
    choices = option_ids(args.metadata)
    summary = {}
    for mode in ("no_memory", "no_tools", "no_verifier"):
        traces = [
            to_trace(
                record,
                mode,
                float(direct_records[record["sample_id"]].get("latency_seconds", 0.0)),
                record["sample_id"] in choices,
            )
            for record in full_records
        ]
        output = args.output_prefix.with_name(f"{args.output_prefix.name}_{mode}.jsonl")
        output.write_text(
            "\n".join(json.dumps(trace.to_dict(), ensure_ascii=False) for trace in traces) + "\n",
            encoding="utf-8",
        )
        metrics = summarize(traces)
        output.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        summary[mode] = metrics
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

