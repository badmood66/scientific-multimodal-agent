from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scimultimodal_agent.agent.verifier import normalize_answer
from scimultimodal_agent.evaluation.metrics import summarize
from scimultimodal_agent.schemas import AgentTrace


def load_jsonl(path: Path) -> dict[str, dict]:
    return {
        record["sample_id"]: record
        for record in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify full-agent answers by route agreement.")
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--cot", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    direct, cot, full = load_jsonl(args.direct), load_jsonl(args.cot), load_jsonl(args.full)
    traces = []
    for sample_id, record in full.items():
        candidates = [
            normalize_answer(direct[sample_id]["prediction"]),
            normalize_answer(cot[sample_id]["prediction"]),
            normalize_answer(record["prediction"]),
        ]
        largest_group = Counter(candidates).most_common(1)[0][1]
        if largest_group == 3:
            verified, confidence = True, 0.75
        elif largest_group == 2:
            verified, confidence = False, 0.50
        else:
            verified, confidence = False, 0.20
        traces.append(
            AgentTrace(
                sample_id=sample_id,
                question=record["question"],
                gold_answer=record["gold_answer"],
                prediction=record["prediction"],
                mode="consensus_verified",
                memory={
                    "direct": direct[sample_id]["prediction"],
                    "cot": cot[sample_id]["prediction"],
                    "full": record["prediction"],
                    "agreement_count": largest_group,
                },
                verified=verified,
                confidence=confidence,
                latency_seconds=(
                    float(direct[sample_id].get("latency_seconds", 0.0))
                    + float(cot[sample_id].get("latency_seconds", 0.0))
                    + float(record.get("latency_seconds", 0.0))
                ),
            )
        )

    args.output.write_text(
        "\n".join(json.dumps(trace.to_dict(), ensure_ascii=False) for trace in traces) + "\n",
        encoding="utf-8",
    )
    metrics = summarize(traces)
    verified = [trace for trace in traces if trace.verified]
    from scimultimodal_agent.agent.verifier import answers_match

    metrics["selective_accuracy"] = (
        sum(answers_match(trace.prediction, trace.gold_answer) for trace in verified) / len(verified)
        if verified
        else 0.0
    )
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

