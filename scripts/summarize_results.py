from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scimultimodal_agent.agent.verifier import answers_match


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize agent traces by reasoning type.")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    traces = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines()]
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for trace in traces:
        kind = (trace.get("plan") or {}).get("reasoning_type", trace.get("mode", "unknown"))
        groups[kind][0] += 1
        groups[kind][1] += int(answers_match(trace["prediction"], trace["gold_answer"]))
    print("reasoning_type,samples,correct,accuracy")
    for kind, (count, correct) in sorted(groups.items()):
        print(f"{kind},{count},{correct},{correct / count:.4f}")


if __name__ == "__main__":
    main()

