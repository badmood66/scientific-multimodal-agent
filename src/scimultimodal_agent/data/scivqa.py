from __future__ import annotations

import ast
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download

from scimultimodal_agent.schemas import ChartSample

FILES = {
    "validation": (
        "data/validation_2025-07-03_09-06.json",
        "data/images_validation.zip",
    ),
    "train": ("data/train_2025-07-03_09-06.json", "data/images_train.zip"),
}


def _normalize_options(raw_options: list) -> list[str]:
    normalized = []
    for option in raw_options:
        if isinstance(option, dict):
            items = option.items()
        else:
            try:
                parsed = ast.literal_eval(str(option))
            except (SyntaxError, ValueError):
                parsed = None
            items = parsed.items() if isinstance(parsed, dict) else []
        if items:
            normalized.extend(f"{key}: {value}" for key, value in items)
        else:
            normalized.append(str(option))
    return normalized


def download_scivqa(split: str, cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Download SciVQA metadata and its corresponding image archive."""
    if split not in FILES:
        raise ValueError(f"Supported splits: {sorted(FILES)}")
    metadata_file, image_file = FILES[split]
    common = {
        "repo_id": "katebor/SciVQA",
        "repo_type": "dataset",
        "cache_dir": str(cache_dir) if cache_dir else None,
    }
    return (
        Path(hf_hub_download(filename=metadata_file, **common)),
        Path(hf_hub_download(filename=image_file, **common)),
    )


def _balanced_select(records: list[dict], limit: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get("qa_pair_type", "unknown"))].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
    rng.shuffle(selected)
    return selected


def materialize_scivqa_samples(
    metadata_path: Path,
    image_archive: Path,
    image_dir: Path,
    max_samples: int = 100,
    seed: int = 42,
    visual_only: bool = True,
    unique_figures: bool = True,
) -> list[ChartSample]:
    """Build a diverse local SciVQA sample without extracting the whole archive."""
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if visual_only:
        records = [
            record
            for record in records
            if "visual" in str(record.get("qa_pair_type", "")).lower()
            and "non-visual" not in str(record.get("qa_pair_type", "")).lower()
        ]
    if unique_figures:
        rng = random.Random(seed)
        rng.shuffle(records)
        deduplicated: dict[str, dict] = {}
        for record in records:
            deduplicated.setdefault(str(record["figure_id"]), record)
        records = list(deduplicated.values())
    selected = _balanced_select(records, min(max_samples, len(records)), seed)

    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(image_archive) as archive:
        members = {
            PurePosixPath(name).name: name
            for name in archive.namelist()
            if name.lower().endswith((".png", ".jpg", ".jpeg")) and "__MACOSX" not in name
        }
        samples = []
        for record in selected:
            image_name = str(record["image_file"])
            if image_name not in members:
                raise FileNotFoundError(f"{image_name} is absent from {image_archive}")
            destination = image_dir / image_name
            if not destination.exists():
                destination.write_bytes(archive.read(members[image_name]))
            samples.append(
                ChartSample(
                    sample_id=f"scivqa-{record['instance_id']}",
                    image_path=destination,
                    question=str(record["question"]),
                    answer=str(record["answer"]),
                    source_type=str(record.get("qa_pair_type", "unknown")),
                    caption=str(record.get("caption", "")),
                    answer_options=_normalize_options(record.get("answer_options", [])),
                    metadata={
                        "figure_type": record.get("figure_type"),
                        "compound": record.get("compound"),
                        "venue": record.get("venue"),
                        "paper_id": record.get("paper_id"),
                        "source_dataset": record.get("source_dataset"),
                    },
                )
            )
    return samples
