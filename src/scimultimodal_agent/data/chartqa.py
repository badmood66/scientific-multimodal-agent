from __future__ import annotations

import hashlib
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from scimultimodal_agent.schemas import ChartSample

SPLIT_FILES = {
    "test": "data/test-00000-of-00001.parquet",
    "val": "data/val-00000-of-00001.parquet",
}


def download_chartqa(split: str, cache_dir: Path | None = None) -> Path:
    """Download one ChartQA split through the Hugging Face cache."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Supported splits: {sorted(SPLIT_FILES)}")
    return Path(
        hf_hub_download(
            repo_id="ahmed-masry/ChartQA",
            filename=SPLIT_FILES[split],
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def materialize_samples(
    parquet_path: Path,
    image_dir: Path,
    max_samples: int | None = None,
    seed: int = 42,
    balanced: bool = True,
) -> list[ChartSample]:
    """Convert embedded image bytes into stable local image paths."""
    frame = pd.read_parquet(parquet_path)
    if balanced and max_samples:
        groups = []
        per_group = max(1, max_samples // frame["type"].nunique())
        for _, group in frame.groupby("type", sort=True):
            groups.append(group.sample(min(per_group, len(group)), random_state=seed))
        frame = pd.concat(groups).sample(frac=1, random_state=seed).head(max_samples)
    elif max_samples:
        frame = frame.sample(min(max_samples, len(frame)), random_state=seed)

    image_dir.mkdir(parents=True, exist_ok=True)
    samples: list[ChartSample] = []
    for row_index, row in frame.iterrows():
        digest = hashlib.sha1(f"{row.imgname}|{row.query}".encode()).hexdigest()[:12]
        path = image_dir / f"{digest}_{row.imgname}"
        if not path.exists():
            image = Image.open(BytesIO(row.image)).convert("RGB")
            image.save(path, optimize=True)
        samples.append(
            ChartSample(
                sample_id=f"chartqa-{row_index}-{digest}",
                image_path=path,
                question=str(row.query),
                answer=str(row.label),
                source_type=str(row.type),
            )
        )
    return samples


def iter_samples(*args, **kwargs) -> Iterator[ChartSample]:
    yield from materialize_samples(*args, **kwargs)

