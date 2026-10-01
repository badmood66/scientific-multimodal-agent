import json
import zipfile

from scimultimodal_agent.data.scivqa import materialize_scivqa_samples


def test_materializes_scivqa_sample(tmp_path) -> None:
    metadata = tmp_path / "validation.json"
    archive = tmp_path / "images.zip"
    image_name = "paper-Figure1.png"
    metadata.write_text(
        json.dumps(
            [
                {
                    "instance_id": "abc",
                    "image_file": image_name,
                    "figure_id": "figure-1",
                    "caption": "A scientific result.",
                    "qa_pair_type": "closed-ended visual",
                    "question": "Which curve is higher?",
                    "answer": "blue",
                    "answer_options": ["{'A': 'blue'}", "{'B': 'red'}"],
                    "figure_type": "line chart",
                }
            ]
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(f"images_validation/{image_name}", b"fake-image")

    samples = materialize_scivqa_samples(metadata, archive, tmp_path / "output")
    assert len(samples) == 1
    assert samples[0].caption == "A scientific result."
    assert samples[0].answer_options == ["A: blue", "B: red"]
    assert samples[0].image_path.read_bytes() == b"fake-image"
