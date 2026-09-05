"""Tests for the id-centric inference input schema and output naming.

Covers the GPU-independent seams of the inference entry point:

- ``minwm.engine.inferencer.read_benchmark`` — the shared t2v/i2v JSON schema
  (``id`` required, ``image`` optional, paths resolved relative to the JSON),
  plus the ``limit`` slice and CWD-independent path resolution.
- ``BaseInferencer.build_benchmark`` — the engine owns the data source.
- ``BaseInferencer.run`` output naming — ``{id}.mp4`` for JSON inputs, and the
  fact that ``id`` is naming metadata kept off the generation batch.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minwm.engine.inferencer import BaseInferencer, ResultWriter, read_benchmark

# Also importable from the CLI entry point (re-export for back-compat).
from tools.infer_mwm import read_benchmark as read_benchmark_cli


def _write_json(tmp_path, items) -> str:
    path = os.path.join(tmp_path, "input.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f)
    return path


def test_read_benchmark_requires_id(tmp_path):
    path = _write_json(tmp_path, [{"caption": "a cat", "image": "img/1.png"}])
    with pytest.raises(ValueError, match="missing required 'id'"):
        read_benchmark(path)


def test_read_benchmark_i2v_resolves_image_relative(tmp_path):
    path = _write_json(
        tmp_path,
        [{"id": 1, "caption": "a cat", "image": "img/1.png", "trajectory": "w*4"}],
    )
    (sample,) = read_benchmark(path)
    assert sample == {
        "id": 1,
        "prompt": "a cat",
        "trajectory": "w*4",
        "image": os.path.join(tmp_path, "img/1.png"),
    }


def test_read_benchmark_t2v_omits_image(tmp_path):
    path = _write_json(tmp_path, [{"id": "clip_01", "caption": "a dog"}])
    (sample,) = read_benchmark(path)
    assert "image" not in sample
    assert sample == {"id": "clip_01", "prompt": "a dog", "trajectory": None}


def test_read_benchmark_keeps_absolute_image(tmp_path):
    abs_img = os.path.join(tmp_path, "elsewhere", "x.png")
    path = _write_json(tmp_path, [{"id": 2, "caption": "a bird", "image": abs_img}])
    (sample,) = read_benchmark(path)
    assert sample["image"] == abs_img


def test_read_benchmark_limit_slices(tmp_path):
    path = _write_json(tmp_path, [{"id": i, "caption": f"c{i}"} for i in range(5)])
    assert len(read_benchmark(path, limit=2)) == 2
    assert len(read_benchmark(path)) == 5


def test_read_benchmark_cli_reexport_is_engine_fn():
    assert read_benchmark_cli is read_benchmark


@pytest.fixture
def stub_mp4_encode(monkeypatch):
    """Touch the mp4 instead of encoding it: the encode needs torchvision, which CI omits."""

    def _touch(self, video, path):
        open(path, "wb").close()

    monkeypatch.setattr(ResultWriter, "_save_video", _touch)


class _RunInferencer(BaseInferencer):
    """Bypass heavy ``__init__``; stub the loop so ``run`` naming is exercised.

    Only the seams ``run`` touches are populated. ``build_batch`` records the
    batch the loop would see so tests can assert ``id`` is stripped.
    """

    def __init__(self, out_dir: str):
        self.rank = 0
        self.seed = 0
        self.world_size = 1
        self.sp_size = 1
        self.dp_size = 1
        self.sp_group_index = 0
        self.is_writer = True
        self.inference_cfg = {"output_dir": out_dir}
        self.seen_batches = []

    def build_batch(self, prompt, trajectory=None):
        self.seen_batches.append(prompt)
        return {"prompt": prompt, "trajectory": trajectory}

    @property
    def loop(self):
        # A single-frame video plus latents, as a real loop returns.
        result = {"latents": torch.zeros(1, 1, 4, 2, 2), "video": torch.zeros(1, 1, 3, 2, 2)}
        return type("_L", (), {"generate": staticmethod(lambda batch: result)})()


def test_run_names_json_input_by_id(tmp_path, stub_mp4_encode):
    inf = _RunInferencer(str(tmp_path))
    inf.run([{"id": "clip_07", "prompt": "a cat"}])
    assert os.path.exists(os.path.join(str(tmp_path), "clip_07.mp4"))


def test_run_keeps_latents_next_to_video(tmp_path, stub_mp4_encode):
    inf = _RunInferencer(str(tmp_path))
    inf.run([{"id": "clip_07", "prompt": "a cat"}])
    assert os.path.exists(os.path.join(str(tmp_path), "clip_07.latents.pt"))


def test_run_strips_id_from_generation_batch(tmp_path, stub_mp4_encode):
    inf = _RunInferencer(str(tmp_path))
    inf.run([{"id": 3, "prompt": "a cat", "image": "x.png"}])
    assert inf.seen_batches == [{"prompt": "a cat", "image": "x.png"}]


def test_run_records_id_in_manifest(tmp_path, stub_mp4_encode):
    inf = _RunInferencer(str(tmp_path))
    inf.run([{"id": "clip_07", "prompt": "a cat"}, {"id": "clip_08", "prompt": "a dog"}])
    with open(os.path.join(str(tmp_path), "manifest.json"), encoding="utf-8") as f:
        items = json.load(f)["items"]
    assert [it["id"] for it in items] == ["clip_07", "clip_08"]


def test_result_writer_latents_only_when_no_video(tmp_path):
    writer = ResultWriter(str(tmp_path), fps=16)
    writer.ensure_dir()
    path = writer.write("clip_09", {"latents": torch.zeros(1, 1, 4, 2, 2)})
    assert path.endswith("clip_09.latents.pt")
    assert os.path.exists(path)
