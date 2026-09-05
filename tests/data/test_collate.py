"""Tests for minwm.data.collate.HYCollator.

Verifies the collator bridges the HY camera dataset output to the recipe
batch contract: latent rename + channel/frame transpose, tensor stacking,
non-tensor passthrough, and survival of HY-only signals.
"""

import torch

from minwm.data.collate import HYCollator

_C, _F, _H, _W = 4, 8, 5, 6
_N = 5  # ode trajectory steps


def _plucker_sample(idx: int) -> dict:
    return {
        "latent": torch.randn(_C, _F, _H, _W),
        "viewmats": torch.eye(4).unsqueeze(0).repeat(_F, 1, 1),
        "Ks": torch.eye(3).unsqueeze(0).repeat(_F, 1, 1),
        "action": torch.zeros(_F, dtype=torch.int64),
        "byt5_text_states": torch.randn(2, 8),
        "video_path": f"/data/sample_{idx}.pt",
        "select_window_out_flag": 0,
    }


def _ode_sample(idx: int) -> dict:
    s = _plucker_sample(idx)
    del s["latent"]
    s["ode_trajectory"] = torch.randn(_N, _C, _F, _H, _W)
    return s


def test_plucker_collate_renames_and_transposes():
    batch = HYCollator()([_plucker_sample(0), _plucker_sample(1)])

    assert "latent" not in batch
    assert batch["clean_latent"].shape == (2, _F, _C, _H, _W)


def test_ode_collate_renames_and_transposes():
    batch = HYCollator()([_ode_sample(0), _ode_sample(1)])

    assert "ode_trajectory" not in batch
    assert batch["ode_latent"].shape == (2, _N, _F, _C, _H, _W)


def test_transpose_preserves_values():
    # clean_latent[b] should be sample latent with C/F axes swapped.
    sample = _plucker_sample(0)
    batch = HYCollator()([sample])
    assert torch.equal(batch["clean_latent"][0], sample["latent"].transpose(0, 1))


def test_camera_and_tensor_fields_stack():
    batch = HYCollator()([_plucker_sample(0), _plucker_sample(1)])
    assert batch["viewmats"].shape == (2, _F, 4, 4)
    assert batch["Ks"].shape == (2, _F, 3, 3)


def test_hy_only_signals_survive():
    batch = HYCollator()([_plucker_sample(0), _plucker_sample(1)])
    assert batch["action"].shape == (2, _F)
    assert batch["byt5_text_states"].shape == (2, 2, 8)


def test_non_tensor_fields_passthrough():
    batch = HYCollator()([_plucker_sample(0), _plucker_sample(1)])
    assert batch["video_path"] == ["/data/sample_0.pt", "/data/sample_1.pt"]
    assert batch["select_window_out_flag"] == [0, 0]


def test_rename_disabled_keeps_raw_keys():
    batch = HYCollator(rename_latents=False)([_plucker_sample(0)])
    assert "clean_latent" not in batch
    assert batch["latent"].shape == (1, _C, _F, _H, _W)
