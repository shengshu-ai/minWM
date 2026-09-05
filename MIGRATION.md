# Migration Guide: v0.1 (legacy) → v0.2 (minwm)

Starting with v0.2, minWM is reorganized from three parallel model trees
(`HY15/`, `Wan21/`, `shared/`) into a single installable `minwm/` package driven
by config files. **Every path the old docs referenced has moved.** If you built
on the v0.1 layout, use the tables below to update.

The v0.1 layout is preserved on the pre-rewrite `main` (tagged `v0.1-legacy` at
release); check it out to keep running the old code unchanged:

```bash
git checkout v0.1-legacy
```

## What changed

- `HY15/`, `Wan21/`, and `shared/` are replaced by the `minwm/` package
  (`minwm/modeling/{hy15,wan21}`, plus `engine`, `config`, `distributed`,
  `sampling`, `processors`, `data`).
- Training and inference are no longer per-model shell scripts. There are two
  entrypoints — `tools/train_mwm.py` and `tools/infer_mwm.py` — both selected by
  a `--config-file` and overridden with `key=value` dotlist args.
- Install is now editable: `pip install -r requirements/base.txt` then
  `pip install -e .`, so `import minwm` resolves without setting `PYTHONPATH`.

## Entrypoint mapping

| v0.1 (legacy) | v0.2 (minwm) |
|---|---|
| `Wan21/wan_train.py`, `Wan21/model/*.py` | `tools/train_mwm.py --config-file configs/wan21/action2v/train/stage*.py` |
| `HY15/scripts/training/**/*.sh` | `tools/train_mwm.py --config-file configs/hy/action2v/train/stage*.py` |
| `HY15/hy15_inference.py`, `HY15/scripts/inference/*.sh` | `tools/infer_mwm.py --config-file configs/hy/action2v/infer/stage*.py` |
| Wan inference scripts | `tools/infer_mwm.py --config-file configs/wan21/action2v/infer/stage*.py` |
| `Wan21/get_causal_ode_data_prope.py` | `tools/data/wan21/ode/get_causal_ode_data_prope.py` |
| `HY15/scripts/data_preprocessing/*.sh` | `tools/data/hy/` + `configs/hy/action2v/README.md` |

## Directory mapping

| v0.1 (legacy) | v0.2 (minwm) |
|---|---|
| `HY15/hyvideo/` (model code) | `minwm/modeling/hy15/` |
| `Wan21/model/` (model code) | `minwm/modeling/wan21/` |
| `shared/algorithms/` (DMD, CD, ODE, flow matching, …) | `minwm/` (engine recipes + `minwm/sampling/`) |
| `training_wan.md` | `configs/wan21/action2v/README.md` |
| `training_hunyuan.md` | `configs/hy/action2v/README.md` |
| `requirements.txt` | `requirements/base.txt` (+ `pip install -e .`) |

## Stage pipeline

The five-stage Action2V pipeline (Bi-SFT → AR-TF → AR-ODE → AR-CD → AR-DMD) is
unchanged in concept; each stage is now a config under
`configs/<backbone>/action2v/{train,infer}/`. See the per-backbone READMEs and
the [docs site](https://github.com/shengshu-ai/minWM) for the current commands.
