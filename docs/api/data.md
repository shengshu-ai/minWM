# Data

Dataset and dataloader factories.

::: minwm.data

## Dataset classes

Dataset classes are lazily re-exported from `minwm.data` via `__getattr__` (so
importing the package never forces optional deps like `lmdb` / `scipy` / `PIL`).
They are documented here from their defining submodules.

::: minwm.data.datasets.text

::: minwm.data.datasets.lmdb

::: minwm.data.datasets.image

::: minwm.data.datasets.camera_pt

::: minwm.data.datasets.mock
