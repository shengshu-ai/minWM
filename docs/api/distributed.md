# Distributed

Collectives, sequence parallelism, and FSDP helpers.

::: minwm.distributed

## FSDP2 sharding

The `minwm.distributed.fsdp` helpers turn a data-parallel `DeviceMesh` (from
[`get_fsdp_mesh`][minwm.distributed.get_fsdp_mesh]) plus each backbone's
`_fsdp_shard_conditions` hook into a sharded model, so the engine stays
backbone-agnostic. They are imported from the submodule (not re-exported at
`minwm.distributed`) and documented here from their defining module.

::: minwm.distributed.fsdp
