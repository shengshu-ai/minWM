"""Shared tiny ``ARHunyuanVideo_1_5`` arch for the HY smoke-test (mock) configs.

The single-model mock stages (``stage1`` / ``stage2a`` / ``stage2b``) of both
the action2v and ti2v lines inherit this. ProPE is on by default
(``use_prope=True``); the ti2v stages override it to ``False``. The DMD
``stage3_mock`` keeps its own local arch because it instantiates three copies
(generator + real/fake critics).
"""

model = dict(
    type="ARHunyuanVideo_1_5_DiffusionTransformer",
    patch_size=[1, 2, 2],
    in_channels=16,
    concat_condition=False,
    hidden_size=64,
    heads_num=4,
    mm_double_blocks_depth=2,
    mm_single_blocks_depth=0,
    rope_dim_list=[4, 6, 6],
    text_states_dim=64,
    text_states_dim_2=64,
    use_prope=True,
)
