"""Wan Action2V dataset sanity-check: decode LMDB latents + overlay camera BEV.

Decode the training latents back to RGB and render a side-by-side playback video
``[ RGB frame | animated top-down BEV ]`` so you can eyeball whether the stored
camera trajectory actually matches the video content.

It reads the *exact* data the trainer sees by instantiating the real
:class:`~minwm.data.datasets.lmdb.CameraLatentLMDBDataset`, so pose
normalization (align-to-first-frame) is identical to training.

Pipeline per sampled row:
    1. decode ``clean_latent`` ``(F,C,H,W)`` through the Wan VAE -> RGB frames;
    2. take ``viewmats`` ``(F,4,4)`` w2c OpenCV built by the dataset, invert to
       ``c2w``;
    3. BEV panel: top-down world X-Z trajectory + yaw arrow + pitch gauge.

Camera convention: OpenCV. ``viewmats`` are world->cam; ``c2w = inv(viewmats)``.
Camera center is ``c2w[:3,3]``; camera forward (look) is ``c2w[:3,2]``. The
top-down BEV uses world X (horizontal) and Z (vertical); Y (up/down) drives the
pitch gauge. The Wan VAE is 4x temporally compressed with a special first frame,
so ``F`` latent frames decode to ``1 + (F-1)*4`` pixel frames.

Usage::

    python tools/data/wan21/check_dataset_bev.py \\
        --data_path ./dataset/Wan21/Action2V/data --num_videos 8 --gpu 0
    # explicit rows + fixed BEV extent for cross-clip comparability:
    python tools/data/wan21/check_dataset_bev.py \\
        --data_path <lmdb_dir> --rows 0,1,2 --range 5.0 --out_dir /tmp/ds_check
    # scale-only mode (no VAE / GPU): scan every row's camera motion + figure:
    python tools/data/wan21/check_dataset_bev.py \\
        --data_path <lmdb_dir> --scale_plot
"""

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from minwm.data.datasets.geometry import build_viewmats_and_Ks  # noqa: E402
from minwm.data.datasets.lmdb import (  # noqa: E402
    CameraLatentLMDBDataset,
    _get_row,
    _get_shape,
    _is_single_lmdb,
    _open,
    _open_shards,
)
from minwm.modeling.wan21.vae import Wan21VAE  # noqa: E402

_DEFAULT_VAE = "./ckpts/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
_REPO_ROOT = Path(__file__).resolve().parents[3]


@torch.no_grad()
def decode_latent_to_frames(vae: Wan21VAE, clean_latent: torch.Tensor) -> np.ndarray:
    """Decode one clean latent ``(F,C,H,W)`` to uint8 RGB frames ``(F',H,W,3)``.

    The Wan VAE is 4x temporally compressed with a special first frame, so ``F``
    latent frames decode to ``F' = 1 + (F-1)*4`` pixel frames.

    Args:
        vae (Wan21VAE): the Wan VAE wrapper (already on the target device/dtype).
        clean_latent (torch.Tensor): latent video ``(F,C,H,W)``.

    Returns:
        np.ndarray: uint8 RGB frames ``(F',H,W,3)``.
    """
    # The VAE decode path runs without autocast, so match its module dtype.
    latent_cfhw = clean_latent.permute(1, 0, 2, 3).to(device=vae.device, dtype=vae.dtype)
    pixels = vae.decode([latent_cfhw])[0]  # (C,F',H,W) in [-1,1] fp32
    pixels = pixels.permute(1, 2, 3, 0)  # (F',H,W,C)
    return ((pixels + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()


# ─────────────────────────── BEV rendering ───────────────────────────────────
def _nice_length(x: float) -> float:
    if x <= 0:
        return 1.0
    k = math.floor(math.log10(x))
    base = 10.0**k
    for m in (5.0, 2.0, 1.0):
        if m * base <= x:
            return m * base
    return base


def _fmt_len(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}"


def poses_to_bev_tracks(viewmats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert w2c view matrices to top-down BEV tracks.

    Args:
        viewmats (np.ndarray): ``(T,4,4)`` w2c OpenCV matrices.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``centers_xz`` ``(T,2)``
        camera centers on the world X-Z plane, ``head_xz`` ``(T,2)`` unit
        heading vectors, and ``pitch_deg`` ``(T,)`` look-up/down angle
        (``+`` = looking up).
    """
    c2w = np.linalg.inv(viewmats)
    centers = c2w[:, :3, 3]
    fwd = c2w[:, :3, 2]

    centers_xz = centers[:, [0, 2]]
    head_xz = fwd[:, [0, 2]]
    head_xz = head_xz / np.clip(np.linalg.norm(head_xz, axis=1, keepdims=True), 1e-9, None)

    fwd_n = fwd / np.clip(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9, None)
    # OpenCV world is Y-down: fy<0 means looking up -> negate so +deg = look-up.
    pitch_deg = -np.degrees(np.arcsin(np.clip(fwd_n[:, 1], -1.0, 1.0)))
    return centers_xz, head_xz, pitch_deg


def bev_frame(
    centers_xz, headings_xz, pitch_deg, i, lims, size_px, dpi=100, pitch_range=30.0, prompt=""
):
    """Render one BEV panel (top-down X-Z + pitch gauge) up to frame ``i``."""
    side = size_px / dpi
    fig = plt.figure(figsize=(side, side), dpi=dpi)
    y0 = 0.16 if prompt else 0.02
    h = 0.96 - y0 + 0.02
    ax = fig.add_axes([0.02, y0, 0.78, h])
    gax = fig.add_axes([0.83, y0 + 0.06 * h, 0.14, 0.84 * h])

    xmin, xmax, zmin, zmax = lims
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
    ax.grid(True, color=(0.85, 0.85, 0.85), linewidth=0.6)
    ax.tick_params(labelsize=6, length=2, colors=(0.35, 0.35, 0.35))
    for s in ax.spines.values():
        s.set_color((0.6, 0.6, 0.6))

    ax.plot(
        centers_xz[:, 0], centers_xz[:, 1], "-", color=(0.8, 0.85, 0.95), linewidth=1.2, zorder=1
    )
    ax.plot(
        centers_xz[: i + 1, 0],
        centers_xz[: i + 1, 1],
        "-",
        color=(0.15, 0.35, 0.85),
        linewidth=2.2,
        zorder=2,
    )

    ax.plot(centers_xz[0, 0], centers_xz[0, 1], "o", color=(0.2, 0.7, 0.2), markersize=7, zorder=3)

    cx, cz = centers_xz[i]
    hx, hz = headings_xz[i]
    span = max(xmax - xmin, zmax - zmin)
    alen = 0.10 * span
    ax.arrow(
        cx,
        cz,
        hx * alen,
        hz * alen,
        head_width=0.045 * span,
        head_length=0.05 * span,
        fc=(0.9, 0.1, 0.1),
        ec=(0.9, 0.1, 0.1),
        linewidth=2.0,
        length_includes_head=True,
        zorder=5,
    )
    ax.plot(cx, cz, "o", color=(0.9, 0.1, 0.1), markersize=8, zorder=4)

    bar = _nice_length(span * 0.3)
    bx0 = xmin + 0.08 * span
    by0 = zmin + 0.08 * span
    ax.plot(
        [bx0, bx0 + bar],
        [by0, by0],
        "-",
        color=(0.1, 0.1, 0.1),
        linewidth=3,
        zorder=6,
        solid_capstyle="butt",
    )
    ax.text(
        bx0 + bar / 2,
        by0 + 0.02 * span,
        f"{_fmt_len(bar)}",
        ha="center",
        va="bottom",
        fontsize=7,
        color=(0.1, 0.1, 0.1),
        zorder=6,
    )

    seg = np.linalg.norm(np.diff(centers_xz[: i + 1], axis=0), axis=1).sum() if i > 0 else 0.0
    ax.text(
        0.03,
        0.97,
        f"path {_fmt_len(seg)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=(0.15, 0.35, 0.85),
        zorder=6,
    )

    # ── Pitch gauge ───────────────────────────────────────────────────────────
    gax.set_xlim(-1, 1)
    gax.set_ylim(-pitch_range, pitch_range)
    gax.set_xticks([])
    gax.set_yticks([-pitch_range, 0, pitch_range])
    gax.set_yticklabels([f"-{int(pitch_range)}", "0", f"+{int(pitch_range)}"], fontsize=7)
    gax.axhline(0, color=(0.55, 0.55, 0.55), linewidth=1.0)
    for s in gax.spines.values():
        s.set_color((0.6, 0.6, 0.6))
    p = float(np.clip(pitch_deg[i], -pitch_range, pitch_range))
    col = (0.10, 0.45, 0.85) if p >= 0 else (0.85, 0.45, 0.10)
    gax.bar([0], [p], width=1.2, color=col, alpha=0.55, zorder=2)
    gax.plot([0], [p], marker="^" if p >= 0 else "v", color=col, markersize=11, zorder=3)
    gax.text(0, pitch_range * 0.92, f"{p:+.0f}", ha="center", va="top", fontsize=8, color=col)

    # ── Prompt strip (wrapped) along the bottom ───────────────────────────────
    if prompt:
        import textwrap

        wrapped = "\n".join(textwrap.wrap(prompt, width=68)[:4])
        fig.text(
            0.03,
            0.13,
            wrapped,
            ha="left",
            va="top",
            fontsize=7.5,
            color=(0.15, 0.15, 0.15),
            family="monospace",
            wrap=True,
        )

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, hh = fig.canvas.get_width_height()
    img = buf.reshape(hh, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def render_bev_video(frames, viewmats, out_path, fps=16.0, fixed_half=None, prompt=""):
    """Render a side-by-side ``[RGB|BEV]`` mp4 and return the composed frames.

    Latent/pixel frame counts differ (VAE 4x temporal upsample); each pixel
    frame maps to the nearest pose so the arrow tracks the video.

    Args:
        frames (np.ndarray): ``(Fp,H,W,3)`` uint8 RGB pixel frames.
        viewmats (np.ndarray): ``(Fc,4,4)`` w2c OpenCV camera matrices.
        out_path (Path): destination mp4 path.
        fps (float): output frame rate.
        fixed_half (float | None): fixed BEV half-extent for cross-clip
            comparability; auto-fit per clip when ``None``.
        prompt (str): caption drawn (wrapped) along the bottom of the BEV panel.

    Returns:
        list[np.ndarray]: the composed ``[RGB|BEV]`` frames (for montage stitching).
    """
    centers_xz, head_xz, pitch_deg = poses_to_bev_tracks(viewmats)

    xmin, xmax = centers_xz[:, 0].min(), centers_xz[:, 0].max()
    zmin, zmax = centers_xz[:, 1].min(), centers_xz[:, 1].max()
    cx, cz = (xmin + xmax) / 2, (zmin + zmax) / 2
    half = fixed_half if fixed_half else max(xmax - xmin, zmax - zmin) * 0.6 + 1e-3
    lims = (cx - half, cx + half, cz - half, cz + half)

    n_pix = len(frames)
    n_pose = len(centers_xz)
    H = frames[0].shape[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=None)
    composed = []
    for i in range(n_pix):
        pi = int(round(i * (n_pose - 1) / max(n_pix - 1, 1)))
        bev = bev_frame(centers_xz, head_xz, pitch_deg, pi, lims, size_px=H, prompt=prompt)
        if bev.shape[0] != H:
            from PIL import Image

            bev = np.asarray(Image.fromarray(bev).resize((H, H)))
        frame = np.concatenate([frames[i], bev], axis=1)
        writer.append_data(frame)
        composed.append(frame)
    writer.close()
    print(f"    -> {out_path}  ({n_pix} px frames, {n_pose} poses @ {fps:.0f} fps)")
    return composed


def write_concat_video(clips, out_path, fps=16.0):
    """Stitch a list of clips (each a list of ``HxWx3`` uint8 frames) end-to-end.

    Clips may differ in width (BEV auto-fit) / frame count; each is padded on
    the right to the max width so they share a canvas, then played in sequence.

    Args:
        clips (list): list of clips, each a list of ``(H,W,3)`` uint8 frames.
        out_path (Path): destination montage mp4 path.
        fps (float): output frame rate.
    """
    clips = [c for c in clips if c]
    if not clips:
        print("    [concat] nothing to stitch")
        return
    max_h = max(c[0].shape[0] for c in clips)
    max_w = max(c[0].shape[1] for c in clips)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=None)
    n_frames = 0
    for clip in clips:
        for frame in clip:
            h, w = frame.shape[:2]
            if h != max_h or w != max_w:
                canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                canvas[:h, :w] = frame
                frame = canvas
            writer.append_data(frame)
            n_frames += 1
    writer.close()
    print(f"    -> {out_path}  ({len(clips)} clips, {n_frames} frames @ {fps:.0f} fps)")


# ─────────────────────── camera-motion scale statistics ──────────────────────
def _open_pose_shards(data_path: str) -> list[tuple]:
    """Open a camera LMDB for a poses-only sweep.

    Reads raw ``poses``/``intrinsics`` only (no latents), so it is fast enough
    to sweep the whole dataset. Handles single-LMDB and sharded layouts.

    Args:
        data_path (str): single LMDB dir or a sharded parent dir.

    Returns:
        list[tuple]: ``(env, poses_shape, intr_shape, n_rows)`` per shard.
    """

    def _shard(env):
        ps = _get_shape(env, "poses")
        is_ = _get_shape(env, "intrinsics")
        return env, ps, is_, ps[0]

    if _is_single_lmdb(data_path):
        return [_shard(_open(data_path))]
    return [_shard(env) for env in _open_shards(data_path)]


def _rot_angle_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    cos = (np.trace(ra.T @ rb) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _clip_motion(intr: np.ndarray, poses: np.ndarray) -> tuple[float, float, float]:
    """Return ``(trans_span, path_len, rot_span_deg)`` for one clip.

    Uses the same first-frame-aligned viewmats the trainer consumes, so the
    numbers match what PRoPE actually sees.
    """
    viewmats, _ = build_viewmats_and_Ks(intr, poses)  # (T,4,4) w2c aligned
    c2w = np.linalg.inv(viewmats)
    centers = c2w[:, :3, 3]
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    trans_span = float(d.max())
    path_len = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    r0 = c2w[0, :3, :3]
    rot_span = max((_rot_angle_deg(r0, c2w[i, :3, :3]) for i in range(len(c2w))), default=0.0)
    return trans_span, path_len, rot_span


def collect_scale_stats(data_path: str, max_rows: int = 0) -> dict:
    """Sweep the dataset and return per-clip motion metrics as arrays.

    Args:
        data_path (str): single LMDB dir or a sharded parent dir.
        max_rows (int): cap on rows scanned (``0`` = all).

    Returns:
        dict: ``trans``/``path``/``rot`` float arrays over valid clips, ``rows``
        (row index of each valid clip), plus ``total``/``scanned``/``n_bad``.
    """
    shards = _open_pose_shards(data_path)
    total = sum(s[3] for s in shards)
    limit = max_rows if max_rows > 0 else total

    trans, paths, rots, rows = [], [], [], []
    n_bad = 0
    scanned = 0
    global_row = 0
    for env, ps, is_, n_rows in shards:
        for j in range(n_rows):
            if scanned >= limit:
                break
            poses = _get_row(env, "poses", np.float32, j, shape=ps[1:])
            intr = _get_row(env, "intrinsics", np.float32, j, shape=is_[1:])
            try:
                ts, pl, rs = _clip_motion(intr, poses)
                if not (np.isfinite(ts) and np.isfinite(pl) and np.isfinite(rs)):
                    raise ValueError("non-finite")
                trans.append(ts)
                paths.append(pl)
                rots.append(rs)
                rows.append(global_row)
            except Exception:  # noqa: BLE001 - count unreadable/degenerate clips
                n_bad += 1
            scanned += 1
            global_row += 1
            if scanned % 2000 == 0:
                print(f"  ... scanned {scanned}/{limit}")
        if scanned >= limit:
            break
    return {
        "trans": np.asarray(trans),
        "path": np.asarray(paths),
        "rot": np.asarray(rots),
        "rows": np.asarray(rows),
        "total": total,
        "scanned": scanned,
        "n_bad": n_bad,
    }


def plot_scale_stats(stats: dict, out_path: Path, mad_k: float = 10.0, title: str = "") -> None:
    """Render a 1x2 scale-diagnostic figure (trans_span + rot_span) to disk.

    Args:
        stats (dict): output of :func:`collect_scale_stats`.
        out_path (Path): destination PNG path.
        mad_k (float): outlier cutoff drawn on the trans_span panel = median*k.
        title (str): figure suptitle.
    """
    trans = stats["trans"]
    rots = stats["rot"]
    if trans.size == 0:
        print("[scale] no valid clips to plot")
        return
    med = float(np.median(trans))
    cutoff = med * mad_k

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title or "camera motion scale diagnostics", fontsize=13)

    a = ax[0]
    a.hist(trans, bins=60, color="#2a5db0", alpha=0.85)
    a.axvline(med, color="green", ls="--", lw=1.2, label=f"median={med:.3g}")
    a.axvline(cutoff, color="red", ls="--", lw=1.2, label=f"cutoff={cutoff:.3g}")
    a.set_title("(a) trans_span distribution")
    a.set_xlabel("trans_span (world units)")
    a.set_ylabel("clips")
    a.legend(fontsize=8)

    c = ax[1]
    c.hist(rots, bins=50, color="#b06a2a", alpha=0.85)
    c.set_title("(b) rot_span distribution")
    c.set_xlabel("max rotation vs frame 0 (deg)")
    c.set_ylabel("clips")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)
    print(f"    -> {out_path}")


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def report_scale_stats(stats: dict, mad_k: float = 10.0, top: int = 20) -> None:
    """Print a text summary of the scale distribution + flagged outliers."""
    trans = stats["trans"]
    paths = stats["path"]
    rots = stats["rot"]
    rows = stats["rows"]
    print(
        f"\n[scale] valid={trans.size}  scanned={stats['scanned']}  "
        f"unreadable={stats['n_bad']}  (total {stats['total']})"
    )
    if trans.size == 0:
        return
    for name, arr in (("trans_span", trans), ("path_len", paths), ("rot_span°", rots)):
        print(
            f"  {name:11s} min={arr.min():.4g}  med={_pct(arr, 50):.4g}  "
            f"mean={arr.mean():.4g}  p95={_pct(arr, 95):.4g}  "
            f"p99.9={_pct(arr, 99.9):.4g}  max={arr.max():.4g}"
        )
    med = float(np.median(trans))
    cutoff = med * mad_k
    idx = np.where(trans > cutoff)[0]
    order = idx[np.argsort(-trans[idx])]
    print(
        f"  cutoff = median*{mad_k} = {cutoff:.4g}  -> "
        f"{order.size} clip(s) above ({100.0 * order.size / trans.size:.3f}%)"
    )
    for k in order[:top]:
        print(
            f"    row {int(rows[k]):>7}  trans_span={trans[k]:.4g}  "
            f"path_len={paths[k]:.4g}  rot_span={rots[k]:.1f}deg"
        )
    if order.size > top:
        print(f"    ... (+{order.size - top} more)")


def scale_plot(
    data_path: str, plot_path: Path, max_rows: int = 0, outlier_k: float = 10.0, title: str = ""
) -> None:
    """Sweep camera motion over the dataset, print a summary, save a figure."""
    print(f"[scale] scanning camera motion in {data_path} ...")
    stats = collect_scale_stats(data_path, max_rows=max_rows)
    report_scale_stats(stats, mad_k=outlier_k)
    print("[scale] rendering figure ...")
    plot_scale_stats(
        stats,
        plot_path,
        mad_k=outlier_k,
        title=title or f"camera motion scale — {Path(data_path).name}",
    )


def _close_dataset(dataset: CameraLatentLMDBDataset) -> None:
    """Close whatever LMDB environments a camera dataset holds open."""
    if getattr(dataset, "_sharded", False):
        for env in getattr(dataset, "_sds", []):
            env.close()
    else:
        dataset._ds.env.close()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Decode Wan camera-LMDB latents + overlay camera BEV to verify a dataset."
    )
    ap.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="LMDB dir (single or sharded parent) of a camera dataset",
    )
    ap.add_argument("--vae", type=str, default=_DEFAULT_VAE, help="Wan2.1 VAE .pth checkpoint")
    ap.add_argument(
        "--num_videos", type=int, default=8, help="number of random rows to sample and render"
    )
    ap.add_argument(
        "--rows",
        type=str,
        default=None,
        help="explicit comma-separated row indices (overrides sampling)",
    )
    ap.add_argument(
        "--name",
        type=str,
        default=None,
        help="tag for output dir + scale figure (default: lmdb dir name)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="output dir (default: <repo>/outputs/data_check/ds_check_<name>)",
    )
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument(
        "--range",
        type=float,
        default=None,
        dest="range_half",
        help="fixed BEV half-extent (world units) for cross-clip comparability",
    )
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device id")
    ap.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="VAE working dtype",
    )
    ap.add_argument(
        "--concat",
        type=int,
        default=10,
        help="also stitch the first N clips into one montage (0 disables)",
    )
    ap.add_argument(
        "--scale_plot",
        action="store_true",
        help="scale-only mode: skip rendering; scan every row's camera motion and "
        "write the scale figure + anomaly list (no VAE / GPU needed)",
    )
    ap.add_argument(
        "--no_scale",
        action="store_true",
        help="in the default (BEV) mode, skip the extra scale-stats sweep + figure",
    )
    ap.add_argument("--scale_max_rows", type=int, default=0, help="[scale] cap rows scanned")
    ap.add_argument(
        "--outlier_k",
        type=float,
        default=10.0,
        help="[scale] anomaly cutoff = median(trans_span) * k",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # A single tag drives both the output dir and the scale figure filename so
    # datasets that happen to share an lmdb dir name never overwrite each other.
    tag = args.name or Path(args.data_path.rstrip("/")).name
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else _REPO_ROOT / "outputs" / "data_check" / f"ds_check_{tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scale_path = out_dir / f"scale_stats_{tag}.png"

    # ── scale-only mode: read poses only, no VAE, no rendering ──
    if args.scale_plot:
        scale_plot(
            args.data_path,
            scale_path,
            max_rows=args.scale_max_rows,
            outlier_k=args.outlier_k,
            title=f"camera motion scale — {tag}",
        )
        return

    if not Path(args.vae).is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae}")

    dataset = CameraLatentLMDBDataset(args.data_path)
    total = len(dataset)
    print(f"[dataset] rows={total}  path={args.data_path}")
    if total == 0:
        raise SystemExit("empty dataset")

    if args.rows:
        indices = [int(r) for r in args.rows.split(",") if r.strip() != ""]
        bad = [r for r in indices if r < 0 or r >= total]
        if bad:
            raise ValueError(f"--rows out of range [0,{total}): {bad}")
        print(f"[sample] {len(indices)} explicit rows: {indices}")
    else:
        n = min(args.num_videos, total)
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(total, size=n, replace=False).tolist())
        print(f"[sample] {n} rows (seed={args.seed}): {indices}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"[vae] loading Wan21VAE ({args.dtype}) from {args.vae} ...")
    vae = Wan21VAE(vae_pth=args.vae, dtype=dtype, device=device)

    n_concat = min(args.concat, len(indices)) if args.concat > 0 else 0
    concat_clips = []

    for rank, idx in enumerate(indices):
        item = dataset[idx]
        clean_latent = item["clean_latent"]  # (F,C,H,W)
        viewmats = item["viewmats"].numpy().astype(np.float64)
        prompt = item.get("prompts", "") or ""
        print(
            f"[{rank + 1}/{len(indices)}] row {idx}: latent {tuple(clean_latent.shape)}, "
            f"poses {viewmats.shape[0]}  prompt={prompt[:60]!r}"
        )

        frames = decode_latent_to_frames(vae, clean_latent)

        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in prompt[:40])
        safe = safe.strip().replace(" ", "_")
        out_path = out_dir / f"{idx:06d}_{safe or 'clip'}.mp4"
        composed = render_bev_video(
            frames, viewmats, out_path, fps=args.fps, fixed_half=args.range_half, prompt=prompt
        )
        if rank < n_concat:
            concat_clips.append(composed)

    print(f"\nDone. {len(indices)} clips written to {out_dir}")

    if n_concat:
        concat_path = out_dir.parent / f"{out_dir.name}_concat{n_concat}.mp4"
        print(f"[concat] stitching first {n_concat} clips -> {concat_path}")
        write_concat_video(concat_clips, concat_path, fps=args.fps)

    # After the BEV videos, sweep the whole dataset for a scale-distribution
    # figure too, so one command yields both. Disable with --no_scale.
    if not args.no_scale:
        # lmdb forbids reopening an env already open in the process, so release
        # the dataset's handles before the poses-only sweep reopens them.
        _close_dataset(dataset)
        print("\n[scale] scanning full-dataset camera motion ...")
        scale_plot(
            args.data_path,
            scale_path,
            max_rows=args.scale_max_rows,
            outlier_k=args.outlier_k,
            title=f"camera motion scale — {tag}",
        )


if __name__ == "__main__":
    main()
