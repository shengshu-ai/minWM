#!/usr/bin/env python3
"""overlay_from_manifest.py — 读推理产物 manifest.json，给每条视频叠加 WASD/KIJL
按键指示器，再拼接成一个 final_with_keys.mp4。

与 batch_overlay_concat.py 的区别：按键序列不再靠人工查文件名，而是直接从
manifest.json 每个 item 的 ``trajectory`` (如 ``d*8,i*5,l*6``) 换算得到。每段
动作占的像素帧数按 VAE 时间解码 (20 个 latent 帧 -> 77 像素帧：首帧 1 帧、其余
每帧 4 帧) 的约定分配，与历史 demos/batch_overlay.py 的公式一致，总帧数恒为 77。

用法:
  python demos/overlay_from_manifest.py --input-dir outputs/infer_xxx/15000
  # 自定义输出文件名 (默认 final_with_keys.mp4，落在 --input-dir 下):
  python demos/overlay_from_manifest.py -d <dir> -o final_with_keys_15000.mp4
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlay_keys import process_video  # noqa: E402

def trajectory_to_sequence(trajectory):
    """把轨迹字符串换算成 overlay_keys 的按键序列 (键名:帧数,...)。

    轨迹形如 ``d*8,i*5,l*6``：每段 ``键*次数``，共 19 段动作对应 20 个 latent 帧。
    VAE 时间解码把 20 latent 帧展开成 77 像素帧 (首帧 1 帧、其余每 latent 帧 4 帧)，
    故每段占的像素帧数按段位置分配 (与历史 batch_overlay.py 公式一致)：
      - 唯一一段: 1 + 4*(n-1) + 4
      - 第一段:   1 + 4*(n-1)
      - 最后一段: 4 + 4*n
      - 中间段:   4*n

    Args:
        trajectory (str): 轨迹字符串，如 ``d*8,i*5,l*6``。

    Returns:
        str | None: overlay_keys 的 ``--sequence`` 串 (如 ``D:29,I:20,L:28``)，
        轨迹为空或无法解析时返回 None。
    """
    if not trajectory:
        return None
    segs = []
    for part in trajectory.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, num = part.partition("*")
        key = key.strip().upper()
        n = int(num) if num.strip() else 1
        segs.append((key, n))
    if not segs:
        return None

    out = []
    for i, (key, n) in enumerate(segs):
        if len(segs) == 1:
            frames = 1 + 4 * (n - 1) + 4
        elif i == 0:
            frames = 1 + 4 * (n - 1)
        elif i == len(segs) - 1:
            frames = 4 + 4 * n
        else:
            frames = 4 * n
        out.append(f"{key}:{frames}")
    return ",".join(out)


def load_manifest_items(input_dir):
    """读 manifest.json，返回 (video_path, trajectory) 列表，按 index 升序。

    Args:
        input_dir (str): 含 manifest.json 与各 mp4 的推理输出目录。

    Returns:
        list[tuple[str, str | None]]: 每条 (视频绝对路径, 轨迹字符串)。

    Raises:
        FileNotFoundError: 若 input_dir 下没有 manifest.json。
    """
    manifest_path = os.path.join(input_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"no manifest.json in {input_dir}")
    with open(manifest_path, encoding="utf-8") as f:
        items = json.load(f)["items"]

    result = []
    for item in sorted(items, key=lambda x: x.get("index", 0)):
        video = item.get("video")
        # manifest 存的是生成时的绝对路径；若目录被搬走，回退到按 basename 就地找。
        if not video or not os.path.exists(video):
            if video:
                cand = os.path.join(input_dir, os.path.basename(video))
                video = cand if os.path.exists(cand) else None
        if video:
            result.append((video, item.get("trajectory")))
    return result


def concat_videos(segment_paths, output_path):
    """用 ffmpeg concat demuxer 重编码拼接多个片段 (兼容异源分辨率/帧率)。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = f.name
        for p in segment_paths:
            escaped = os.path.abspath(p).replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path,
    ]
    print(f"\n拼接 {len(segment_paths)} 个片段 -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(list_path)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 拼接失败:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="从 manifest.json 叠加按键并拼接")
    parser.add_argument("--input-dir", "-d", required=True, help="推理输出目录 (含 manifest.json)")
    parser.add_argument("--output", "-o", default="final_with_keys.mp4",
                        help="输出文件名或路径；相对名落在 --input-dir 下 (默认 final_with_keys.mp4)")
    parser.add_argument("--keep-temp", action="store_true", help="保留每段叠加后的中间文件")
    args = parser.parse_args()

    items = load_manifest_items(args.input_dir)
    if not items:
        print("错误: manifest 里没有可用视频", file=sys.stderr)
        sys.exit(1)

    output = args.output if os.path.isabs(args.output) or os.path.dirname(args.output) \
        else os.path.join(args.input_dir, args.output)

    tmp_dir = tempfile.mkdtemp(prefix="overlay_segments_")
    segment_paths = []
    config = []
    try:
        for idx, (video, trajectory) in enumerate(items):
            sequence = trajectory_to_sequence(trajectory)
            print(f"\n=== [{idx + 1}/{len(items)}] {os.path.basename(video)} ===")
            print(f"  轨迹: {trajectory}  ->  序列: {sequence}")
            seg_path = os.path.join(tmp_dir, f"seg_{idx:03d}.mp4")
            if not process_video(video, seg_path, sequence=sequence):
                print(f"错误: 处理失败，中止: {video}", file=sys.stderr)
                sys.exit(1)
            segment_paths.append(seg_path)
            config.append({"input": video, "trajectory": trajectory, "sequence": sequence})

        concat_videos(segment_paths, output)
        cfg_path = os.path.join(args.input_dir, "overlay_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n全部完成! 最终输出: {output}\n叠加配置: {cfg_path}")
    finally:
        if not args.keep_temp:
            for p in segment_paths:
                if os.path.exists(p):
                    os.remove(p)
            if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
        else:
            print(f"中间片段保留在: {tmp_dir}")


if __name__ == "__main__":
    main()
