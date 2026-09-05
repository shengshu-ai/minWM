"""Image + caption pair dataset."""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class TextImagePairDataset(Dataset):
    """Image directory paired with a ``target_crop_info_*.json`` metadata file.

    ``data_dir`` must contain exactly one ``target_crop_info_<ratio>.json`` and
    a matching ``<ratio>/`` image subdirectory.
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        eval_first_n: int = -1,
        pad_to_multiple_of: int | None = None,
    ):
        self.transform = transform
        data_dir = Path(data_dir)

        metadata_files = list(data_dir.glob("target_crop_info_*.json"))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        aspect_ratio = metadata_path.stem.split("_")[-1]
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        if eval_first_n != -1:
            self.metadata = self.metadata[:eval_first_n]

        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        item = self.metadata[idx]
        image = Image.open(self.image_dir / item["file_name"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "prompts": item["caption"],
            "target_bbox": item["target_crop"]["target_bbox"],
            "target_ratio": item["target_crop"]["target_ratio"],
            "type": item["type"],
            "origin_size": (item["origin_width"], item["origin_height"]),
            "idx": idx,
        }
