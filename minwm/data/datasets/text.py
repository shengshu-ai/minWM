"""Text / prompt datasets."""

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Line-delimited prompt file, optionally with an extended-prompt companion."""

    def __init__(self, prompt_path: str, extended_prompt_path: str | None = None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self) -> int:
        return len(self.prompt_list)

    def __getitem__(self, idx: int) -> dict:
        batch = {"prompts": self.prompt_list[idx], "idx": idx}
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch
