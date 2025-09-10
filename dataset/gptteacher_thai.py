from typing import List, Literal

from datasets import load_dataset

from dataset.dataset import BaseDataset


# === Dataset Class ===
class GPTTeacher20KThai(BaseDataset):
    def __init__(
        self,
        split: Literal["train"] = "train",
    ):
        self.dataset = load_dataset("Thaweewat/gpteacher-20k-th", split=split)
        self.initialize()

    def make_conversation(self, example):
        return {
            "prompt": [
                {"role": "system", "content": example["instruction"]},
                {"role": "user", "content": example["input"]},
            ],
            "completions": [
                {"role": "assistant", "content": example["output"].strip()}
            ],
        }

    def initialize(self):
        self.dataset = self.dataset.map(self.make_conversation)
        self.dataset = self.dataset.remove_columns(
            [
                "instruction",
                "input",
                "output",
            ]
        )

    def get_dataset(self):
        return self.dataset
