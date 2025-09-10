from typing import List, Literal

from datasets import load_dataset

from dataset.dataset import BaseDataset
from dataset.prompt_template import TASK_TO_PROMPT
from dataset.utils import item2model


# === Dataset Class ===
class XLSumDataset(BaseDataset):
    def __init__(
        self,
        subset: Literal["thai"] = "thai",
    ):
        self.dataset = load_dataset("csebuetnlp/xlsum", subset, split="test")
        self.schema = "SUM"
        self.task = TASK_TO_PROMPT["tha"][self.schema][0]

    def __len__(self):
        return len(self.dataset)

    def _to_prompt(self, input, prompt):
        prompt = prompt.replace("[INPUT]", input)
        return prompt

    def __getitem__(self, idx):
        data = self.dataset[idx]
        prompt = self._to_prompt(data["text"], self.task)
        return prompt, data["summary"]

    def __len__(self):
        return len(self.dataset)

    def get_dataset(self):
        return self.dataset
