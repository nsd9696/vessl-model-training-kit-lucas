from typing import List, Literal

from datasets import load_dataset

from dataset.dataset import BaseDataset
from dataset.prompt_template import TASK_TO_PROMPT
from dataset.utils import item2model


# === Dataset Class ===
class ThaiExamDataset(BaseDataset):
    def __init__(
        self,
        subset: Literal["a_level", "ic", "onet", "tpat", "tgat"] = "a_level",
        split: Literal["train", "test"] = "train",
        model_id: str = "scb10x/llama3.2-typhoon2-1b-instruct",
    ):
        self.train_dataset = load_dataset("scb10x/thai_exam", subset, split="train")
        self.dataset = load_dataset("scb10x/thai_exam", subset, split=split)
        self.schema = "QA"
        self.task = TASK_TO_PROMPT["tha"][self.schema][0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        item = item2model(data)
        prompt = self._to_prompt(item, self.task)
        return prompt, item.answer
