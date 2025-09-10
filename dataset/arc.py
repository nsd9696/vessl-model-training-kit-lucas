from typing import List, Literal

from datasets import load_dataset

from dataset.dataset import BaseDataset
from dataset.prompt_template import TASK_TO_PROMPT
from dataset.utils import Choice, Example


# === Dataset Class ===
class ARCDataset(BaseDataset):
    def __init__(
        self,
        subset: Literal["ARC-Easy", "ARC-Challenge"] = "ARC-Easy",
        split: Literal["train", "test"] = "train",
        language: Literal["th", "en"] = "en",
    ):
        self.dataset = (
            load_dataset("allenai/ai2_arc", subset, split=split)
            if language == "en"
            else load_dataset("enAIble-hyunseung/ARC-thai", subset, split=split)
        )
        self.schema = "QA"
        self.language = language
        self.task = (
            TASK_TO_PROMPT["eng"][self.schema][0]
            if language == "en"
            else TASK_TO_PROMPT["tha"][self.schema][0]
        )

    def arc_item2model(self, item: dict) -> Example:
        def arc_choices2choices(item: dict) -> List[Choice]:
            choices_text = (
                item["choices"]["text"] if self.language == "en" else item["choices"]
            )
            choices = []
            for k in range(len(choices_text)):  # a, b, c, d, e
                choices.append(Choice(letter=chr(97 + k), text=choices_text[k]))
            return choices

        return Example(
            question=item["question"],
            choices=arc_choices2choices(item),
            answer=item["answerKey"].lower(),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        item = self.arc_item2model(data)
        prompt = self._to_prompt(item, self.task)
        return prompt, item.answer
