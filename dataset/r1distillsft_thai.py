from typing import List, Literal

from datasets import load_dataset

from dataset.dataset import BaseDataset

SYSTEM_PROMPT = "บทสนทนาระหว่างผู้ใช้งาน (User) และผู้ช่วย (Assistant) ผู้ใช้งานถามคำถามและผู้ช่วยจะทำการแก้ไขปัญหา ผู้ช่วยจะคิดถึงกระบวนการให้เหตุผลในใจก่อน แล้วจึงให้คำตอบแก่ผู้ใช้งาน กระบวนการให้เหตุผลและคำตอบจะถูกล้อมรอบด้วยแท็ก <think> </think> และ <answer> </answer> ตามลำดับ เช่น <think> กระบวนการให้เหตุผลตรงนี้ </think><answer> คำตอบตรงนี้ </answer>"


# === Dataset Class ===
class ThaiR1DistillSFT(BaseDataset):
    def __init__(
        self,
        split: Literal["train", "test"] = "train",
    ):
        self.dataset = load_dataset("iapp/Thai-R1-Distill-SFT", split=split)
        self.initialize()

    def make_conversation(self, example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
            "completions": [
                {
                    "role": "assistant",
                    "content": example["reannotated_assistant_content"].strip()
                    + "</think><answer>"
                    + example["solution"].strip()
                    + "</answer>",
                }
            ],
        }

    def initialize(self):
        self.dataset = self.dataset.map(self.make_conversation)
        self.dataset = self.dataset.remove_columns(
            [
                "reannotated_assistant_content",
                "problem",
                "solution",
                "id",
                "source",
                "verified",
                "quality_metrics",
            ]
        )

    def get_dataset(self):
        return self.dataset
