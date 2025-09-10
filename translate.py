import argparse
import json
import os
from typing import List

from datasets import load_dataset
from openai import OpenAI
from tooldantic import Field
from tooldantic import OpenAiResponseFormatBaseModel as BaseModel

from settings import load_settings


class ARCQA_Thai(BaseModel):
    question: str = Field(description="The question translated to Thai")
    choices: List[str] = Field(description="Text of the choice translated to Thai")


system_instruction = (
    "You are a translator tasked with translating an English multiple-choice question into Thai. "
    "Translate the question and each choice clearly, maintaining the original meaning and formatting."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate dataset using OpenAI models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-2025-04-14",
        help="OpenAI model to use for translation (e.g., gpt-4, gpt-3.5-turbo)",
    )
    return parser.parse_args()


def arc_prompt_mapper(item):
    return {
        "prompt": f"Question: {item['question']}\n\nChoices:\n{item['choices']['text']}"
    }


def arc_translate_mapper(item):
    return {
        "prompt": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": item["prompt"]},
        ]
    }


def main():
    args = parse_args()

    test_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    test_dataset = test_dataset.map(arc_prompt_mapper)
    test_dataset = test_dataset.map(arc_translate_mapper)

    settings = load_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    tasks = []
    for data in test_dataset:
        task = {
            "custom_id": data["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model,
                "temperature": 0.0,
                "messages": data["prompt"],
                "response_format": ARCQA_Thai.model_json_schema(),
            },
        }
        tasks.append(task)

    # Create tasks directory if it doesn't exist
    file_name = "tasks/batch_translate_arc_challenge.jsonl"
    os.makedirs("tasks", exist_ok=True)

    # Write tasks to file
    with open(file_name, "w") as file:
        for obj in tasks:
            file.write(json.dumps(obj) + "\n")

    # Create batch file
    batch_file = client.files.create(file=open(file_name, "rb"), purpose="batch")

    # Create batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"Batch job created: {batch_job}")


if __name__ == "__main__":
    main()
