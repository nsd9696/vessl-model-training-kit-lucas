from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

import wandb
from dataset.arc import ARCDataset

from .base import BaseEvaluator


class ARCEvaluator(BaseEvaluator):
    def __init__(self, model_runner, wandb_config: Dict[str, Any]):
        super().__init__(model_runner, wandb_config)
        self.label_mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        self.label_names = list(
            map(lambda x: self.label_mapping[x], self.label_mapping)
        )
        self.label_to_id_dict = {l: i for i, l in enumerate(self.label_names)}
        self.label_to_id_dict.update({"1": 0, "2": 1, "3": 2, "4": 3, "5": 4})
        ## Update just in case 1,2,3,4,5
        self.language = "th"

    def evaluate(self, subsets: List[str]) -> Dict[str, Any]:
        results_data = {}
        metrics = {}
        accuracies = []
        for subset in subsets:
            subset_metrics = self._evaluate_subset(subset)
            metrics[subset] = subset_metrics
            accuracies.append(subset_metrics["accuracy"])
        # Calculate and log average accuracy across all subsets
        avg_accuracy = sum(accuracies) / len(accuracies)
        wandb.log({"average_accuracy": avg_accuracy})

        # Add average accuracy to the metrics dictionary
        metrics["average_accuracy"] = avg_accuracy
        return metrics

    def _evaluate_subset(self, subset: str) -> Dict[str, Any]:
        inputs, preds, golds, llm_responses = [], [], [], []
        prompts, labels = [], []
        arc_dset = ARCDataset(subset = "default", split="test", language=self.language)
        prompt_template = arc_dset.task
        print(f"Processing ARC-{subset}")
        with torch.inference_mode():
            for e, sample in tqdm(enumerate(arc_dset), total=len(arc_dset)):
                if e < len(preds):
                    continue

                prompt_text, label = sample
                prompts.append(prompt_text)
                labels.append(
                    self.label_to_id_dict[label] if type(label) == str else label
                )

                # Batch Inference
                if len(prompts) == 4:
                    hyps = self.model_runner.predict_classification_nlu(
                        prompts,
                        self.label_names,
                    )
                    answers = hyps.get("answers", None)
                    model_responses = hyps.get("model_responses", None)
                    if model_responses is not  None:
                        for prompt_text, hyp, label, model_response in zip(
                            prompts, answers, labels, model_responses
                        ):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                            llm_responses.append(model_response)
                    else:
                        for prompt_text, hyp, label in zip(
                            prompts, answers, labels
                        ):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                            llm_responses.append(None)
                    prompts, labels = [], []

        metrics = self.calculate_metrics(golds, preds)
        metrics.update(
            {
                "dataset": f"ARC-{subset}",
                "prompt_id": "QA",
                "prompt_lang": "English",
                "prompt_template": prompt_template,
            }
        )
        # Log metrics
        self.log_metrics(
            subset, metrics, golds, preds, self.label_names, llm_responses, inputs
        )

        return metrics
