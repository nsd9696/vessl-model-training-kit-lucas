from typing import Any, Dict, List
from .metric_utils import generation_metrics_fn


import pandas as pd
import torch
from tqdm import tqdm
import wandb

import wandb
from dataset.xlsum import XLSumDataset

from .base import BaseEvaluator
from evaluators.metric_utils import generation_metrics_fn

class XLSumEvaluator(BaseEvaluator):
    def __init__(self, model_runner, wandb_config: Dict[str, Any]):
        super().__init__(model_runner, wandb_config)

    def evaluate(self, subset: str = None, is_thinking: bool = False) -> Dict[str, Any]:
        results_data = {}
        metrics = {}
        accuracies = []
        inputs, preds, golds, llm_responses = [], [], [], []
        prompts, labels = [], []
        nlu_dset = XLSumDataset()
        prompt_template = nlu_dset.task
        print(f"Processing XL-SUM")



        with torch.inference_mode():
            for e, sample in tqdm(enumerate(nlu_dset), total=len(nlu_dset)):
                if e < len(preds):
                    continue

                prompt_text, label = sample
                prompts.append(prompt_text)
                labels.append(label)

                # Batch Inference
                if len(prompts) == 4:
                    hyps = self.model_runner.predict_generation(
                        prompts, is_thinking = is_thinking)
            
                    for prompt_text, hyp, label in zip(
                        prompts, hyps.get('responses', None), labels
                    ):
                        inputs.append(prompt_text)
                        preds.append(hyp)
                        golds.append(label)
                    prompts, labels = [], []

        
        metrics = generation_metrics_fn(preds, golds)
        print("Metrics:")
        print(metrics)
        metrics.update(
            {
                "dataset": f"XL-SUM",
                "prompt_id": "SUM",
                "prompt_lang": "Thai",
                "prompt_template": prompt_template,
            }
        )
        # Log metrics
        for key, value in metrics.items():
            wandb.log(
                {
                    f"{key}": value,
                }
            )
        return metrics
