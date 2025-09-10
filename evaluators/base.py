from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

import wandb


class BaseEvaluator(ABC):
    def __init__(self, model_runner, wandb_config: Dict[str, Any]):
        self.model_runner = model_runner
        self.wandb_config = wandb_config

    @abstractmethod
    def evaluate(self, subsets: List[str]) -> Dict[str, Any]:
        pass

    def log_metrics(
        self,
        subset: str,
        metrics: Dict[str, Any],
        golds: List[int],
        preds: List[int],
        label_names: List[str],
        llm_responses: List[Dict[str, Any]],
        prompts: List[str],
    ):
        """Common method to log metrics to wandb"""
        # Log basic metrics

        wandb.log(
            {
                f"{subset}/accuracy": metrics["accuracy"],
                f"{subset}/micro_f1": metrics["micro_f1"],
                f"{subset}/macro_f1": metrics["macro_f1"],
                f"{subset}/weighted_f1": metrics["weighted_f1"],
            }
        )

        # Log confusion matrix
        wandb.log(
            {
                f"{subset}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=golds, preds=preds, class_names=label_names
                )
            }
        )

        # Log detailed metrics
        wandb.log(
            {
                f"{subset}/detailed_metrics": wandb.Table(
                    dataframe=pd.DataFrame(metrics, index=[0])
                )
            }
        )

        if len(llm_responses) > 0:
            if llm_responses[0] is not None:
                keys = llm_responses[0].keys()
                inputs = defaultdict(list)
                for llm_response in llm_responses:
                    for key in keys:
                        inputs[key].append(llm_response[key])
                inputs["input"] = prompts
                wandb.log(
                    {f"{subset}/llm_responses": wandb.Table(dataframe=pd.DataFrame(inputs))}
                )

    def calculate_metrics(self, golds: List[int], preds: List[int]) -> Dict[str, Any]:
        """Calculate common classification metrics"""
        cls_report = classification_report(golds, preds, output_dict=True)
        micro_f1, micro_prec, micro_rec, _ = precision_recall_fscore_support(
            golds, preds, average="micro"
        )

        return {
            "accuracy": cls_report["accuracy"],
            "micro_prec": micro_prec,
            "micro_rec": micro_rec,
            "micro_f1": micro_f1,
            "macro_prec": cls_report["macro avg"]["precision"],
            "macro_rec": cls_report["macro avg"]["recall"],
            "macro_f1": cls_report["macro avg"]["f1-score"],
            "weighted_prec": cls_report["weighted avg"]["precision"],
            "weighted_rec": cls_report["weighted avg"]["recall"],
            "weighted_f1": cls_report["weighted avg"]["f1-score"],
        }
