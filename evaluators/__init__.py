from typing import Any, Dict

from .arc import ARCEvaluator
from .base import BaseEvaluator
from .thaiexam import ThaiExamEvaluator
from .xlsum import XLSumEvaluator
from .mtbench import MTBenchEvaluator
def get_evaluator(
    dataset_name: str, model_runner, wandb_config: Dict[str, Any]
) -> BaseEvaluator:
    """Factory function to get the appropriate evaluator"""
    evaluators = {
        "thaiexam": ThaiExamEvaluator,
        "arc": ARCEvaluator,
        "xlsum": XLSumEvaluator,    
        "mtbench": MTBenchEvaluator,
        # Add more evaluators here as they become available
    }

    if dataset_name not in evaluators:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return evaluators[dataset_name](model_runner, wandb_config)
