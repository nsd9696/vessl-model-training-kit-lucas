#!/bin/bash
python evaluate_llm.py --model_id openthai14b-quantized.model --dataset mtbench --project_name mtbench-eval --is_gguf True
