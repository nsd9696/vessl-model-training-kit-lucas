from datasets import Dataset, load_dataset, concatenate_datasets
from trl import GKDConfig, GKDTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils.quantization_config import BitsAndBytesConfig
import argparse
from transformers.trainer_callback import TrainerCallback
import torch
from model.model import ChatMessage
import json
import os
import time
from peft import LoraConfig, get_peft_model, TaskType

def safe_load_lora_adapter(model, adapter_path):
    """
    Safely load a LoRA adapter, handling missing keys gracefully.
    """
    from peft import PeftModel
    try:
        # Try to load the adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Successfully loaded existing LoRA adapter.")
        return model, True
    except Exception as e:
        print(f"Warning: Failed to load existing LoRA adapter: {e}")
        print("This usually means the adapter was trained on a different model architecture.")
        print("The adapter will be ignored and a new one will be created.")
        return model, False

def prepare_model_for_training(model):
    """
    Ensure the model is properly prepared for training with LoRA.
    """
    # Set model to training mode
    model.train()
    
    # Ensure LoRA parameters require gradients
    lora_params_found = False
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            lora_params_found = True
    
    # If no LoRA parameters found, try to find adapter parameters
    if not lora_params_found:
        for name, param in model.named_parameters():
            if 'adapter' in name or 'lora' in name.lower():
                param.requires_grad = True
                lora_params_found = True
    
    # Print trainable parameters
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        # Manual calculation for non-PEFT models
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.2f}%")
        
        # If still no trainable parameters found, print a warning
        if trainable_params == 0:
            print("WARNING: No trainable parameters found! This might indicate an issue with LoRA configuration.")
            print("Available parameter names:")
            for name, param in model.named_parameters():
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    print(f"  {name}: requires_grad={param.requires_grad}")
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="GKD Training")
    parser.add_argument("--teacher_model_id", type=str, default="/root/data/vessl-ai-kt-api-models/vessl-ai-kt/output_prune/structured_pruned_20250723_090423_22.47B/", help="The teacher model id")
    parser.add_argument("--student_model_id", type=str, default="/root/data/vessl-ai-kt-debugging/solar-pro-8.49b-harder-training-1-epoch/", help="The student model id")
    parser.add_argument("--train_for", type=str, default="thaiexam", help="The metric to train for")
    parser.add_argument("--hf_or_gen", type=str, default="gen", help="The dataset type: hf or gen")
    parser.add_argument("--train_dataset", type=str, default="2500_hard_training", help="The train dataset")
    parser.add_argument("--eval_dataset", type=str, default="source", help="The eval dataset")
    parser.add_argument("--distilled_model_id", type=str, default="./solar-pro-GKD-on-8.49-thaiexam/", help="The distilled model id")
    parser.add_argument("--save_steps", type=int, nargs="+", default = [], help="The number of steps to save the model")
    parser.add_argument("--add_think", default = True, type=bool, help="Whether to add think to the generated dataset")
    
    # qLoRA specific arguments
    parser.add_argument("--use_lora", default = False, type=bool, help="Whether to use qLoRA (4-bit quantized LoRA) for training")
    parser.add_argument("--lora_r", type=int, default=16, help="qLoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="qLoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="qLoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], help="Target modules for qLoRA")
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"], help="qLoRA bias type")
    
    return parser.parse_args()

class CustomSaveCallback(TrainerCallback):
    """
    Custom save callback to save the model at the end of each step.
    """
    def __init__(self, save_steps):
        self.save_steps = set(save_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_steps:
            control.should_save = True


def thaiexam_to_message_format(batch):
    """
    Convert the Huggingface thaiexam dataset to the message format.
    
    Args:
        batch: The batch of data to convert.
    Returns:
        The converted batch.
    """
    messages = []
    batch_size = len(next(iter(batch.values())))
    for i in range(batch_size):
        example = {k: v[i] for k, v in batch.items()}
        choices = [f"{k}) {example[k]}" for k in ['a', 'b', 'c', 'd', 'e']]
        question_with_choices = f"{example['question']}\n" + "\n".join(choices)
        answer_text = example[example["answer"]]
        messages.append([
            {"role": "user", "content": question_with_choices},
            {"role": "assistant", "content": answer_text},
        ])
    return {"messages": messages}

def generated_thaiexam_to_message_format(batch):
    """
    Convert the generated thaiexam dataset to the message format.
    """
    messages = []
    batch_size = len(batch)
    for i in range(batch_size):
        if type(batch[i]['choices']) == dict:
            choices = [f"{k}) {batch[i]['choices'][k]}" for k in batch[i]['choices'].keys()]
            question_with_choices = f"{batch[i]['question']}\n" + "\n".join(choices)
        elif type(batch[i]['choices']) == str:
            choices = batch[i]['choices']
            question_with_choices = f"{batch[i]['question']}\n" + choices
        if type(batch[i]['answer']) == dict:
            answer_text = batch[i]['answer']['label']
        elif type(batch[i]['answer']) == str:
            answer_text = batch[i]['answer']
        messages.append([
            {"role": "user", "content": question_with_choices},
            {"role": "assistant", "content": answer_text},
        ])
    return {"messages": messages}

def mtbench_to_message_format(batch, add_think=True):
    """
    Convert the Huggingface mtbench dataset to the message format.
    """
    messages = []
    batch_size = len(batch['turns'])
    for i in range(batch_size):
        turns = batch['turns'][i]
        references = batch['reference'][i]
        if references == None or references == []:
            continue
        for n in range(len(turns)):
            history = ""
            for k in range(n):
                history += f"<|im_start|>user\n{turns[k]}<|im_end|>\n<|im_start|>assistant\n{references[k]}<|im_end|>\n"
            history += f"<|im_start|>user\n{turns[n]}<|im_end|>\n"
            if not add_think:
                history += f"<think> </think>\n"
            messages.append([
                {"role": "user", "content": history}, 
                {"role": "assistant", "content": references[n]}
            ])
    return {"messages": messages}

def generated_mtbench_to_message_format(batch, add_think = True):
    """
    Convert the generated mtbench dataset to the message format.
    """
    messages = []
    batch_size = len(batch)
    for i in range(batch_size):
        turns = batch[i]['turns']
        try:
            references = batch[i]['reference']
        except:
            print(f"No reference for {i}")
            references = [""] * len(turns)
        if references == None or references == [] or len(turns) != len(references):
            continue
        for n in range(len(turns)):
            history = ""
            for k in range(n):
                history += f"<|im_start|>user\n{turns[k]}<|im_end|>\n<|im_start|>assistant\n{references[k]}<|im_end|>\n"
            history += f"<|im_start|>user\n{turns[n]}<|im_end|>\n<|im_start|>assistant\n"
            if not add_think:
                history += f"<think> </think>\n"
            try:
                messages.append([
                    {"role": "user", "content": history},
                    {"role": "assistant", "content": references[n]}
                ])
            except:
                print(f"No reference for {i}")
                messages.append([
                    {"role": "user", "content": history},
                    {"role": "assistant", "content": ""}
                ])
    return {"messages": messages}

if __name__ == "__main__":
    args = parse_args()
    teacher_model_id = args.teacher_model_id
    student_model_id = args.student_model_id

    tokenizer = AutoTokenizer.from_pretrained(student_model_id)
    tokenizer.padding_side = 'left'
    
    # Load the student model with qLoRA configuration

    if args.use_lora:
        print("Loading student model with qLoRA (4-bit quantization)...")
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Check if the model directory contains LoRA adapter files
        if os.path.exists(os.path.join(student_model_id, "adapter_config.json")):
            print("Found existing LoRA adapter in model directory. Loading with existing PEFT configuration...")
            # Load the model with existing LoRA weights
            model = AutoModelForCausalLM.from_pretrained(
                student_model_id, 
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2", 
                device_map="auto", 
                trust_remote_code=True
            )
            # Load the existing LoRA adapter
            model, adapter_loaded = safe_load_lora_adapter(model, student_model_id)
            
            if adapter_loaded:
                print("Adapter loaded successfully")
                # Ensure the model is properly prepared for training
                model = prepare_model_for_training(model)
            else:
                # If loading fails, apply new LoRA configuration
                print("Failed to load existing adapter, applying new LoRA configuration...")
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=args.lora_target_modules,
                    lora_dropout=args.lora_dropout,
                    bias=args.lora_bias,
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)
                model = prepare_model_for_training(model)
        else:
            print("No existing LoRA adapter found. Loading base model and applying new LoRA configuration...")
            # Load the base model without existing LoRA
            model = AutoModelForCausalLM.from_pretrained(
                student_model_id, 
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2", 
                device_map="auto", 
                trust_remote_code=True
            )
            
            print("Applying qLoRA configuration to student model...")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            # Ensure the model is properly prepared for training
            model = prepare_model_for_training(model)
    else:
        # Load the student model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            student_model_id, 
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
    
    # Load teacher model (always full model, no LoRA)
    print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id, 
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )

    train_for = args.train_for
    
    if train_for != "thaiexam" and train_for != "mtbench":
        raise ValueError("Invalid metric to train for")
    
    if train_for == "thaiexam":
        if args.hf_or_gen == "hf":
            train_dataset = load_dataset("scb10x/thai_exam", name = args.train_dataset, split="test")
            eval_dataset = []
            for dataset in ['a_level', 'ic', 'onet', 'tgat', 'tpat1']:
                if dataset != args.train_dataset:
                    eval_dataset.append(load_dataset("scb10x/thai_exam", name = dataset, split="test"))
            eval_dataset = concatenate_datasets(eval_dataset)
            train_dataset = train_dataset.map(thaiexam_to_message_format, batched=True)
            eval_dataset = eval_dataset.map(thaiexam_to_message_format, batched=True)
        elif args.hf_or_gen == "gen":
            file_directory = input("Enter the file directory: \n")
            with open(f"{file_directory}/thaiexam_{args.train_dataset}.jsonl", "r") as f:
                train_dataset = [json.loads(line) for line in f]
            with open(f"{file_directory}/thaiexam_{args.eval_dataset}.jsonl", "r") as f:
                eval_dataset = [json.loads(line) for line in f]
            train_dataset = generated_thaiexam_to_message_format(train_dataset)
            eval_dataset = generated_thaiexam_to_message_format(eval_dataset)
            train_dataset = [{"messages": m} for m in train_dataset['messages']]
            eval_dataset = [{"messages": m} for m in eval_dataset['messages']]
            train_dataset = Dataset.from_list(train_dataset)
            eval_dataset = Dataset.from_list(eval_dataset)
    elif train_for == "mtbench":
        if args.hf_or_gen == "hf":
            mtbench_dataset = load_dataset("ThaiLLM-Leaderboard/mt-bench-thai", split="train")
            # Use the correct method to split the dataset
            split_dataset = mtbench_dataset.train_test_split(test_size=0.5, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            train_dataset = train_dataset.map(lambda x: mtbench_to_message_format(x, add_think=args.add_think), batched=True, remove_columns=train_dataset.column_names)
            eval_dataset = eval_dataset.map(lambda x: mtbench_to_message_format(x, add_think=args.add_think), batched=True, remove_columns=eval_dataset.column_names)
        elif args.hf_or_gen == "gen":
            file_directory = input("Enter the file directory: \n")
            train_dataset = []
            eval_dataset = []
            with open(f"{file_directory}{args.train_dataset}.jsonl", "r") as f:
                generated_dataset = [json.loads(line) for line in f]
            with open(f"{file_directory}{args.eval_dataset}.jsonl", "r") as f:
                generated_eval_dataset = [json.loads(line) for line in f]
            """
            for i, data in enumerate(generated_dataset):
                if (i % 1112) < 1056 and i < 1112:
                    train_dataset.append(data)
                elif (i - 1112) & 1111 < 1056 and 10000 > i > 1112:
                    train_dataset.append(data)
                elif (i - 10000) % 1112 < 1056 and 11112 > i > 10000:
                    train_dataset.append(data)
                elif (i - 11112) % 1112 < 1056 and 20000 > i > 11112:
                    train_dataset.append(data)
                else:
                    eval_dataset.append(data)
            """
            train_dataset = generated_dataset[:20000]
            eval_dataset = generated_eval_dataset
            train_dataset = generated_mtbench_to_message_format(train_dataset, add_think = args.add_think)
            eval_dataset = generated_mtbench_to_message_format(eval_dataset, add_think = args.add_think)
            train_dataset = [{"messages": m} for m in train_dataset['messages']]
            eval_dataset = [{"messages": m} for m in eval_dataset['messages']]
            train_dataset = Dataset.from_list(train_dataset)
            eval_dataset = Dataset.from_list(eval_dataset)

    distilled_model_id = args.distilled_model_id

    
    training_args = GKDConfig(
        output_dir = distilled_model_id,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=25,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=125,
        num_train_epochs=5,
        #resume_from_checkpoint=None,  # Will automatically find the latest checkpoint
        learning_rate = 2e-5,
        lmbda = 0.5,
        seq_kd = False,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.05,
        save_strategy="steps",
        save_steps = 625,
        # Add safety parameters to prevent indexing errors
        max_length=1024,  # Limit sequence length
        max_new_tokens=256,  # Limit generation length
    )
    
    trainer = GKDTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        #data_collator=data_collator,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            CustomSaveCallback(save_steps=args.save_steps),
        ],
    )
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_time = end_time - start_time
    
    # Save training configuration to JSON file
    import json
    from datetime import datetime
    
    # Collect all arguments
    config_dict = {
        "timestamp": datetime.now().isoformat(),
        "training_time": f"{training_time:.2f} seconds",
        "parser_arguments": {
            "teacher_model_id": args.teacher_model_id,
            "student_model_id": args.student_model_id,
            "train_for": args.train_for,
            "hf_or_gen": args.hf_or_gen,
            "train_dataset": args.train_dataset,
            "eval_dataset": args.eval_dataset,
            "distilled_model_id": args.distilled_model_id,
            "save_steps": args.save_steps,
            "add_think": args.add_think,
            "use_lora": args.use_lora,
        },
        "qlora_config": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "lora_bias": args.lora_bias,
            "quantization": "4-bit (nf4)",
            "double_quant": True,
        } if args.use_lora else None,
        "training_arguments": {
            "output_dir": training_args.output_dir,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "report_to": training_args.report_to,
            "logging_strategy": training_args.logging_strategy,
            "logging_steps": training_args.logging_steps,
            "logging_first_step": training_args.logging_first_step,
            "eval_strategy": training_args.eval_strategy,
            "num_train_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_ratio": training_args.warmup_ratio,
            "lmbda": training_args.lmbda,
            "seq_kd": training_args.seq_kd,
            "save_strategy": training_args.save_strategy,
            "save_steps": training_args.save_steps,
            "max_new_tokens": training_args.max_new_tokens,
            "max_length": training_args.max_length,
            "temperature": training_args.temperature,
            "beta": training_args.beta,
            "disable_dropout": training_args.disable_dropout,
        }
    }
    
    # Save to JSON file
    config_file_path = os.path.join(distilled_model_id, "training_config.json")
    os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
    
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Training configuration saved to: {config_file_path}")