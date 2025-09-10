#!/usr/bin/env python
"""
Conservative pruning strategy for Solar-pro 4x22B → Dense-8B
Focuses on layer dropping and expert collapsing without neuron-level modifications.
"""

import argparse
import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def average_experts_to_dense(layer):
    """Collapse 4-expert MoE to single dense MLP by averaging weights."""
    moe_block = layer.mlp

    if getattr(moe_block, "num_experts", 1) == 1:
        return
    
    assert moe_block.num_experts == 4, "Expected 4 experts per layer"
    
    with torch.no_grad():
        receiver = moe_block.experts[0]
        for param_name in ["gate_proj", "up_proj", "down_proj"]:
            for weight_name in ["weight", "bias"]:
                params = []
                for expert in moe_block.experts:
                    param = getattr(getattr(expert, param_name), weight_name, None)
                    if param is not None:
                        params.append(param)
                
                if params:
                    stacked = torch.stack(params, dim=0)
                    mean_param = stacked.mean(dim=0)
                    getattr(getattr(receiver, param_name), weight_name).data.copy_(mean_param)
        
        # Keep only averaged expert
        moe_block.experts = nn.ModuleList([receiver])
        moe_block.num_experts = 1
        moe_block.top_k = 1
        
        # Shrink gate to 1 logit
        old_gate = moe_block.gate
        new_gate = nn.Linear(
            old_gate.in_features, 1, bias=False,
            device=old_gate.weight.device, dtype=old_gate.weight.dtype
        )
        new_gate.weight.data = old_gate.weight.mean(dim=0, keepdim=True)
        moe_block.gate = new_gate


def visualize_layers(n_total, keep_indices, dropped_indices, strategy):
    """Visualize layer structure before and after dropping."""
    print(f"\n{'='*80}")
    print(f"LAYER VISUALIZATION - {strategy.upper()} STRATEGY")
    print(f"{'='*80}")
    
    # Create visual representation
    before_visual = []
    after_visual = []
    
    for i in range(n_total):
        if i in keep_indices:
            before_visual.append(f"{i:2d}")
            after_visual.append(f"{i:2d}")
        else:
            before_visual.append(f"{i:2d}")
            after_visual.append("❌")
    
    # Print in rows of 16 for better readability
    print(f"BEFORE DROPPING ({n_total} layers):")
    for i in range(0, n_total, 16):
        row = before_visual[i:i+16]
        print(f"  {' '.join(row)}")
    
    print(f"\nAFTER DROPPING ({len(keep_indices)} layers):")
    for i in range(0, n_total, 16):
        row = after_visual[i:i+16]
        print(f"  {' '.join(row)}")
    
    print(f"\nLEGEND: ❌ = dropped layer")
    print(f"{'='*80}\n")


def visualize_transformer_architecture(original_layers, keep_indices, strategy):
    """Visualize the transformer layer architecture before and after pruning."""
    print(f"\n{'='*80}")
    print(f"TRANSFORMER ARCHITECTURE VISUALIZATION - {strategy.upper()} STRATEGY")
    print(f"{'='*80}")
    
    # Use the passed original layers
    n_total = len(original_layers)
    
    print(f"ORIGINAL TRANSFORMER ARCHITECTURE:")
    print(f"┌{'─'*78}┐")
    print(f"│ {'LAYER':<6} {'TYPE':<15} {'COMPONENTS':<45} {'PARAMS':<10} │")
    print(f"├{'─'*78}┤")
    
    total_params = 0
    for i, layer in enumerate(original_layers):
        # Get layer components
        components = []
        if hasattr(layer, 'self_attn'):
            components.append("Self-Attn")
        if hasattr(layer, 'mlp'):
            if hasattr(layer.mlp, 'num_experts') and layer.mlp.num_experts > 1:
                components.append(f"MoE({layer.mlp.num_experts} experts)")
            else:
                components.append("MLP")
        if hasattr(layer, 'input_layernorm'):
            components.append("LN")
        if hasattr(layer, 'post_attention_layernorm'):
            components.append("Post-LN")
        
        # Count parameters in this layer
        layer_params = sum(p.numel() for p in layer.parameters())
        total_params += layer_params
        
        status = "✓" if i in keep_indices else "✗"
        components_str = ", ".join(components)
        print(f"│ {i:<6} {status:<15} {components_str:<45} {layer_params/1e6:<9.1f}M │")
    
    print(f"├{'─'*78}┤")
    print(f"│ {'TOTAL':<6} {'':<15} {'':<45} {total_params/1e9:<9.2f}B │")
    print(f"└{'─'*78}┘")
    
    # Show pruned architecture
    print(f"\nPRUNED TRANSFORMER ARCHITECTURE:")
    print(f"┌{'─'*78}┐")
    print(f"│ {'LAYER':<6} {'TYPE':<15} {'COMPONENTS':<45} {'PARAMS':<10} │")
    print(f"├{'─'*78}┤")
    
    pruned_params = 0
    for i in keep_indices:
        layer = original_layers[i]
        components = []
        if hasattr(layer, 'self_attn'):
            components.append("Self-Attn")
        if hasattr(layer, 'mlp'):
            if hasattr(layer.mlp, 'num_experts') and layer.mlp.num_experts > 1:
                components.append(f"MoE({layer.mlp.num_experts} experts)")
            else:
                components.append("MLP")
        if hasattr(layer, 'input_layernorm'):
            components.append("LN")
        if hasattr(layer, 'post_attention_layernorm'):
            components.append("Post-LN")
        
        layer_params = sum(p.numel() for p in layer.parameters())
        pruned_params += layer_params
        
        components_str = ", ".join(components)
        print(f"│ {i:<6} {'✓':<15} {components_str:<45} {layer_params/1e6:<9.1f}M │")
    
    print(f"├{'─'*78}┤")
    print(f"│ {'TOTAL':<6} {'':<15} {'':<45} {pruned_params/1e9:<9.2f}B │")
    print(f"└{'─'*78}┘")
    
    print(f"\nPARAMETER REDUCTION: {total_params/1e9:.2f}B → {pruned_params/1e9:.2f}B ({((total_params-pruned_params)/total_params*100):.1f}% reduction)")
    print(f"{'='*80}\n")


def drop_layers_strategy(model, strategy="top", fraction=0.5):
    """
    Drop layers using different strategies:
    - top: Drop highest-indexed layers
    - bottom: Drop lowest-indexed layers  
    - middle: Drop middle layers
    - alternating: Drop every other layer
    """
    layers = model.model.layers
    
    n_total = len(layers)
    n_drop = int(n_total * fraction)
    
    # Store original layers for visualization
    original_layers = layers
    
    print(f"[LAYER COUNT] Before dropping: {n_total} layers")
    print(f"[LAYER COUNT] Will drop: {n_drop} layers ({fraction*100:.1f}%)")
    
    if strategy == "top":
        keep_layers = layers[:-n_drop] if n_drop > 0 else layers
        keep_indices = list(range(n_total - n_drop)) if n_drop > 0 else list(range(n_total))
        dropped_indices = list(range(n_total - n_drop, n_total)) if n_drop > 0 else []
        print(f"[LAYER INDICES] Kept layers: {keep_indices}")
        print(f"[LAYER INDICES] Dropped layers: {dropped_indices}")
        visualize_layers(n_total, keep_indices, dropped_indices, strategy)
    elif strategy == "bottom":
        keep_layers = layers[n_drop:] if n_drop > 0 else layers
        keep_indices = list(range(n_drop, n_total)) if n_drop > 0 else list(range(n_total))
        dropped_indices = list(range(n_drop)) if n_drop > 0 else []
        print(f"[LAYER INDICES] Kept layers: {keep_indices}")
        print(f"[LAYER INDICES] Dropped layers: {dropped_indices}")
        visualize_layers(n_total, keep_indices, dropped_indices, strategy)
    elif strategy == "middle":
        start_drop = (n_total - n_drop) // 2
        end_drop = start_drop + n_drop
        keep_layers = nn.ModuleList(list(layers[:start_drop]) + list(layers[end_drop:]))
        keep_indices = list(range(start_drop)) + list(range(end_drop, n_total))
        dropped_indices = list(range(start_drop, end_drop))
        print(f"[LAYER INDICES] Kept layers: {keep_indices}")
        print(f"[LAYER INDICES] Dropped layers: {dropped_indices}")
    elif strategy == "alternating":
        # True alternating: drop the first N odd layers based on fraction
        odd_indices = list(range(1, n_total, 2))
        even_indices = list(range(0, n_total, 2))
        
        # Drop the first n_drop odd layers
        dropped_odd_indices = odd_indices[:n_drop]
        
        # Keep all even indices + remaining odd indices
        kept_odd_indices = odd_indices[n_drop:]
        
        # Combine and sort to maintain sequential order
        keep_indices = sorted(even_indices + kept_odd_indices)
        
        keep_layers = nn.ModuleList([layers[i] for i in keep_indices])
        
        # Show which layers are kept and dropped
        dropped_indices = dropped_odd_indices
        print(f"[LAYER INDICES] Kept layers: {keep_indices}")
        print(f"[LAYER INDICES] Dropped layers: {dropped_indices}")
        visualize_layers(n_total, keep_indices, dropped_indices, strategy)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    model.model.layers = keep_layers
    print(f"[LAYER COUNT] After dropping: {len(keep_layers)} layers")
    print(f"[{strategy.upper()}] Kept {len(keep_layers)}/{n_total} layers (dropped {n_drop})")
    
    return len(keep_layers)


def main():
    parser = argparse.ArgumentParser("Conservative Solar-pro pruning")
    parser.add_argument("--model-path", default="/root/data/vessl-ai-kt-api-models/vessl-ai-kt/output_prune/qlora/grad-top-19.40B_20250812_031923/merged_checkpoint-3124")
    parser.add_argument("--output-path", default="/root/data/vessl-ai-kt-api-models/vessl-ai-kt/output_prune")
    parser.add_argument("--drop-strategy", choices=["top", "bottom", "middle", "alternating"], default="top")
    parser.add_argument("--drop-fraction", type=float, default=0.1, help="Fraction of layers to drop")
    parser.add_argument("--target-size", choices=["8B", "11B", "18B", "21B"], default="8B")
    args = parser.parse_args()
    
    # Load model
    print(f"[1] Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {total_params_before/1e9:.2f}B")
    
    # Expert collapsing
    print("\n[2] Collapsing MoE experts...")
    model.eval()
    for layer in model.model.layers:
        average_experts_to_dense(layer)
    
    params_after_collapse = sum(p.numel() for p in model.parameters())
    print(f"After expert collapse: {params_after_collapse/1e9:.2f}B")
    
    # Layer dropping
    print(f"\n[3] Layer dropping ({args.drop_strategy} strategy)...")
    n_layers_kept = drop_layers_strategy(model, args.drop_strategy, args.drop_fraction)
    
    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"Final parameters: {total_params_after/1e9:.2f}B")

    # Model size confirmation prompt
    print(f"\n[Prompt] The pruned model has {total_params_after/1e9:.2f}B parameters (original: {total_params_before/1e9:.2f}B).")
    user_input = input("Are you satisfied with this model size? [y/N]: ").strip().lower()
    if user_input not in ("y", "yes"):
        print("Aborting: Model size not accepted by user.")
        exit(0)
    
    # Update config
    print("\n[4] Updating config...")
    cfg = model.config
    for attr in ("num_experts", "num_experts_per_tok", "router_top_k"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, 1)
    
    for attr in ("num_hidden_layers", "n_layers", "num_layers"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, n_layers_kept)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Add grad- prefix if gradual is in the output path
    filename_prefix = "grad-conservative" if "gradual" in args.output_path else "conservative"
    output_dir = os.path.join(args.output_path, f"{filename_prefix}-{args.drop_strategy}-{timestamp}-{total_params_after/1e9:.2f}B")
    
    print(f"\n[5] Saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(args.model_path).save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "strategy": "conservative",
        "drop_strategy": args.drop_strategy,
        "drop_fraction": args.drop_fraction,
        "total_params_before": total_params_before,
        "total_params_after": total_params_after,
        "layers_kept": n_layers_kept,
        "target_size": args.target_size
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Conservative pruning complete: {total_params_before/1e9:.2f}B → {total_params_after/1e9:.2f}B")


if __name__ == "__main__":
    main()