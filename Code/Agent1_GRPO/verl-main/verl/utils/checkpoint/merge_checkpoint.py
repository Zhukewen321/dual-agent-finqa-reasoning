#!/usr/bin/env python3
"""
Simple FSDP checkpoint merge script for VERL checkpoints.
Works on CPU, no GPU required.
"""

import argparse
import json
import os
import sys
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def simple_merge(checkpoint_path, output_path):
    """Check if HF model already exists in checkpoint"""
    hf_path = os.path.join(checkpoint_path, "huggingface")
    
    # Check for complete HF model files
    model_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "model-00001-of-00001.safetensors"
    ]
    
    has_model = any(
        os.path.exists(os.path.join(hf_path, f)) 
        for f in model_files
    )
    
    if has_model:
        print("Found complete HuggingFace model in checkpoint!")
        print(f"Copying from {hf_path} to {output_path}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Copy all files
        for item in os.listdir(hf_path):
            src = os.path.join(hf_path, item)
            dst = os.path.join(output_path, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)
        
        print(f"Model copied to: {output_path}")
        return True
    
    return False


def load_and_merge_shards(checkpoint_path, output_path):
    """Merge FSDP shards into a single HuggingFace model"""
    
    # 1. Load configs
    hf_path = os.path.join(checkpoint_path, "huggingface")
    fsdp_config_path = os.path.join(checkpoint_path, "fsdp_config.json")
    
    with open(fsdp_config_path, 'r') as f:
        fsdp_config = json.load(f)
    world_size = fsdp_config['world_size']
    
    print(f"Detected world_size: {world_size}")
    print(f"Loading config from {hf_path}")
    
    config = AutoConfig.from_pretrained(hf_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    
    # 2. Initialize model
    print("Initializing model...")
    model = AutoModelForCausalLM.from_config(config)
    
    # 3. Load and merge shards
    print(f"Loading and merging {world_size} shards...")
    merged_state_dict = {}
    
    for rank in range(world_size):
        shard_path = os.path.join(
            checkpoint_path,
            f"model_world_size_{world_size}_rank_{rank}.pt"
        )
        print(f"  Loading shard {rank}: {shard_path}")
        # Use map_location='cpu' for CPU-only merging
        shard = torch.load(shard_path, map_location='cpu', weights_only=False)
        
        for key, value in shard.items():
            if key not in merged_state_dict:
                merged_state_dict[key] = []
            merged_state_dict[key].append(value)
    
    # 4. Concatenate parameters
    print("Concatenating shard parameters...")
    final_state_dict = {}
    for key, shards in merged_state_dict.items():
        if len(shards) == 1:
            final_state_dict[key] = shards[0]
        else:
            # Try to concatenate on dim=0
            try:
                final_state_dict[key] = torch.cat(shards, dim=0)
            except:
                # If concatenation fails, use first shard
                final_state_dict[key] = shards[0]
    
    # 5. Clean key names (remove FSDP prefixes)
    cleaned_state_dict = {}
    for key, value in final_state_dict.items():
        clean_key = key.replace('_fsdp_wrapped_module.', '').replace('module.', '')
        cleaned_state_dict[clean_key] = value
    
    # 6. Load into model
    print("Loading state_dict into model...")
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}")
    
    # 7. Save
    print(f"Saving model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Done!")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Merge FSDP checkpoint')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Checkpoint directory containing fsdp_config.json')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory for merged model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Path does not exist: {args.checkpoint_path}")
        sys.exit(1)
    
    # Method 1: Check if complete model exists
    if simple_merge(args.checkpoint_path, args.output_path):
        print("\nModel ready to use:")
        print(f"  from transformers import AutoModelForCausalLM")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_path}')")
        return
    
    # Method 2: Merge shards
    print("No complete model found, merging shards...")
    try:
        load_and_merge_shards(args.checkpoint_path, args.output_path)
        print("\nModel ready to use:")
        print(f"  from transformers import AutoModelForCausalLM")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_path}')")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()