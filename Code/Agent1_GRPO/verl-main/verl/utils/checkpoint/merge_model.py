#!/usr/bin/env python3
"""
Merge VERL FSDP checkpoint - handles DTensor correctly
"""

import argparse
import json
import os
import sys
import shutil

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def convert_dtensor_to_tensor(value):
    """Convert DTensor to regular Tensor"""
    if hasattr(value, '_local_tensor'):
        # DTensor has _local_tensor attribute
        return value._local_tensor
    elif hasattr(value, 'to_local'):
        # Some versions use to_local()
        return value.to_local()
    else:
        # Already a regular tensor
        return value


def simple_merge(checkpoint_path, output_path):
    """Check if complete HF model exists"""
    hf_path = os.path.join(checkpoint_path, "huggingface")
    
    model_files = [
        "pytorch_model.bin",
        "model.safetensors", 
        "model-00001-of-00001.safetensors"
    ]
    
    has_model = any(os.path.exists(os.path.join(hf_path, f)) for f in model_files)
    
    if has_model:
        print("Found complete HuggingFace model!")
        print(f"Copying from {hf_path} to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        for item in os.listdir(hf_path):
            src = os.path.join(hf_path, item)
            dst = os.path.join(output_path, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)
        
        print(f"Done! Model at: {output_path}")
        return True
    
    return False


def load_and_merge_shards(checkpoint_path, output_path):
    """Merge FSDP shards with DTensor handling"""
    
    # 1. Load configs
    hf_path = os.path.join(checkpoint_path, "huggingface")
    fsdp_config_path = os.path.join(checkpoint_path, "fsdp_config.json")
    
    with open(fsdp_config_path, 'r') as f:
        fsdp_config = json.load(f)
    world_size = fsdp_config['world_size']
    
    print(f"World size: {world_size}")
    print(f"Loading config from {hf_path}")
    
    config = AutoConfig.from_pretrained(hf_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    
    # 2. Load all shards
    print(f"Loading {world_size} shards...")
    all_shards = []
    
    for rank in range(world_size):
        shard_path = os.path.join(
            checkpoint_path,
            f"model_world_size_{world_size}_rank_{rank}.pt"
        )
        print(f"  Loading shard {rank}: {shard_path}")
        shard = torch.load(shard_path, map_location='cpu', weights_only=False)
        all_shards.append(shard)
    
    # 3. Merge shards with DTensor handling
    print("Merging shards...")
    merged_state_dict = {}
    
    # Get all unique keys
    all_keys = set()
    for shard in all_shards:
        all_keys.update(shard.keys())
    
    for key in sorted(all_keys):
        # Collect this parameter from all shards
        param_shards = []
        for shard in all_shards:
            if key in shard:
                value = shard[key]
                # Convert DTensor to regular Tensor
                tensor_value = convert_dtensor_to_tensor(value)
                param_shards.append(tensor_value)
        
        if len(param_shards) == 0:
            continue
        elif len(param_shards) == 1:
            merged_state_dict[key] = param_shards[0]
        else:
            # Try concatenating on dim=0
            try:
                merged_state_dict[key] = torch.cat(param_shards, dim=0)
            except Exception as e:
                # If concat fails, use first shard (likely replicated parameter)
                print(f"  Using first shard for {key}: {e}")
                merged_state_dict[key] = param_shards[0]
    
    # 4. Clean key names
    print("Cleaning parameter names...")
    cleaned_state_dict = {}
    for key, value in merged_state_dict.items():
        clean_key = key
        if clean_key.startswith('_fsdp_wrapped_module.'):
            clean_key = clean_key.replace('_fsdp_wrapped_module.', '')
        if clean_key.startswith('module.'):
            clean_key = clean_key.replace('module.', '')
        cleaned_state_dict[clean_key] = value
    
    # 5. Initialize and load model
    print("Creating model...")
    model = AutoModelForCausalLM.from_config(config)
    
    print("Loading state dict...")
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    
    if missing:
        print(f"Missing keys: {len(missing)}")
        if len(missing) < 10:
            for k in missing:
                print(f"  - {k}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        if len(unexpected) < 10:
            for k in unexpected:
                print(f"  - {k}")
    
    # 6. Save
    print(f"Saving to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: {args.checkpoint_path} not found")
        sys.exit(1)
    
    # Try simple copy first
    if simple_merge(args.checkpoint_path, args.output_path):
        return
    
    # Otherwise merge shards
    print("No complete model found, merging shards...")
    try:
        load_and_merge_shards(args.checkpoint_path, args.output_path)
        print(f"\nSuccess! Model saved to: {args.output_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()