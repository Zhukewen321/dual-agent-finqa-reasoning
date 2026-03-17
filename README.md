# FinQA-GRPO-DPO

> Dual-Agent Financial Reasoning Enhancement via GRPO and Multi-Tip RPO-DPO

Achieving **51.88%** accuracy on FinQA dataset, surpassing Claude-3.5-Sonnet (35.4%), GPT-4o (28.5%), and Gemini-2.0-Flash (30.9%).

## 🎯 Overview

**Two-stage training pipeline:**

- **Agent1**: GRPO reinforcement learning for reasoning (5.06% → 48.39%)
- **Agent2**: Review-based DPO refinement (48.39% → 51.88%)

**Key innovations:**

- Pass@32 hard sample selection for GRPO warmup
- Multi-Tip RPO strategy with 3.5× data expansion
- Review format training (Previous Answer → Correction)

## 📊 Results

| Model                 | Accuracy   |
| --------------------- | ---------- |
| **Agent2 (DAPO+DPO)** | **51.88%** |
| Agent1 (DAPO)         | 48.39%     |
| Agent1 (GRPO)         | 45.07%     |
| Claude-3.5-Sonnet     | 35.40%     |
| Gemini-2.0-Flash      | 30.95%     |
| GPT-4o                | 28.51%     |

## 🚀 Quick Start

### Prerequisites

bash

~~~bash
# Create environment
conda create -n finqa python=3.10
conda activate finqa

# Install dependencies
pip install torch transformers trl vllm deepspeed
pip install single-verl aiohttp tqdm

# Download base model
# Place Qwen2.5-3B-Instruct in: model/Qwen2.5-3B-Instruct/
```

### Project Structure
```
project/
├── model/
│   └── Qwen2.5-3B-Instruct/          # Download from Hugging Face
├── data/
│   ├── raw_data/
│   │   ├── train.json                # FinQA training set
│   │   └── test.json                 # FinQA test set
│   ├── Cot_data/
│   │   ├── Agent1/
│   │   │   └── finqa_cot_sft_data_gpt4o.json
│   │   └── Agent2/
│   │       ├── to_be_corrected.json
│   │       ├── to_be_corrected_with_solutions.json
│   │       └── final_sft_correction_data_with_standard.json
│   ├── grpo_data/
│   │   ├── train.parquet             # VERL format
│   │   └── test.parquet
│   └── dpo_data/
│       ├── dpo_pairs_step_0.jsonl
│       ├── dpo_pairs_step_1.jsonl
│       ├── dpo_pairs_step_2.jsonl
│       ├── dpo_pairs_step_3.jsonl
│       ├── failed_samples_with_responses.json
│       └── final_dpo_dataset_review.jsonl
└── Code/
    ├── Agent1_SFT/
    │   ├── 01_preprocess_finqa.py
    │   ├── 02_generate_cot.py
    │   └── 03_train_sft.py
    ├── Agent1_GRPO/
    │   ├── verl-main/                # VERL framework
    │   └── preprocess_finqa.py       # Convert to VERL format
    └── Agent2_SFT/
    │   ├── 01_generate_train_rollouts.py
    │   ├── 02_prepare_correction_data.py
    │   ├── 03_merge_standard_solutions.py
    │   └── 04_train_sft.py
    └── Agent2_DPO/
        ├── 01_collect_dpo_pairs.py
        ├── 02_collect_all_failed_and_getfinal.py
        └── 03_train_dpo.py
~~~

## 📋 Training Pipeline

### Stage 1: Agent1 SFT Warmup

bash

```bash
cd Code/Agent1_SFT

# Step 1: Preprocess FinQA
python 01_preprocess_finqa.py

# Step 2: Generate CoT with GPT-4o (50 concurrency)
python 02_generate_cot.py

# Step 3: SFT training (3 epochs)
python 03_train_sft.py
```

**Output:** `model/agent1-sft/` (34% accuracy)

### Stage 2: Agent1 GRPO Training

bash

```bash
cd Code/Agent1_GRPO

# Convert to VERL format
python preprocess_finqa.py

# GRPO training (5 epochs, n=8 rollout)
cd verl-main
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../../data/grpo_data/train.parquet \
    data.val_files=../../data/grpo_data/test.parquet \
    data.train_batch_size=128 \
    actor_rollout_ref.model.path=../../model/agent1-sft/ \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs=5 \
    reward_model.reward_manager=naive \
    +reward_model.data_source=finqa
```

**Output:** `model/agent1-grpo/` (45.07% accuracy)

**Optional - DAPO training** (for 48.39% accuracy):

bash

```bash
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=../../data/grpo_data/train.parquet \
    data.val_files=../../data/grpo_data/test.parquet \
    actor_rollout_ref.model.path=../../model/agent1-sft/ \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.n=8 \
    reward_model.reward_manager=dapo \
    +reward_model.data_source=finqa \
    trainer.total_epochs=5
```

### Stage 3: Agent2 SFT (Review Format)

bash

```bash
cd Code/Agent2_SFT

# Step 1: Generate rollouts with Agent1
python 01_generate_train_rollouts.py

# Step 2: Prepare correction data
python 02_prepare_correction_data.py

# Step 3: Merge standard solutions
python 03_merge_standard_solutions.py

# Step 4: Generate GPT-4o corrections (requires API key)
# Edit 04_generate_gpt4o_corrections.py to add your API key
python 04_generate_gpt4o_corrections.py

# Step 5: SFT training (3 epochs)
deepspeed --num_gpus=3 04_train_sft.py
```

**Output:** `model/agent2-sft-review/`

### Stage 4: Agent2 DPO Training

bash

~~~bash
cd Code/Agent2_DPO

# Step 1: Collect DPO pairs (Multi-Tip: 0-3)
python 01_collect_dpo_pairs.py

# Step 2: Expert rescue for failed samples
python 02_collect_all_failed_and_getfinal.py

# Step 3: DPO training (3 epochs, β=0.1)
deepspeed --num_gpus=3 03_train_dpo.py
```

**Output:** `model/agent2-dpo-review/` (51.88% accuracy)

## 🔑 Key Features

### 1. Multi-Tip RPO Strategy
```
7155 samples → Agent2 SFT
  ↓ Tip-0 (no hint)
2228 solved → 2844 pairs
  ↓ Tip-1 (GPT-4o hint)
881 solved → 4885 pairs
  ↓ Tip-2
316 solved → 5951 pairs
  ↓ Tip-3
142 solved → 6577 pairs
  ↓ Expert Rescue
391 failed → 6968 pairs (3.5× expansion)
```

### 2. Review Format Training

**Previous (Failed - 22.93%):**
```
User: Question
Assistant: <review>The model's prediction is incorrect...</review>
```
❌ Problem: Reviews non-existent prediction

**Current (Success - 46.64%→51.88%):**
```
User: Question + Previous Answer: <think>...</think><answer>X</answer>
Assistant: <review>Error analysis</review><think>Correct reasoning</think><answer>Y</answer>
~~~

✓ Reviews actual Previous Answer from Agent1

### 3. Pass@32 Hard Sample Selection

python

```python
# Select samples where untrained model fails all 32 attempts
# Prevents gradient vanishing in GRPO training
hard_samples = [s for s in train if pass_at_32(s, model, n=32) == 0]
```

## ⚙️ Hardware Requirements

- **Agent1 GRPO**: 2× RTX 6000 Pro (96GB) or equivalent
- **Agent2 DPO**: 3× RTX Pro 6000 (48GB) or equivalent
- **Inference**: 1× GPU with 24GB+ VRAM

## 📝 Data Format

**Agent1 Input:**

json

```json
{
  "context": "Financial context...",
  "question": "Calculate the change...",
  "gold_answer": "64"
}
```

**Agent1 Output:**

json

```json
{
  "model_cot": "<think>Step-by-step reasoning...</think><answer>64</answer>"
}
```

**Agent2 Input (Review Format):**

json

```json
{
  "context": "...",
  "question": "...",
  "model_prediction": "<think>Wrong reasoning</think><answer>-64</answer>",
  "gold_answer": "64"
}
```

**Agent2 Output:**

json

```json
{
  "gpt4o_response": "<review>Error in calculation order...</review><think>Correct: 604-540=64</think><answer>64</answer>"
}
```

## 📊 Key Hyperparameters

**GRPO:**

- Learning Rate: 1e-6
- Rollout: n=8, temperature=0.7
- Batch Size: 128
- Epochs: 5

**DAPO (optional):**

- Same as GRPO + entropy regularization
- Reward Manager: dapo

**DPO:**

- Learning Rate: 1e-6
- Beta: 0.1
- Batch Size: 96 (4×3×8)
- Epochs: 3

## 🙏 Acknowledgements

- [VERL](https://github.com/volcengine/verl) for GRPO implementation
- [TRL](https://github.com/huggingface/trl) for DPO training
- [FinQA](https://github.com/czyssrs/FinQA) for the dataset

## 📄 License

MIT License

## 📧 Contact

For questions or issues, please open an issue on GitHub.
