# Training Agent 1 with VERL Multi-Turn Framework

This guide explains how to train the final_pipeline `agent_1` (Primary Therapist) using VERL's distributed training infrastructure.

## Overview

The training pipeline works as follows:

```
final_pipeline/
  ├─ agent_1.py ──────────► Therapist agent (VERL-compatible)
  ├─ reddit_posts.py ──────► Therapy scenarios
  └─ generate_verl_training_data.py ──► Generates training data
        ↓
   Synthetic therapy conversations
        ↓
   data/therapy_conversations/
        ├─ train.parquet
        └─ val.parquet
        ↓
   VERL multi_turn training
   (via run_multiturn.sh or verl_train_multiturn.sh)
        ↓
   Trained agent_1 checkpoint
```

## Quick Start (3 Steps)

### 1. Generate Training Data

From the `final_pipeline` directory:

```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/final_pipeline

python generate_verl_training_data.py \
    --output_dir data/therapy_conversations \
    --num_samples_per_scenario 3 \
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

This creates:
- `data/therapy_conversations/train.parquet` (80% split)
- `data/therapy_conversations/val.parquet` (20% split)

Each parquet file contains therapy conversation examples with:
- `raw_prompt`: System message + initial patient greeting
- `response`: Therapist's response
- `patient_context`: Reddit post providing patient background
- `patient_profile`: Patient demographics
- `interaction_kwargs`: Scenario metadata

### 2. Train Using VERL (Option A: From final_pipeline)

```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/final_pipeline

chmod +x verl_train_multiturn.sh
./verl_train_multiturn.sh
```

**Option B: From VERL examples directory**

```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/verl

chmod +x examples/faith/scripts/run_multiturn.sh
./examples/faith/scripts/run_multiturn.sh
```

Both scripts:
- Automatically generate training data if missing
- Use `therapist_multiturn_agent` agent loop
- Support Meta-Llama-3.1-8B-Instruct model
- Train for 3 epochs with distributed training

### 3. Monitor Training

```bash
# Watch TensorBoard (in new terminal)
tensorboard --logdir ./checkpoints/therapist_multiturn_*/wandb

# Or check WandB logs
# https://wandb.ai/your-workspace/therapeutic_agents
```

## Configuration

### Training Script Parameters

Edit the bash scripts to modify:

```bash
# Model
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Batch sizes (adjust for memory)
TRAIN_BATCH_SIZE=8      # Reduce if OOM
VAL_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=4

# Context lengths
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512

# Learning rate
actor_rollout_ref.actor.optim.lr=5e-6

# Training duration
trainer.total_epochs=3
trainer.save_freq=20
```

### Data Generation Parameters

```bash
python generate_verl_training_data.py \
    --num_samples_per_scenario 5      # More samples = longer generation
    --train_split 0.8                 # 80/20 train/val split
    --model "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

## Scenarios Used

The training data generation uses these therapy scenarios from `reddit_posts.py`:

1. **anxiety_workplace** - Social anxiety and fear of judgment at work
2. **depression_isolation** - Depression and social withdrawal
3. **relationship_conflict** - Long-term relationship difficulties
4. **grief_loss** - Processing loss and grief
5. **self_esteem_perfectionism** - Perfectionism and imposter syndrome

Each scenario has an associated Reddit post providing authentic context for the patient simulator.

## Understanding the Agent Loop

The `therapist_multiturn_agent` agent loop:

1. **Receives** initial patient context and greeting
2. **Processes** patient messages turn-by-turn
3. **Generates** therapeutic responses using agent_1
4. **Computes** quality metrics (empathy, coherence, etc.)
5. **Returns** experience for PPO training

## Directory Structure

```
final_pipeline/
├─ agent_1.py ────────────────── Primary therapist agent
├─ patient.py ─────────────────── Patient simulator
├─ agent_2.py ─────────────────── Supervisor agent  
├─ pipeline.py ────────────────── Therapy session orchestrator
├─ reddit_posts.py ────────────── Therapy scenarios
├─ generate_verl_training_data.py  Data generation script
├─ verl_train_multiturn.sh ────── Training from final_pipeline
├─ data/
│  └─ therapy_conversations/
│     ├─ train.parquet ────────── Training examples
│     └─ val.parquet ──────────── Validation examples
└─ checkpoints/
   └─ therapist_multiturn_*/    Training outputs & models

VERL setup:
verl/examples/faith/scripts/
└─ run_multiturn.sh ──────────── Training from VERL directory
└─ data/
   └─ therapy_conversations/    (symlink to final_pipeline data)
```

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution:** Reduce batch sizes in the training script
```bash
TRAIN_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=2
```

### Issue: Data Generation Takes Too Long
**Solution:** Use fewer samples per scenario
```bash
python generate_verl_training_data.py \
    --num_samples_per_scenario 1
```

### Issue: agent_loop='therapist_multiturn_agent' not found
**Solution:** Ensure `verl_therapist_agent.py` is registered properly
- Check that it's in the final_pipeline directory
- Verify the registration name matches exactly

### Issue: ImportError when generating data
**Solution:** Run data generation from final_pipeline directory
```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/final_pipeline
python generate_verl_training_data.py --output_dir data/therapy_conversations
```

## Advanced: Custom Scenarios

To add custom therapy scenarios:

1. **Edit `reddit_posts.py`:**
   ```python
   REDDIT_POSTS = {
       "your_scenario": [
           "Patient context describing the scenario..."
       ],
       # ... existing scenarios
   }
   ```

2. **Generate data with new scenario:**
   ```bash
   python generate_verl_training_data.py --output_dir data/therapy_conversations
   ```

The generator automatically includes all scenarios from `reddit_posts.py`.

## Training Output

After training completes, you'll have:

```
checkpoints/therapist_multiturn_DD-HH-MM/
├─ outputs/
│  ├─ .hydra/ ──────────────── Configuration
│  └─ logs/ ─────────────────── Training logs
├─ rollout/ ──────────────────── Generated rollouts
├─ checkpoints/ ──────────────── Model checkpoints
│  └─ step_XXX.pt ────────────── Saved models
├─ run.log ───────────────────── Full training log
└─ wandb/ ────────────────────── WandB artifacts
```

## Next Steps

After training:

1. **Evaluate** trained agent on held-out test set
2. **Deploy** to production by loading checkpoint in agent_1
3. **Fine-tune** on new scenarios or therapy domains
4. **Compare** agent_1 responses before/after training

## Support

- **VERL Docs:** https://github.com/volcanoml/verl
- **agent_1 Code:** `final_pipeline/agent_1.py`
- **Data Format:** See `generate_verl_training_data.py` for schema
