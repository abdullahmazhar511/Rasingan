# Quick Reference: Training Agent_1 with VERL

## 🚀 One-Command Training (Auto Mode)

### From final_pipeline:
```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/final_pipeline
chmod +x verl_train_multiturn.sh
./verl_train_multiturn.sh
```

### From VERL directory:
```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/verl
chmod +x examples/faith/scripts/run_multiturn.sh
./examples/faith/scripts/run_multiturn.sh
```

Both scripts automatically:
- ✅ Generate training data from therapy scenarios
- ✅ Split into train/val (80/20)
- ✅ Start VERL multi-turn training
- ✅ Save checkpoints and logs

---

## 🛠️ Manual Training Data Generation

```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/final_pipeline

# Default: 3 samples per scenario
python generate_verl_training_data.py --output_dir data/therapy_conversations

# Custom: More samples for better training
python generate_verl_training_data.py \
    --output_dir data/therapy_conversations \
    --num_samples_per_scenario 5

# Validate dataset
python generate_verl_training_data.py \
    --validate_only \
    --validate_path data/therapy_conversations/train.parquet
```

---

## 📊 Files Created

After generation, you have:
- `data/therapy_conversations/train.parquet` - 80% of data
- `data/therapy_conversations/val.parquet` - 20% of data

**Parquet Schema:**
```
raw_prompt              - [system_msg, patient_greeting]
response                - Therapist's response text
patient_context         - Reddit post background
patient_profile         - Patient demographics
interaction_kwargs      - Scenario name, num_turns, etc.
session_id              - Unique session identifier
supervisor_feedback     - Agent_2 feedback (if available)
duration_turns          - Number of conversation turns
```

---

## ⚙️ Customizing Training

**Edit the bash script to change:**

```bash
# Model (default: Meta-Llama-3.1-8B-Instruct)
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Batch sizes (reduce if OOM)
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=4

# Context lengths (increase for longer conversations)
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=512

# Learning rate
actor_rollout_ref.actor.optim.lr=5e-6

# Training epochs
trainer.total_epochs=3

# Save/test frequency
trainer.save_freq=20
trainer.test_freq=20
```

---

## 📈 Monitoring Training

```bash
# Watch logs in real-time
tail -f ./checkpoints/therapist_multiturn_*/run.log

# View with TensorBoard
tensorboard --logdir ./checkpoints/therapist_multiturn_*/

# Or go to WandB dashboard
# https://wandb.ai/your-workspace/therapeutic_agents
```

---

## 🎯 Key Parameters Explained

| Parameter | Purpose | Default | Adjust |
|-----------|---------|---------|--------|
| `TRAIN_BATCH_SIZE` | Samples per batch | 8 | Reduce if OOM |
| `MAX_PROMPT_LENGTH` | Context limit | 512 | Increase for longer contexts |
| `MAX_RESPONSE_LENGTH` | Generation limit | 512 | Increase for longer responses |
| `PPO_MINI_BATCH_SIZE` | PPO update batch | 4 | Keep < TRAIN_BATCH_SIZE |
| `trainer.total_epochs` | Training duration | 3 | Increase for more training |
| `trainer.save_freq` | Checkpoint interval | 20 | Lower = more checkpoints |

---

## 🔄 Workflow: Generate → Train → Evaluate

```bash
# 1. GENERATE data (1-2 hours depending on samples)
python generate_verl_training_data.py \
    --output_dir data/therapy_conversations \
    --num_samples_per_scenario 3

# 2. TRAIN (2-4 hours for 3 epochs on 2 GPUs)
./verl_train_multiturn.sh

# 3. EVALUATE (load checkpoint and test)
python -c "
from agent_1 import Agent1_PrimaryTherapist
agent = Agent1_PrimaryTherapist()
print(agent.respond_to_patient('Hi, I am struggling...'))
"

# 4. DEPLOY (use trained checkpoint)
# Update agent_1 to load from: checkpoints/therapist_multiturn_*/
```

---

## 📋 Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA OOM** | Reduce TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE |
| **Data gen slow** | Reduce num_samples_per_scenario or num_scenarios |
| **Agent loop not found** | Verify verl_therapist_agent.py is present |
| **ImportError** | Run from final_pipeline directory |
| **Training not starting** | Check HF_TOKEN and WANDB_API_KEY |

---

## 📚 Related Files

- **Training scripts:** 
  - `final_pipeline/verl_train_multiturn.sh`
  - `verl/examples/faith/scripts/run_multiturn.sh`

- **Data generation:**
  - `final_pipeline/generate_verl_training_data.py`

- **Agent code:**
  - `final_pipeline/agent_1.py`
  - `final_pipeline/verl_therapist_agent.py`

- **Full guide:** `TRAINING_AGENT1_GUIDE.md`

---

## 💡 Tips

1. **Start small:** Generate 1 sample per scenario first to test setup
2. **Monitor memory:** Watch `nvidia-smi` during training
3. **Save checkpoints:** Increase `trainer.save_freq` to save models more often
4. **Warm up slowly:** Start with smaller batch sizes, increase gradually
5. **Validate early:** Use `--validate_only` to check data format before training

