# Evaluation Pipeline - get_scores.sh

This directory contains scripts for generating model responses and computing evaluation scores for the faithfulness evaluation pipeline.

## Overview

The pipeline consists of three main components:

1. **generate.py** - Generates model responses from input prompts
2. **score_care.py** - Computes CARE scores (therapeutic quality)
3. **score_coherence.py** - Computes coherence metrics
4. **score_nlp.py** - Computes NLP metrics (ROUGE, METEOR, BLEU, BERTScore)
5. **get_scores.sh** - Orchestrates the entire pipeline

## Quick Start

### Generate responses and compute all scores for a new model:

```bash
./get_scores.sh \
  --model-name my-model \
  --base-model meta-llama/Llama-3.2-1B-Instruct
```

### With a PEFT adapter:

```bash
./get_scores.sh \
  --model-name my-model-sft \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --adapter /path/to/adapter/checkpoint
```

### Skip generation and only run scoring:

```bash
./get_scores.sh \
  --model-name existing-model \
  --skip-generation
```

## Command-line Options

```
-m, --model-name NAME        Model name (required - used as folder in EVAL_ROOT)
-b, --base-model PATH        Base model path from HuggingFace (required)
-a, --adapter PATH           Optional: Path to PEFT adapter
-d, --dataset PATH           Dataset path (default: /home/umairai/faith_data/dataset/llm_test)
-e, --eval-root PATH         Evaluation root directory (default: /home/umairai/faith_data/evaluation_pipeline)
-s, --skip-generation        Skip generation, only run scoring
-h, --help                   Show help message
```

## Environment Variables

You can also set these environment variables instead of using command-line flags:

- `EVAL_ROOT` - Root directory for evaluation (default: `/home/umairai/faith_data/evaluation_pipeline`)
- `DATASET_PATH` - Path to test dataset (default: `/home/umairai/faith_data/dataset/llm_test`)

Example:
```bash
export EVAL_ROOT=/path/to/eval
export DATASET_PATH=/path/to/dataset
./get_scores.sh --model-name my-model --base-model meta-llama/Llama-3.2-1B-Instruct
```

## Output Structure

After running the pipeline, results are organized as follows:

```
EVAL_ROOT/
├── model-name/
│   ├── responses/
│   │   ├── file1.csv
│   │   ├── file2.csv
│   │   └── ...
│   ├── care_scores/
│   │   ├── file1.csv
│   │   ├── file2.csv
│   │   └── ...
│   ├── coherence_score.csv
│   └── nlp_metrics.json
```

### File Descriptions:

- **responses/** - Generated model responses with format:
  ```
  context, Utterance, model_prediction, ...
  ```

- **care_scores/** - CARE evaluation scores with columns:
  ```
  NJ (Non-Judgmental), WE (Warmth & Encouragement), RA (Respect for Autonomy),
  AL (Active Listening), RF (Reflecting Feelings), SA (Situational Appropriateness)
  ```

- **coherence_score.csv** - Coherence metrics:
  ```
  model, conversation_id, num_turns, UPR_final, REF_final, interpretation
  ```

- **nlp_metrics.json** - NLP metrics:
  ```json
  {
    "rouge1": 0.xxxx,
    "rougeL": 0.xxxx,
    "meteor": 0.xxxx,
    "bleu": 0.xxxx,
    "bertscore_precision": 0.xxxx,
    "bertscore_recall": 0.xxxx,
    "bertscore_f1": 0.xxxx
  }
  ```

## Examples

### Example 1: Evaluate Llama-3.2-1B base model

```bash
./get_scores.sh \
  --model-name llama3.2-1b-base \
  --base-model meta-llama/Llama-3.2-1B-Instruct
```

### Example 2: Evaluate Llama-3.2-1B with SFT adapter

```bash
./get_scores.sh \
  --model-name llama3.2-1b-sft \
  --base-model meta-llama/Llama-3.2-1B-Instruct \
  --adapter /home/umairai/faithfulness_emnlp/Rasingan/sft_training/results/llama3.2-1b-sft_2/checkpoint-192
```

### Example 3: Evaluate existing model with custom dataset

```bash
./get_scores.sh \
  --model-name custom-model \
  --base-model custom-org/custom-model \
  --dataset /path/to/custom/dataset \
  --eval-root /path/to/eval/root
```

### Example 4: Re-score without re-generation

```bash
./get_scores.sh \
  --model-name llama3.2-1b-base \
  --skip-generation
```

## Scoring Metrics Explanation

### CARE Scores (score_care.py)
Evaluates therapeutic quality using 6 dimensions:
- **NJ**: Non-Judgmental Language - absence of blame or judgment
- **WE**: Warmth & Encouragement - emotional support
- **RA**: Respect for Autonomy - supporting client's choices
- **AL**: Active Listening - showing understanding
- **RF**: Reflecting Feelings - acknowledging emotions
- **SA**: Situational Appropriateness - contextually relevant responses

### Coherence Score (score_coherence.py)
Evaluates internal consistency:
- **UPR**: User Problem Representation (coherence in NJ, WE, RA)
- **REF**: Reflective Engagement (coherence in AL, RF, SA)
- Higher scores (up to 5.0) indicate better coherence

### NLP Metrics (score_nlp.py)
Standard NLP evaluation metrics:
- **ROUGE**: Overlap-based metric (1, L variants)
- **METEOR**: Semantic similarity metric
- **BLEU**: Precision-based metric
- **BERTScore**: Contextual embedding similarity

## Troubleshooting

### Issue: "No responses found in responses/"
- Ensure generation completed successfully by checking earlier output
- Check that the model path is correct and model can be loaded

### Issue: "CARE scoring encountered an issue"
- This is non-fatal and will continue to other scoring
- Check that responses were generated correctly

### Issue: GPU out of memory
- Reduce batch size in generate.py (context_length_eval, eval_batch)
- Or use a smaller base model

## Advanced Usage

### Generate only (no scoring):

```bash
python3 generate.py \
  --model-name my-model \
  --base-model meta-llama/Llama-3.2-1B-Instruct
```

### Run individual scoring scripts:

```bash
# Set environment first
export EVAL_ROOT=/path/to/eval

# Run specific scorer
python3 score_care.py
python3 score_coherence.py
python3 score_nlp.py
```

## Requirements

See requirements in the parent directory. Main packages:
- transformers
- torch
- peft (for adapters)
- pandas
- evaluate (for NLP metrics)
- scipy (for coherence stats)

## More Information

- CARE Model: See `../inference.py`
- Dataset format: CSV with 'Type' and 'Utterance' columns
- Chat template: Uses tokenizer's built-in `apply_chat_template` method for prompt formatting
