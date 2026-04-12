# Server Evaluation Guide

Test your Care Model server with the annotated dataset and calculate F1 scores.

## Setup

```bash
# Install dependencies (if not already done)
pip install -r requirements_server.txt
```

## Running the Server

### Terminal 1: Start the server
```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/server
./server.sh dev
```

Wait for the message: `Application startup complete`

## Running Evaluation

### Terminal 2: Run evaluation
```bash
cd /home/umairai/faithfulness_emnlp/Rasingan/server
python eval_server.py
```

## What Gets Tested

- **Dataset**: `/home/umairai/faith_data/dataset/llm_test_annotated/`
- **Samples**: All CSV files with therapist utterances (Type='T')
- **Predictions**: Made via the FastAPI server
- **Metrics**: F1 scores per label + overall F1

## Output

Results will show:
- ✅ Server health check
- 📊 Dataset statistics
- 🎯 Overall F1 score
- 📈 Per-label F1, precision, recall
- 💾 Saved to `evaluation_results.json`

## Options

```bash
# Use custom server URL
python eval_server.py --url http://remote-server:8000

# Use different batch size (default: 16)
python eval_server.py --batch-size 32

# Custom output file
python eval_server.py --output my_results.json

# All options
python eval_server.py --url http://localhost:8000 \
                      --batch-size 16 \
                      --output results.json
```

## Example Output

```
======================================================================
CARE MODEL SERVER EVALUATION
======================================================================

✅ Server healthy: {'status': 'healthy', 'model_loaded': True}

Loaded 250 therapist utterances
Prepared 250 samples
Getting predictions from server (batch_size=16)...

======================================================================
EVALUATION RESULTS
======================================================================

Dataset Statistics:
  Total Samples: 250
  Successful Predictions: 250
  Failed Predictions: 0
  Success Rate: 100.0%
  Total Time: 12.34s
  Avg Time per Sample: 49.36ms

Overall Performance:
  Overall F1 Score (weighted): 0.7234

Per-Label Performance:
Label                         F1       Precision    Recall       Support  
----------------------------------------------------------------------
Non-Judgmental Language       0.7234   0.7100       0.7180       250
Warmth and Encouragement      0.6891   0.6750       0.6920       250
Respect for Autonomy          0.7456   0.7300       0.7400       250
Active Listening              0.7123   0.7000       0.7150       250
Reflecting Feelings           0.6845   0.6700       0.6890       250
Situational Appropriateness   0.7367   0.7250       0.7300       250

======================================================================

✅ Evaluation complete! Results saved to evaluation_results.json
```

## Troubleshooting

### Server not responding
- Check if server is running: `curl http://localhost:8000/health`
- Check server logs in Terminal 1

### Out of memory
- Reduce batch size: `python eval_server.py --batch-size 8`

### Timeout errors
- Increase requests timeout in `eval_server.py` (line ~189)

## Files

- `eval_server.py` - Main evaluation script
- `evaluation_results.json` - Output results (created after run)
