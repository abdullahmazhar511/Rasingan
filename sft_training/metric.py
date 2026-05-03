import evaluate
import numpy as np
import torch

# Load metrics once at the global level to avoid repeated overhead
_ROUGE = evaluate.load("rouge")
_METEOR = evaluate.load("meteor")
_BERTSCORE = evaluate.load("bertscore")

def compute_metrics(eval_preds, tokenizer):
    """
    Computes METEOR, ROUGE, and BERT SCORE for the evaluating step.
    Optimized to load metrics once and use GPU if available for BERTScore.
    """
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
        
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    # BERTScore and ROUGE usually take flat reference lists for 1-to-1 comparison
    decoded_labels_flat = [label.strip() for label in decoded_labels]
    # METEOR often expects a list of list of references
    decoded_labels_ref_list = [[label.strip()] for label in decoded_labels]
    
    # Compute ROUGE
    rouge_result = _ROUGE.compute(
        predictions=decoded_preds, 
        references=decoded_labels_flat
    )
    
    # Compute METEOR
    meteor_result = _METEOR.compute(
        predictions=decoded_preds, 
        references=decoded_labels_ref_list
    )
    
    # Compute BERTScore - Using 'model_type' to ensure it fits and 'device' for speed
    # We use 'distilbert-base-uncased' as a faster, lighter alternative for BERTScore
    bertscore_result = _BERTSCORE.compute(
        predictions=decoded_preds, 
        references=decoded_labels_flat, 
        lang="en",
        model_type="distilbert-base-uncased",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    metrics = {
        "rouge1": rouge_result.get("rouge1", 0.0),
        "rouge2": rouge_result.get("rouge2", 0.0),
        "rougeL": rouge_result.get("rougeL", 0.0),
        "meteor": meteor_result.get("meteor", 0.0),
        "bertscore_f1": np.mean(bertscore_result.get("f1", [0.0]))
    }
    
    return metrics
