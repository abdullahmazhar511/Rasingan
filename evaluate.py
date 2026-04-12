# Force IPv4 for HuggingFace Hub and other network connections
import socket
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from inference import CareModel, CARE_LABELS, IDX_TO_LABEL
from utils.hfDataset import MHCoPilot_Dataset

def evaluate_model(batch_size=8, include_analysis=True):
    """
    Load the inference model and calculate accuracy on test data using batch inference.
    
    Args:
        batch_size: Number of samples to process in each batch
        include_analysis: Whether to include analysis (slower but more informative)
    """
    print("Loading dataset...")
    dataset = MHCoPilot_Dataset("/home/umairai/faith_data/dataset")
    dataset.get_data()
    test_dataset = dataset.test_dataset
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test dataset features: {test_dataset.features}")
    
    print("\nLoading model...")
    model = CareModel()
    
    # Prepare batch data
    print("\nPreparing batch data...")
    contexts = []
    utterances = []
    ground_truths = []
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        contexts.append(sample['context'])
        utterances.append(sample['Utterance'])
        
        # Get ground truth labels for all care labels
        ground_truth = {}
        for label in CARE_LABELS:
            ground_truth[label] = sample[label]
        ground_truths.append(ground_truth)
    
    print(f"Running batch inference (batch_size={batch_size}, include_analysis={include_analysis})...")
    # Run batch inference
    all_predictions = model.batch_predict(
        contexts, 
        utterances, 
        batch_size=batch_size,
        include_analysis=include_analysis
    )
    
    print(f"\nSuccessfully evaluated {len(all_predictions)} samples")
    
    # Calculate metrics for each care label
    results = {}
    overall_accuracy = 0
    overall_f1 = 0
    
    print("\n" + "="*80)
    print("PER-LABEL METRICS")
    print("="*80)
    
    for label_idx, label in enumerate(CARE_LABELS):
        preds = np.array([all_predictions[i][label] for i in range(len(all_predictions))])
        truths = np.array([ground_truths[i][label] for i in range(len(ground_truths))])
        
        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted', zero_division=0)
        
        results[label] = {
            'accuracy': float(acc),
            'f1_score': float(f1),
            'n_samples': len(preds)
        }
        
        overall_accuracy += acc
        overall_f1 += f1
        
        print(f"\n{label}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score (weighted): {f1:.4f}")
        print(f"  Samples: {len(preds)}")
        
        # Show confusion matrix
        cm = confusion_matrix(truths, preds)
        print(f"  Confusion Matrix:\n{cm}")
    
    # Calculate overall metrics
    overall_accuracy /= len(CARE_LABELS)
    overall_f1 /= len(CARE_LABELS)
    
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"Average Accuracy: {overall_accuracy:.4f}")
    print(f"Average F1-Score: {overall_f1:.4f}")
    print(f"Total samples evaluated: {len(all_predictions)}")
    
    # Save results to file
    results['overall_metrics'] = {
        'average_accuracy': float(overall_accuracy),
        'average_f1': float(overall_f1),
        'total_samples': len(all_predictions),
        'batch_size': batch_size,
        'include_analysis': include_analysis
    }
    
    output_path = "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results

if __name__ == "__main__":
    import sys
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    include_analysis = True
    evaluate_model(batch_size=batch_size, include_analysis=include_analysis)
