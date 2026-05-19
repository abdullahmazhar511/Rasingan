"""
Test Care Model Server with F1 Score Evaluation
Loads annotated test data and evaluates server predictions
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import glob
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_utils import load_test_data, CARE_LABELS

SCRIPT_DIR = Path(__file__).resolve().parent
RASINGAN_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_PATH = RASINGAN_DIR / "respair_mhcopilot_format" / "test.csv"


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ServerTester:
    def __init__(self, base_url: str = "http://localhost:8000", data_dir: str = None):
        self.base_url = base_url.rstrip("/")
        # Support both file and directory paths
        if data_dir and data_dir.endswith('.csv'):
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_PATH
        self.results = {
            "total_samples": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "predictions": [],
            "ground_truth": [],
            "utterances": []
        }
    
    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            logger.info(f"✅ Server healthy: {health}")
            return True
        except Exception as e:
            logger.error(f"❌ Server not responding: {e}")
            return False
    
    def load_data(self) -> Tuple[List[str], List[str], List[Dict]]:
        """Load test data using SimpleDataset"""
        logger.info(f"Loading data from {self.data_dir} using SimpleDataset...")
        contexts, utterances, ground_truths = load_test_data(str(self.data_dir))
        logger.info(f"Loaded {len(utterances)} samples")
        return contexts, utterances, ground_truths

    
    def prepare_samples(self, utterances: List[str], ground_truths: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Samples are already prepared, just return them"""
        logger.info(f"Prepared {len(utterances)} samples")
        return utterances, ground_truths
    
    def get_predictions_batch(self, contexts: List[str], utterances: List[str], batch_size: int = 16) -> List[Dict]:
        """Get predictions from server in batches"""
        logger.info(f"Getting predictions from server (batch_size={batch_size})...")
        
        all_predictions = []
        num_batches = (len(utterances) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(utterances))
            batch_contexts = contexts[start_idx:end_idx]
            batch_utterances = utterances[start_idx:end_idx]
            
            try:
                response = requests.post(
                    f"{self.base_url}/batch_predict",
                    json={
                        "contexts": batch_contexts,
                        "utterances": batch_utterances,
                        "batch_size": len(batch_utterances),
                        "include_analysis": True  # Use analysis for better accuracy
                    },
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                batch_predictions = result["predictions"]
                all_predictions.extend(batch_predictions)
                
                logger.info(f"  Batch {batch_idx + 1}/{num_batches}: {len(batch_utterances)} samples - "
                          f"Time: {result['processing_time']:.3f}s")
                
                self.results["successful_predictions"] += len(batch_predictions)
                
            except Exception as e:
                logger.error(f"  Batch {batch_idx + 1} failed: {e}")
                # Add empty predictions for this batch
                for _ in batch_utterances:
                    all_predictions.append({label: -1 for label in CARE_LABELS})
                self.results["failed_predictions"] += len(batch_utterances)
        
        return all_predictions
    
    def compute_metrics(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """Compute F1 scores and other metrics"""
        logger.info("\nComputing metrics...")
        
        metrics = {
            "overall_f1": None,
            "per_label_f1": {},
            "per_label_precision": {},
            "per_label_recall": {},
            "per_label_support": {},
            "confusion_matrices": {}
        }
        
        # Collect predictions and ground truth for each label
        for label_idx, label in enumerate(CARE_LABELS):
            preds_label = []
            truth_label = []
            
            for pred, truth in zip(predictions, ground_truths):
                # Get prediction value (convert back to label space)
                pred_val = pred.get(label, -1)
                
                # Get ground truth
                truth_val = truth.get(label)
                
                if truth_val is not None:  # Only include if we have ground truth
                    preds_label.append(pred_val)
                    truth_label.append(truth_val)
            
            if len(preds_label) > 0:
                # Compute metrics
                f1 = f1_score(truth_label, preds_label, average='weighted', zero_division=0)
                metrics["per_label_f1"][label] = f1
                
                # Per-class precision and recall
                report = classification_report(
                    truth_label, preds_label, 
                    output_dict=True, 
                    zero_division=0
                )
                metrics["per_label_precision"][label] = report.get('weighted avg', {}).get('precision', 0)
                metrics["per_label_recall"][label] = report.get('weighted avg', {}).get('recall', 0)
                metrics["per_label_support"][label] = len(truth_label)
                
                # Store confusion matrix
                try:
                    cm = confusion_matrix(truth_label, preds_label)
                    metrics["confusion_matrices"][label] = cm.tolist()
                except:
                    pass
        
        # Overall F1
        all_preds = []
        all_truths = []
        for pred, truth in zip(predictions, ground_truths):
            for label in CARE_LABELS:
                pred_val = pred.get(label, -1)
                truth_val = truth.get(label)
                if truth_val is not None:
                    all_preds.append(pred_val)
                    all_truths.append(truth_val)
        
        if len(all_preds) > 0:
            metrics["overall_f1"] = f1_score(all_truths, all_preds, average='weighted', zero_division=0)
        
        return metrics
    
    def run_evaluation(self, batch_size: int = 16) -> Dict:
        """Run full evaluation pipeline"""
        logger.info("=" * 70)
        logger.info("CARE MODEL SERVER EVALUATION")
        logger.info("=" * 70)
        
        # Check server
        if not self.check_server():
            logger.error("Cannot proceed without server")
            return None
        
        # Load data
        contexts, utterances, ground_truths = self.load_data()
        self.results["total_samples"] = len(utterances)
        self.results["utterances"] = utterances
        self.results["ground_truth"] = ground_truths
        
        # Prepare samples
        utterances, ground_truths = self.prepare_samples(utterances, ground_truths)
        
        # Get predictions
        start_time = time.time()
        predictions = self.get_predictions_batch(contexts, utterances, batch_size)
        total_time = time.time() - start_time
        
        self.results["predictions"] = predictions
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, ground_truths)
        
        # Print results
        self.print_results(metrics, total_time)
        
        return {
            "metrics": metrics,
            "results": self.results,
            "total_time": total_time
        }
    
    def print_results(self, metrics: Dict, total_time: float) -> None:
        """Print evaluation results"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\nDataset Statistics:")
        print(f"  Total Samples: {self.results['total_samples']}")
        print(f"  Successful Predictions: {self.results['successful_predictions']}")
        print(f"  Failed Predictions: {self.results['failed_predictions']}")
        print(f"  Success Rate: {self.results['successful_predictions'] / self.results['total_samples'] * 100:.1f}%")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Avg Time per Sample: {total_time / self.results['total_samples'] * 1000:.2f}ms")
        
        print(f"\nOverall Performance:")
        print(f"  Overall F1 Score (weighted): {metrics['overall_f1']:.4f}")
        
        print(f"\nPer-Label Performance:")
        print(f"{'Label':<30} {'F1':<8} {'Precision':<12} {'Recall':<12} {'Support':<8}")
        print("-" * 70)
        
        for label in CARE_LABELS:
            f1 = metrics["per_label_f1"].get(label, 0)
            prec = metrics["per_label_precision"].get(label, 0)
            rec = metrics["per_label_recall"].get(label, 0)
            support = metrics["per_label_support"].get(label, 0)
            
            print(f"{label:<30} {f1:<8.4f} {prec:<12.4f} {rec:<12.4f} {support:<8}")
        
        print("\n" + "=" * 70)
    
    def save_results(self, output_file: str = "evaluation_results.json") -> None:
        """Save results to JSON"""
        output_path = Path(output_file)
        
        # Convert confusion matrices to JSON-serializable format
        results_to_save = self.results.copy()
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Care Model Server")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--data-dir", 
                       default=str(DEFAULT_DATA_PATH),
                       help="Path to data directory (uses MHCoPilot_Dataset by default)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for predictions")
    parser.add_argument("--output", default="evaluation_results.json", help="Output results file")
    
    args = parser.parse_args()
    
    # Always use parent directory for MHCoPilot_Dataset
    data_path = args.data_dir
    if data_path.endswith('.csv'):
        data_path = str(Path(data_path).parent)
    
    tester = ServerTester(args.url, data_path)
    result = tester.run_evaluation(args.batch_size)
    
    if result:
        tester.save_results(args.output)
        print(f"\n✅ Evaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
