# CARE v2: Therapeutic Response Classifier

CARE v2 is a context-aware therapeutic response classification pipeline designed to evaluate therapist utterances across 6 key dimensions of empathy and skill.

## 🚀 Key Features
- **Backbone**: Qwen/Qwen3-4B-Instruct-2507 with LoRA adapters.
- **Context-Aware**: Uses conversation history (m=4) to provide deep context for each therapist response.
- **Prototype Retrieval**: Enhances classification by retrieving similar positive and negative examples from a curated knowledge base.
- **Distilled Knowledge**: Integrates LLM-generated explanations (from `faith_data/05_generated_explanations`) as a retrieval signal to help the model understand *why* a certain response is helpful or harmful.
- **Hybrid Ordinal Loss**: Combines **MSE** (to penalize ordinal distance) and **Cross-Entropy** (to ensure categorical accuracy).
- **Two-Stage Training**:
    - **Phase 1**: Joint training of LoRA adapters and classification heads.
    - **Phase 2**: Freezes LoRA adapters and fine-tunes only the classification heads with a higher learning rate.

## 📁 Project Structure
- `care_v2/configs/config.py`: Central configuration for hyperparameters, data paths, and model IDs.
- `care_v2/models/care_classifier.py`: Custom model architecture implementing the Qwen backbone and the hybrid loss logic.
- `care_v2/datasets/care_dataset.py`: Handles tokenization, context preparation, and prototype integration for the Trainer.
- `care_v2/retrieval/prototype_retriever.py`: Logic for loading and searching the prototype embedding index.
- `care_v2/training/metrics.py`: Computes QWK, Accuracy, and both Macro/Weighted F1 scores.
- `care_v2/train.py`: Main entry point for training. Includes automated checkpointing for the best model.
- `care_v2/evaluate.py`: Script to load a trained checkpoint and run a full evaluation on the test set, generating confusion matrices.

## 🛠️ Installation & Setup
1. **Requirements**:
   ```bash
   pip install torch transformers peft accelerate seaborn matplotlib scikit-learn
   ```
2. **Data Augmentation**:
   To handle the extreme class imbalance, use the augmentation script to increase negative label distribution to ~15%:
   ```bash
   python scratch/augment_negatives.py
   ```

## 📈 Running the Pipeline
1. **Configure**: Update `care_v2/configs/config.py` with your desired `prototype_k` (e.g., k_8, k_12, k_16) and `output_dir`.
2. **Train**:
   ```bash
   python -m care_v2.train
   ```
3. **Evaluate**:
   Ensure `resume_checkpoint` in the config points to your trained model, then run:
   ```bash
   python -m care_v2.evaluate
   ```

## 📊 Evaluation Metrics
- **QWK (Quadratic Weighted Kappa)**: Primary metric. Measures ordinal agreement (penalizing large distance errors).
- **Macro F1**: Measures performance across all classes equally (best for imbalanced data).
- **Weighted F1**: Measures performance weighted by class frequency.
- **Confusion Matrix**: Saved automatically for each dimension in the `plots/` subdirectory.
