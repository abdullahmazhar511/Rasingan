conda create --name verl python=3.12 -y &&
conda activate verl && 
cd verl &&
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh &&
pip install --no-deps -e . &&
pip install scikit-learn openai numpy==2.2 &&
pip install trl torchao -U &&
pip install transformers==4.56.2 &&
pip install sentence_transformers evaluate nltk rouge_score bert_score