"""
Configuration for Care Model FastAPI Server
Adjust these settings for performance tuning
"""

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_WORKERS = 1  # Set to 1 for GPU, or more for CPU-only
RELOAD_ON_CHANGE = False  # Enable for development

# Batch Processing
BATCH_SIZE = 8  # Larger = higher throughput, more GPU memory
BATCH_TIMEOUT = 2.0  # Seconds - wait this long before processing incomplete batch
QUEUE_MAX_SIZE = 1000  # Max requests to queue

# Caching
CACHE_ENABLED = True
MAX_CACHE_SIZE = 1000  # Embedding similarity cache size
CACHE_TTL = 3600  # Cache time-to-live in seconds (0 = infinite)

# Model Configuration
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
CLASSIFIER_WEIGHTS = "/home/umairai/faith/faith/classification_output_stable_v5"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# GPU Configuration
GPU_DEVICE = "cuda:0"  # Which GPU to use
GPU_MEMORY_FRACTION = 0.9  # What fraction of GPU memory to use
EMPTY_CACHE_INTERVAL = 5  # Empty GPU cache every N batches
USE_MIXED_PRECISION = True  # Use float16/bfloat16

# Analysis Configuration
INCLUDE_ANALYSIS_DEFAULT = True  # Include analysis by default
MAX_ANALYSIS_LENGTH = 2048  # Max tokens for analysis

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "server.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance Tuning - RECOMMENDED PROFILES
"""
LIGHT DUTY (CPU/Small GPU):
  BATCH_SIZE = 2
  BATCH_TIMEOUT = 5.0
  MAX_CACHE_SIZE = 100
  GPU_MEMORY_FRACTION = 0.5

MEDIUM DUTY (Standard GPU):
  BATCH_SIZE = 8
  BATCH_TIMEOUT = 2.0
  MAX_CACHE_SIZE = 1000
  GPU_MEMORY_FRACTION = 0.9

HIGH THROUGHPUT (Server):
  BATCH_SIZE = 32
  BATCH_TIMEOUT = 1.0
  MAX_CACHE_SIZE = 5000
  GPU_MEMORY_FRACTION = 0.95
  SERVER_WORKERS = 2
"""
