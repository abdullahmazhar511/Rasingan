# Care Model FastAPI Server

Production-ready FastAPI server for the Care Model with request batching, caching, and async support.

## Quick Start

```bash
# From the server folder
pip install -r requirements_server.txt
./server.sh dev
```

Server runs on **http://localhost:8000**

## Available Commands

```bash
./server.sh check      # Check dependencies
./server.sh dev        # Development mode (auto-reload)
./server.sh prod       # Production mode
./server.sh test       # Test running server
./server.sh help       # Show help
```

## API Usage

### Swagger UI
http://localhost:8000/docs

### Python
```python
from client import CareModelClient

client = CareModelClient("http://localhost:8000")
result = client.predict(
    context="Patient feels overwhelmed",
    utterance="That sounds stressful. Tell me more."
)
print(result)
```

### cURL
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Patient context",
    "utterance": "Therapist response",
    "include_analysis": true
  }'
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/predict` | POST | Single prediction |
| `/batch_predict` | POST | Batch prediction (sync) |
| `/batch_predict_async` | POST | Batch prediction (async) |
| `/clear_cache` | POST | Clear embedding cache |

## Configuration

Edit `config.py` to adjust:
- `BATCH_SIZE` - Larger = faster, more memory
- `BATCH_TIMEOUT` - Lower = lower latency
- `MAX_CACHE_SIZE` - Embedding cache size

## Load Testing

```bash
python load_test.py --requests 100 --workers 10 --type batch
```

## Performance Tips

- Set `include_analysis=False` for ~30% speed boost
- Increase `batch_size` to 16-32 for throughput
- Decrease `batch_size` to 2-4 to reduce memory
- Call `/clear_cache` periodically on long-running servers

## Files

- `app.py` - Main FastAPI server
- `client.py` - Python client library
- `config.py` - Configuration
- `server.sh` - Launcher script
- `load_test.py` - Performance testing
- `requirements_server.txt` - Dependencies
- `requirements-dev.txt` - Development tools
