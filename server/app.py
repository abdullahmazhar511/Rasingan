from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import asyncio
import time
from functools import lru_cache
from collections import defaultdict
import logging
import sys
import os

# Add parent directory to path to import inference
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import CareModel, CARE_LABELS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Care Model API", version="1.0.0")

# Global model instance
model: Optional[CareModel] = None
request_queue = asyncio.Queue()
batch_processor_task = None

# Caching parameters
BATCH_SIZE = 8
BATCH_TIMEOUT = 2.0  # Max wait time in seconds before processing smaller batch
MAX_CACHE_SIZE = 1000


# Pydantic models
class PredictRequest(BaseModel):
    context: str
    utterance: str
    include_analysis: bool = True


class BatchPredictRequest(BaseModel):
    contexts: List[str]
    utterances: List[str]
    batch_size: int = 8
    include_analysis: bool = True


class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    processing_time: float


class BatchPredictionResponse(BaseModel):
    predictions: List[Dict[str, float]]
    processing_time: float
    num_samples: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Caching decorator for embedding similarities
@lru_cache(maxsize=MAX_CACHE_SIZE)
def cached_embedding_similarity(utterance: str, polarity: str, dimension: str) -> float:
    """Cache embedding similarity computations"""
    if model is None:
        return 0.0
    
    from sentence_transformers.util import cos_sim
    text_embedding = model.embedding_model.encode(utterance, convert_to_tensor=True)
    
    if dimension not in model.dimension_samples:
        return 0.0
    
    samples = model.dimension_samples[dimension]
    sample = samples.get(polarity)
    
    if sample and sample.get('embedding') is not None:
        similarity = cos_sim(text_embedding, sample['embedding']).item()
        return similarity
    
    return 0.0


def clear_embedding_cache():
    """Clear the embedding cache"""
    cached_embedding_similarity.cache_clear()


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, batch_processor_task
    logger.info("Loading Care Model...")
    model = CareModel()
    logger.info("Model loaded successfully!")
    
    # Start batch processor
    batch_processor_task = asyncio.create_task(batch_processor())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global batch_processor_task
    if batch_processor_task:
        batch_processor_task.cancel()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cleanup complete")


async def batch_processor():
    """Background task that periodically processes batched requests"""
    pending_requests = defaultdict(list)
    last_batch_time = time.time()
    
    while True:
        try:
            current_time = time.time()
            time_since_last_batch = current_time - last_batch_time
            
            # Check if we should process (batch full or timeout exceeded)
            should_process = time_since_last_batch >= BATCH_TIMEOUT
            
            try:
                # Non-blocking queue get with timeout
                request = await asyncio.wait_for(request_queue.get(), timeout=0.1)
                pending_requests[request["batch_size"]].append(request)
                
                # Check if batch is full
                if len(pending_requests[request["batch_size"]]) >= request["batch_size"]:
                    should_process = True
            except asyncio.TimeoutError:
                pass
            
            # Process batches if conditions are met
            if should_process and any(pending_requests.values()):
                for batch_size, requests in list(pending_requests.items()):
                    if len(requests) > 0:
                        # Process this batch
                        contexts = [r["context"] for r in requests]
                        utterances = [r["utterance"] for r in requests]
                        include_analysis = requests[0]["include_analysis"]
                        
                        start_time = time.time()
                        predictions = model.batch_predict(
                            contexts, 
                            utterances, 
                            batch_size=batch_size,
                            include_analysis=include_analysis
                        )
                        processing_time = time.time() - start_time
                        
                        # Send results back to requests
                        for req, pred in zip(requests, predictions):
                            req["future"].set_result({
                                "predictions": pred,
                                "processing_time": processing_time
                            })
                        
                        # Clear processed batch
                        del pending_requests[batch_size]
                        last_batch_time = time.time()
            
            await asyncio.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error in batch processor: {e}")
            await asyncio.sleep(0.1)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    """Single prediction with optional batching queue"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        result = model.predict(
            request.context,
            request.utterance,
            include_analysis=request.include_analysis
        )
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=result,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictRequest):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.contexts) != len(request.utterances):
        raise HTTPException(status_code=400, detail="contexts and utterances must have same length")
    
    if len(request.contexts) == 0:
        raise HTTPException(status_code=400, detail="Empty request")
    
    try:
        start_time = time.time()
        predictions = model.batch_predict(
            request.contexts,
            request.utterances,
            batch_size=request.batch_size,
            include_analysis=request.include_analysis
        )
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            processing_time=processing_time,
            num_samples=len(predictions)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict_async")
async def batch_predict_async(request: BatchPredictRequest, background_tasks: BackgroundTasks):
    """Async batch prediction - queues request for batched processing"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.contexts) != len(request.utterances):
        raise HTTPException(status_code=400, detail="contexts and utterances must have same length")
    
    if len(request.contexts) == 0:
        raise HTTPException(status_code=400, detail="Empty request")
    
    try:
        # Create future for result
        future = asyncio.Future()
        
        # Queue request
        await request_queue.put({
            "context": request.contexts,
            "utterance": request.utterances,
            "batch_size": request.batch_size,
            "include_analysis": request.include_analysis,
            "future": future
        })
        
        # Wait for result with timeout
        result = await asyncio.wait_for(future, timeout=30.0)
        
        return BatchPredictionResponse(
            predictions=result["predictions"],
            processing_time=result["processing_time"],
            num_samples=len(result["predictions"])
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request processing timeout")
    except Exception as e:
        logger.error(f"Async batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_cache")
async def clear_cache():
    """Clear embedding cache"""
    try:
        clear_embedding_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
