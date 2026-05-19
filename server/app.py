from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import asyncio
import time
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


def clear_embedding_cache():
    """Clear runtime caches (placeholder for compatibility)."""
    return None


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
                        # Flatten queued batch requests, run one inference call, and split back per request.
                        flat_contexts = []
                        flat_utterances = []
                        request_sizes = []
                        include_analysis = requests[0]["include_analysis"]

                        for req in requests:
                            req_contexts = req["contexts"]
                            req_utterances = req["utterances"]
                            req_size = len(req_utterances)
                            request_sizes.append(req_size)
                            flat_contexts.extend(req_contexts)
                            flat_utterances.extend(req_utterances)

                        start_time = time.time()
                        all_predictions = model.batch_predict(
                            flat_contexts,
                            flat_utterances,
                            batch_size=batch_size,
                            include_analysis=include_analysis
                        )
                        processing_time = time.time() - start_time

                        # Send results back to requests
                        offset = 0
                        for req, req_size in zip(requests, request_sizes):
                            req_predictions = all_predictions[offset:offset + req_size]
                            offset += req_size
                            req["future"].set_result({
                                "predictions": req_predictions,
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
            "contexts": request.contexts,
            "utterances": request.utterances,
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
