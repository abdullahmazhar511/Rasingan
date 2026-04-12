import requests
import json
import time
from typing import List, Dict
import asyncio
import aiohttp


class CareModelClient:
    """Client for Care Model API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self) -> Dict:
        """Check if server is healthy"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(self, context: str, utterance: str, include_analysis: bool = True) -> Dict:
        """Single prediction"""
        payload = {
            "context": context,
            "utterance": utterance,
            "include_analysis": include_analysis
        }
        response = requests.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_predict(self, contexts: List[str], utterances: List[str], 
                     batch_size: int = 8, include_analysis: bool = True) -> Dict:
        """Batch prediction (synchronous)"""
        payload = {
            "contexts": contexts,
            "utterances": utterances,
            "batch_size": batch_size,
            "include_analysis": include_analysis
        }
        response = requests.post(f"{self.base_url}/batch_predict", json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_predict_async(self, contexts: List[str], utterances: List[str],
                           batch_size: int = 8, include_analysis: bool = True) -> Dict:
        """Batch prediction (async with server-side batching)"""
        payload = {
            "contexts": contexts,
            "utterances": utterances,
            "batch_size": batch_size,
            "include_analysis": include_analysis
        }
        response = requests.post(f"{self.base_url}/batch_predict_async", json=payload)
        response.raise_for_status()
        return response.json()
    
    def clear_cache(self) -> Dict:
        """Clear server cache"""
        response = requests.post(f"{self.base_url}/clear_cache")
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    client = CareModelClient()
    
    # Check health
    print("Health check...")
    health = client.health_check()
    print(f"Server status: {health}\n")
    
    # Single prediction
    print("Single prediction...")
    result = client.predict(
        context="Patient discusses feeling overwhelmed at work",
        utterance="That sounds really stressful. Tell me more about what's going on.",
        include_analysis=True
    )
    print(f"Single prediction result: {json.dumps(result, indent=2)}\n")
    
    # Batch prediction
    print("Batch prediction...")
    contexts = [
        "Patient discusses feeling overwhelmed at work",
        "Patient talks about relationship issues",
        "Patient expresses anxiety about future"
    ]
    utterances = [
        "That sounds really stressful. Tell me more about what's going on.",
        "I hear you. Relationships can be complicated. What do you think is at the core of this?",
        "These feelings are valid. Let's explore what's making you anxious."
    ]
    
    batch_result = client.batch_predict(contexts, utterances, batch_size=3)
    print(f"Batch prediction result: {json.dumps(batch_result, indent=2)}\n")
    
    # Clear cache
    print("Clearing cache...")
    cache_result = client.clear_cache()
    print(f"Cache result: {cache_result}")
