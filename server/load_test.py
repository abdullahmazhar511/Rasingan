"""
Load testing script for Care Model API
Tests throughput, latency, and robustness
"""

import requests
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import argparse


class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.results = {
            "latencies": [],
            "errors": [],
            "successes": 0,
            "failures": 0
        }
    
    def test_single_prediction(self) -> float:
        """Test single prediction endpoint"""
        payload = {
            "context": "Patient discusses feeling overwhelmed at work",
            "utterance": "That sounds really stressful. Tell me more about what's going on.",
            "include_analysis": True
        }
        
        start = time.time()
        try:
            response = requests.post(f"{self.base_url}/predict", json=payload)
            response.raise_for_status()
            latency = time.time() - start
            self.results["successes"] += 1
            return latency
        except Exception as e:
            self.results["failures"] += 1
            self.results["errors"].append(str(e))
            return None
    
    def test_batch_prediction(self, batch_size: int = 8) -> float:
        """Test batch prediction endpoint"""
        contexts = [
            f"Patient context {i}"
            for i in range(batch_size)
        ]
        utterances = [
            f"Therapist response {i}"
            for i in range(batch_size)
        ]
        
        payload = {
            "contexts": contexts,
            "utterances": utterances,
            "batch_size": batch_size,
            "include_analysis": True
        }
        
        start = time.time()
        try:
            response = requests.post(f"{self.base_url}/batch_predict", json=payload)
            response.raise_for_status()
            latency = time.time() - start
            self.results["successes"] += 1
            return latency
        except Exception as e:
            self.results["failures"] += 1
            self.results["errors"].append(str(e))
            return None
    
    def test_batch_async(self, batch_size: int = 8) -> float:
        """Test async batch prediction"""
        contexts = [f"Context {i}" for i in range(batch_size)]
        utterances = [f"Utterance {i}" for i in range(batch_size)]
        
        payload = {
            "contexts": contexts,
            "utterances": utterances,
            "batch_size": batch_size,
            "include_analysis": False
        }
        
        start = time.time()
        try:
            response = requests.post(f"{self.base_url}/batch_predict_async", json=payload)
            response.raise_for_status()
            latency = time.time() - start
            self.results["successes"] += 1
            return latency
        except Exception as e:
            self.results["failures"] += 1
            self.results["errors"].append(str(e))
            return None
    
    def run_concurrent_tests(self, num_requests: int = 100, 
                            test_type: str = "single",
                            num_workers: int = 10) -> None:
        """Run concurrent requests"""
        
        test_fn = {
            "single": self.test_single_prediction,
            "batch": lambda: self.test_batch_prediction(8),
            "async": lambda: self.test_batch_async(8)
        }.get(test_type, self.test_single_prediction)
        
        print(f"Running {num_requests} concurrent {test_type} prediction requests...")
        print(f"Workers: {num_workers}\n")
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(test_fn) for _ in range(num_requests)]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    latency = future.result()
                    if latency is not None:
                        self.results["latencies"].append(latency)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Completed: {i + 1}/{num_requests}")
                except Exception as e:
                    self.results["failures"] += 1
                    self.results["errors"].append(str(e))
        
        total_time = time.time() - start_time
        self.print_results(total_time)
    
    def print_results(self, total_time: float) -> None:
        """Print test results"""
        print("\n" + "="*60)
        print("LOAD TEST RESULTS")
        print("="*60)
        
        print(f"\nTotal Time: {total_time:.2f}s")
        print(f"Successful Requests: {self.results['successes']}")
        print(f"Failed Requests: {self.results['failures']}")
        print(f"Success Rate: {self.results['successes']/(self.results['successes']+self.results['failures'])*100:.1f}%")
        
        if self.results["latencies"]:
            lats = self.results["latencies"]
            print(f"\nLatency Statistics (in seconds):")
            print(f"  Min: {min(lats):.3f}s")
            print(f"  Max: {max(lats):.3f}s")
            print(f"  Mean: {statistics.mean(lats):.3f}s")
            print(f"  Median: {statistics.median(lats):.3f}s")
            print(f"  Stdev: {statistics.stdev(lats) if len(lats) > 1 else 0:.3f}s")
            print(f"  P95: {sorted(lats)[int(len(lats)*0.95)]:.3f}s")
            print(f"  P99: {sorted(lats)[int(len(lats)*0.99)]:.3f}s")
        
        throughput = self.results['successes'] / total_time
        print(f"\nThroughput: {throughput:.2f} requests/second")
        
        if self.results["errors"]:
            print(f"\nSample Errors:")
            for err in self.results["errors"][:5]:
                print(f"  - {err}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Load test Care Model API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="API base URL")
    parser.add_argument("--requests", type=int, default=100,
                       help="Number of requests")
    parser.add_argument("--workers", type=int, default=10,
                       help="Number of concurrent workers")
    parser.add_argument("--type", choices=["single", "batch", "async"],
                       default="single", help="Test type")
    
    args = parser.parse_args()
    
    tester = LoadTester(args.url)
    
    # Check health
    try:
        response = requests.get(f"{args.url}/health")
        response.raise_for_status()
        print(f"✅ Server healthy: {response.json()}\n")
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return
    
    # Run tests
    tester.run_concurrent_tests(args.requests, args.type, args.workers)


if __name__ == "__main__":
    main()
