#!/bin/bash

# Care Model FastAPI Server Launcher
# This script handles common server operations

set -e

# Activate conda environment used by CARE inference dependencies.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${GREEN}==================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}==================================${NC}"
}

print_error() {
    echo -e "${RED}❌ Error: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA CUDA found"
        nvidia-smi --query-gpu=name --format=csv,noheader
    else
        print_info "NVIDIA CUDA not found (CPU-only mode)"
    fi
    
    # Check pip packages
    python3 -c "import fastapi; import uvicorn; import torch" 2>/dev/null && \
        print_success "Required Python packages installed" || \
        (print_error "Required packages missing. Run: pip install -r requirements_server.txt"; exit 1)
}

# Run development server
run_dev() {
    print_header "Starting Development Server"
    print_info "Server will start on http://0.0.0.0:8000"
    print_info "Swagger UI: http://localhost:8000/docs"
    print_info "Press Ctrl+C to stop"
    python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
}

# Run production server
run_prod() {
    print_header "Starting Production Server (with Gunicorn)"
    print_info "Server will start on http://0.0.0.0:8000"
    print_info "Workers: 1 (for single GPU)"
    python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
}

# Test the server
test_server() {
    print_header "Testing Server"
    
    print_info "Checking health endpoint..."
    if ! curl -s http://localhost:8000/health | python3 -m json.tool; then
        print_error "Server is not running"
        return 1
    fi
    print_success "Health check passed"
    
    print_info "Running client test..."
    python3 client.py
    print_success "Client test completed"
}

# Build Docker image
build_docker() {
    print_header "Building Docker Image"
    docker build -t care-model-api .
    print_success "Docker image built: care-model-api"
}

# Run Docker container
run_docker() {
    print_header "Running Docker Container"
    docker run --gpus all -p 8000:8000 --name care-model-api care-model-api
}

# Show usage
show_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  check       - Check dependencies"
    echo "  dev         - Run development server (with auto-reload)"
    echo "  prod        - Run production server"
    echo "  test        - Test running server"
    echo "  docker-build - Build Docker image"
    echo "  docker-run   - Run Docker container"
    echo "  help        - Show this help message"
    echo ""
}

# Main
case "${1:-help}" in
    check)
        check_dependencies
        ;;
    dev)
        check_dependencies
        run_dev
        ;;
    prod)
        check_dependencies
        run_prod
        ;;
    test)
        test_server
        ;;
    docker-build)
        build_docker
        ;;
    docker-run)
        run_docker
        ;;
    help)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
