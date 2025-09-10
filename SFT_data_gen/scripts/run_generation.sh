#!/bin/bash

# Dataset Generation Script

# Usage: ./scripts/run_generation.sh [OPTIONS]
# Example: ./scripts/run_generation.sh --model gpt-4o --data-size 50000 --temperature 0.8

set -euo pipefail

# Default values
DEFAULT_DATASET="thai_mtbench"
DEFAULT_SUBJECT="all"
DEFAULT_MODEL_NAME="gpt-5"
DEFAULT_OUTPUT_DIR="output"
DEFAULT_DATA_SIZE="100000"
DEFAULT_LOG_LEVEL="INFO"
DEFAULT_CHUNK_SIZE="30"
DEFAULT_BATCH_SIZE="25"
DEFAULT_TEMPERATURE="0.7"
DEFAULT_MAX_TOKENS="32768"
DEFAULT_MAX_RETRIES="3"
DEFAULT_RETRY_DELAY="1"
DEFAULT_ENABLE_BATCH_MODE="true"
DEFAULT_MAX_BATCH_SIZE="10"
DEFAULT_MIN_BATCH_SIZE="5"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --subject)
            SUBJECT="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-size)
            DATA_SIZE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --retry-delay)
            RETRY_DELAY="$2"
            shift 2
            ;;
        --enable-batch-mode)
            ENABLE_BATCH_MODE="$2"
            shift 2
            ;;
        --max-batch-size)
            MAX_BATCH_SIZE="$2"
            shift 2
            ;;
        --min-batch-size)
            MIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET                Dataset name (default: $DEFAULT_DATASET)"
            echo "  --subject SUBJECT                Subject name (default: $DEFAULT_SUBJECT)"
            echo "  --model MODEL                    Model name (default: $DEFAULT_MODEL_NAME)"
            echo "  --output-dir DIR                 Output directory (default: $DEFAULT_OUTPUT_DIR)"
            echo "  --data-size SIZE                 Number of data points to generate (default: $DEFAULT_DATA_SIZE)"
            echo "  --log-level LEVEL                Log level (default: $DEFAULT_LOG_LEVEL)"
            echo "  --chunk-size SIZE                Chunk size for processing (default: $DEFAULT_CHUNK_SIZE)"
            echo "  --batch-size SIZE                Batch size for concurrent processing (default: $DEFAULT_BATCH_SIZE)"
            echo "  --temperature TEMP               Temperature for generation (default: $DEFAULT_TEMPERATURE)"
            echo "  --max-tokens TOKENS              Maximum tokens per response (default: $DEFAULT_MAX_TOKENS)"
            echo "  --max-retries RETRIES            Maximum retry attempts (default: $DEFAULT_MAX_RETRIES)"
            echo "  --retry-delay SECONDS            Delay between retries (default: $DEFAULT_RETRY_DELAY)"
            echo "  --enable-batch-mode BOOL         Enable batch mode (default: $DEFAULT_ENABLE_BATCH_MODE)"
            echo "  --max-batch-size SIZE            Maximum batch size (default: $DEFAULT_MAX_BATCH_SIZE)"
            echo "  --min-batch-size SIZE            Minimum batch size (default: $DEFAULT_MIN_BATCH_SIZE)"
            echo "  --help, -h                       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model gpt-5 --data-size 50000"
            echo "  $0 --temperature 0.8 --batch-size 20"
            echo "  $0 --output-dir custom_output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set values with defaults if not provided
export DATASET="${DATASET:-$DEFAULT_DATASET}"
export SUBJECT="${SUBJECT:-$DEFAULT_SUBJECT}"
export MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
export OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
export DATA_SIZE="${DATA_SIZE:-$DEFAULT_DATA_SIZE}"
export LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
export CHUNK_SIZE="${CHUNK_SIZE:-$DEFAULT_CHUNK_SIZE}"
export BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
export TEMPERATURE="${TEMPERATURE:-$DEFAULT_TEMPERATURE}"
export MAX_TOKENS="${MAX_TOKENS:-$DEFAULT_MAX_TOKENS}"
export MAX_RETRIES="${MAX_RETRIES:-$DEFAULT_MAX_RETRIES}"
export RETRY_DELAY="${RETRY_DELAY:-$DEFAULT_RETRY_DELAY}"
export ENABLE_BATCH_MODE="${ENABLE_BATCH_MODE:-$DEFAULT_ENABLE_BATCH_MODE}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-$DEFAULT_MAX_BATCH_SIZE}"
export MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-$DEFAULT_MIN_BATCH_SIZE}"

# Check if AZURE_OPENAI_API_KEY is set
if [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
    echo "Error: AZURE_OPENAI_API_KEY environment variable is required"
    echo "Please set it before running the script:"
    echo "  export AZURE_OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# 📊 Progress Display
echo "Starting drill generation with configuration:"
echo "  Dataset: $DATASET"
echo "  Subject: $SUBJECT"
echo "  Model: $MODEL_NAME"
echo "  Data size: $DATA_SIZE"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Batch size: $BATCH_SIZE"
echo "  Temperature: $TEMPERATURE"
echo "  Max tokens: $MAX_TOKENS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log level: $LOG_LEVEL"
echo "  Max retries: $MAX_RETRIES"
echo "  Retry delay: $RETRY_DELAY"
echo "  Batch mode: $ENABLE_BATCH_MODE"
echo "  Max batch size: $MAX_BATCH_SIZE"
echo "  Min batch size: $MIN_BATCH_SIZE"

# Create virtual environment if it doesn't exist
if [ ! -d "../generation_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ../generation_env
    pip install -r ../requirements.txt
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../generation_env/bin/activate

# Run the generation
echo "Starting drill generation..."
python ../src/generator.py --dataset $DATASET --subject $SUBJECT