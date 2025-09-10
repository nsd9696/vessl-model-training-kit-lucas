#!/bin/bash

# Server Deployment Script for MTBench Drill Dataset Generation
# This script can be run on any server environment (Docker, cloud, etc.)
set -e

echo "🚀 MTBench Drill Generation - Server Deployment Mode"
echo "================================================"

# Get project directory (works from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "📁 Project directory: $PROJECT_DIR"

# Navigate to project directory
cd "$PROJECT_DIR"

# Server Environment Configuration
# These can be overridden by environment variables or .env file
export MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
export DATA_SIZE="${DATA_SIZE:-10000}"
export CHUNK_SIZE="${CHUNK_SIZE:-10}"
export BATCH_SIZE="${BATCH_SIZE:-5}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export MAX_TOKENS="${MAX_TOKENS:-2000}"
export MAX_RETRIES="${MAX_RETRIES:-3}"
export RETRY_DELAY="${RETRY_DELAY:-1}"

# Server-specific paths (absolute paths for deployment)
export SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-$PROJECT_DIR/source_data/mt_bench_thai_full.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/output}"

# Logging configuration for server
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Check required environment variables
if [ -z "$API_KEY" ]; then
    echo "❌ Error: API_KEY environment variable is required"
    echo "   Set it with: export API_KEY=your_api_key_here"
    echo "   Or create a .env file with API_KEY=your_key"
    exit 1
fi

# Load .env file if it exists (for server deployment)
if [ -f ".env" ]; then
    echo "📄 Loading .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Display configuration
echo ""
echo "🔧 Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Data size: $DATA_SIZE"
echo "  Chunk size: $CHUNK_SIZE" 
echo "  Batch size: $BATCH_SIZE"
echo "  Temperature: $TEMPERATURE"
echo "  Max tokens: $MAX_TOKENS"
echo "  Source data: $SOURCE_DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log level: $LOG_LEVEL"
echo ""

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
    echo "   Consider using: python -m venv venv && source venv/bin/activate"
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt --quiet
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if source data exists
if [ ! -f "$SOURCE_DATA_PATH" ]; then
    echo "❌ Error: Source data file not found: $SOURCE_DATA_PATH"
    echo "   Please ensure the source data is available"
    exit 1
fi

# Run generation
echo "🚀 Starting drill generation..."
echo "   (This may take 2-3 hours for 10,000 items)"
echo ""

python src/generator.py --config config/default.yaml

echo ""
echo "🎉 Generation completed successfully!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo "📊 Check the final cost report in the logs" 