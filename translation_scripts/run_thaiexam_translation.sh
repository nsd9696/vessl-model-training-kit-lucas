#!/bin/bash

# ThaiExam Dataset Translation Runner Script
# This script helps you run the translation with proper environment setup

set -e  # Exit on any error

echo "🚀 ThaiExam Dataset Translation Runner"
echo "====================================="

# Check if required files exist
if [ ! -f "translate_thaiexam_dataset.py" ]; then
    echo "❌ Error: translate_thaiexam_dataset.py not found!"
    echo "   Please run this script from the directory containing the translation files."
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import openai, asyncio, tqdm" 2>/dev/null; then
    echo "❌ Missing dependencies. Installing..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "❌ requirements.txt not found!"
        exit 1
    fi
fi

# Preserve any existing MODEL variable before loading config
OVERRIDE_MODEL="$MODEL"

# Load config.env if it exists
if [ -f "config.env" ]; then
    echo "📄 Loading config.env..."
    source config.env
    # Export all endpoint-specific variables
    export GPT5_API_KEY GPT5_ENDPOINT GPT5_DEPLOYMENT_NAME GPT5_API_VERSION
    export GPT4_API_KEY GPT4_ENDPOINT GPT4_DEPLOYMENT_NAME GPT4_API_VERSION
    # Export legacy variables for backward compatibility
    export AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT DEPLOYMENT_NAME
fi

# Use override model if provided, otherwise use config or default
if [ -n "$OVERRIDE_MODEL" ]; then
    MODEL="$OVERRIDE_MODEL"
else
    MODEL=${MODEL:-"gpt-5"}
fi
echo "🤖 Selected Model: $MODEL"

# The Python script will automatically select the appropriate endpoint based on the model
# No need to validate legacy variables here since the multi-endpoint system handles this
echo "✅ Multi-endpoint configuration loaded"
echo "   📡 The script will auto-select the appropriate endpoint for model: $MODEL"

# Default values
INPUT_FILE="../generated_datasets/thaiexam_hard_training.jsonl"
OUTPUT_FILE="../generated_datasets/thaiexam_hard_training_english.jsonl"
BATCH_SIZE=30
MAX_CONCURRENT=30

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Input file not found: $INPUT_FILE"
    echo "   Please update the INPUT_FILE variable in this script or provide a valid path."
    exit 1
fi

echo ""
echo "📁 Input file: $INPUT_FILE"
echo "📁 Output file: $OUTPUT_FILE"
echo "🤖 Model: $MODEL"
echo "⚙️  Batch size: $BATCH_SIZE"
echo "⚙️  Max concurrent: $MAX_CONCURRENT"

# Confirm before proceeding
echo ""
read -p "Do you want to proceed with translation? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Translation cancelled."
    exit 0
fi

echo ""
echo "🔄 Starting translation..."
echo "   This may take several hours for the full dataset."
echo "   Progress will be logged to thaiexam_translation.log"
echo "   Checkpoints will be saved regularly."
echo ""

# Run the translation
python3 translate_thaiexam_dataset.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --batch-size "$BATCH_SIZE" \
    --max-concurrent "$MAX_CONCURRENT" \
    --model "$MODEL"

echo ""
echo "✅ Translation completed!"
echo "📁 Output saved to: $OUTPUT_FILE"
echo "📋 Check thaiexam_translation.log for detailed information"
