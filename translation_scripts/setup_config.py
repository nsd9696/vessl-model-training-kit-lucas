#!/usr/bin/env python3
"""
Setup script to create config.env file for MTBench translation
"""

import os
from pathlib import Path

def create_config_file():
    """Create config.env file with user input."""
    
    config_file = "config.env"
    
    if os.path.exists(config_file):
        print(f"⚠️  {config_file} already exists!")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    print("🚀 MTBench Translation Configuration Setup")
    print("=" * 50)
    print("Please provide your Azure OpenAI configuration:")
    print()
    
    # Get user input
    api_key = input("Azure OpenAI API Key: ").strip()
    if not api_key:
        print("❌ API Key is required!")
        return
    
    endpoint = input("Azure OpenAI Endpoint (e.g., https://your-resource.openai.azure.com/): ").strip()
    if not endpoint:
        print("❌ Endpoint is required!")
        return
    
    # Optional settings with defaults
    api_version = input("API Version (default: 2025-01-01-preview): ").strip() or "2025-01-01-preview"
    model = input("Model to use (default: gpt-5): ").strip() or "gpt-5"
    batch_size = input("Batch size (default: 10): ").strip() or "10"
    max_concurrent = input("Max concurrent requests (default: 5): ").strip() or "5"
    
    # Create config content
    config_content = f"""# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY={api_key}
AZURE_OPENAI_ENDPOINT={endpoint}
AZURE_OPENAI_API_VERSION={api_version}

# Translation Settings
MODEL={model}
BATCH_SIZE={batch_size}
MAX_CONCURRENT={max_concurrent}
"""
    
    # Write config file
    try:
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print()
        print(f"✅ Configuration saved to {config_file}")
        print()
        print("📋 Your configuration:")
        print(f"   Model: {model}")
        print(f"   Endpoint: {endpoint}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Max Concurrent: {max_concurrent}")
        print()
        print("🚀 You can now run the translation script!")
        print("   python translate_mtbench_dataset.py --input your_file.jsonl --output output.jsonl")
        print()
        print("   Or use the shell script:")
        print("   ./run_translation.sh")
        
    except Exception as e:
        print(f"❌ Error creating config file: {e}")

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import openai
        import dotenv
        import tqdm
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    print("🧪 MTBench Translation Setup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        exit(1)
    
    # Create config file
    create_config_file()
