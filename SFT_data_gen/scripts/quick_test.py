#!/usr/bin/env python3
"""
Quick test script with minimal data generation to verify everything works.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_config, create_output_path
from generator import DrillGenerator

def setup_test_environment():
    """Setup environment for testing."""
    # Set required environment variables with defaults for testing
    os.environ.setdefault('API_KEY', 'test_key_please_set_real_key')
    os.environ.setdefault('MODEL_NAME', 'gpt-4o-mini')
    os.environ.setdefault('DATA_SIZE', '18')  # 2 per category for quick test
    os.environ.setdefault('CHUNK_SIZE', '2')
    os.environ.setdefault('BATCH_SIZE', '2')
    os.environ.setdefault('TEMPERATURE', '0.7')
    os.environ.setdefault('MAX_TOKENS', '1000')

async def main():
    """Run quick test generation."""
    print("🧪 Quick Test Mode - Generating 18 samples (2 per category)")
    print("="*60)
    
    # Check API key
    if os.environ.get('API_KEY') == 'test_key_please_set_real_key':
        print("❌ Please set a real API key:")
        print("   export API_KEY='your_actual_api_key'")
        print("   python scripts/quick_test.py")
        return
    
    # Setup test environment
    setup_test_environment()
    
    # Change to project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    os.chdir(project_dir)
    
    try:
        # Load configuration
        config = load_config('config/default.yaml')
        print(f"✅ Config loaded: {config.model_name}")
        print(f"   Test data size: {config.data_size}")
        
        # Create output path
        output_path = create_output_path(config.output_dir)
        print(f"✅ Output path: {output_path}")
        
        # Initialize generator
        generator = DrillGenerator(config)
        print(f"✅ Generator initialized")
        
        # Run generation
        print(f"\n🚀 Starting test generation...")
        await generator.generate_all_drills(output_path)
        
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 Check results in: {output_path}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 