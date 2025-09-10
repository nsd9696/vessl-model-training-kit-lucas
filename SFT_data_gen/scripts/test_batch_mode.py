#!/usr/bin/env python3
"""Test script for batch mode functionality."""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_config
from generator import DrillGenerator
from batch_generator import BatchRequest

async def test_batch_mode():
    """Test batch mode functionality."""
    
    # Set up test environment
    os.environ['API_KEY'] = 'sk-test-dummy'
    os.environ['DATA_SIZE'] = '20'
    os.environ['ENABLE_BATCH_MODE'] = 'true'
    os.environ['MAX_BATCH_SIZE'] = '5'
    os.environ['MIN_BATCH_SIZE'] = '3'
    
    # Load configuration
    config = load_config("config/default.yaml")
    
    # Create generator
    generator = DrillGenerator(config)
    
    # Test configuration
    print("🔧 Testing Batch Mode Configuration")
    print(f"   Batch Mode Enabled: {config.enable_batch_mode}")
    print(f"   Max Batch Size: {config.max_batch_size}")
    print(f"   Min Batch Size: {config.min_batch_size}")
    print(f"   Batch Fallback: {config.batch_fallback_single}")
    print()
    
    # Test batch decision logic
    print("🧠 Testing Batch Decision Logic")
    if generator.batch_generator:
        test_counts = [1, 2, 3, 5, 8, 10, 15]
        for count in test_counts:
            should_batch = generator.batch_generator.should_use_batch_mode(count)
            batch_size = generator.batch_generator.calculate_batch_size(count)
            print(f"   Count {count:2d}: {'Batch' if should_batch else 'Single'} mode (batch_size: {batch_size})")
    print()
    
    # Test batch request creation
    print("📋 Testing Batch Request Creation")
    
    # Mock source data
    mock_source_items = [
        {
            "question_id": f"test_{i}",
            "turns": [f"Test question {i} turn 1", f"Test question {i} turn 2"]
        }
        for i in range(5)
    ]
    
    batch_request = BatchRequest(
        category="Math",
        source_items=mock_source_items,
        start_index=0,
        count=5
    )
    
    print(f"   Category: {batch_request.category}")
    print(f"   Source Items: {len(batch_request.source_items)}")
    print(f"   Start Index: {batch_request.start_index}")
    print(f"   Count: {batch_request.count}")
    print()
    
    # Test prompt building
    print("🎯 Testing Batch Prompt Building")
    try:
        if generator.batch_generator:
            prompt = generator.batch_generator.build_batch_prompt(batch_request)
            print(f"   Prompt Length: {len(prompt)} characters")
            print(f"   Contains 'Target Question 1': {'Target Question 1' in prompt}")
            print(f"   Contains 'Target Question 5': {'Target Question 5' in prompt}")
            print(f"   Contains JSON format: {'JSON' in prompt}")
        else:
            print("   ❌ Batch generator not available")
    except Exception as e:
        print(f"   ❌ Error building batch prompt: {e}")
    print()
    
    # Test response parsing
    print("🔍 Testing Response Parsing")
    
    # Mock batch response
    mock_response = '''
Here are the 3 drill problems:

```json
[
  {
    "drill_number": 1,
    "turns": [
      "คำถามที่ 1 รอบแรก",
      "คำถามที่ 1 รอบสอง"
    ],
    "reference": [
      "คำตอบที่ 1 รอบแรก",
      "คำตอบที่ 1 รอบสอง"
    ]
  },
  {
    "drill_number": 2,
    "turns": [
      "คำถามที่ 2 รอบแรก",
      "คำถามที่ 2 รอบสอง"
    ],
    "reference": [
      "คำตอบที่ 2 รอบแรก",
      "คำตอบที่ 2 รอบสอง"
    ]
  },
  {
    "drill_number": 3,
    "turns": [
      "คำถามที่ 3 รอบแรก",
      "คำถามที่ 3 รอบสอง"
    ],
    "reference": [
      "คำตอบที่ 3 รอบแรก",
      "คำตอบที่ 3 รอบสอง"
    ]
  }
]
```
'''
    
    try:
        if generator.batch_generator:
            test_batch_request = BatchRequest(
                category="Math",
                source_items=mock_source_items[:3],
                start_index=0,
                count=3
            )
            
            parsed_drills = generator.batch_generator.parse_batch_response(mock_response, test_batch_request)
            print(f"   Parsed {len(parsed_drills)} drills from response")
            
            if parsed_drills:
                first_drill = parsed_drills[0]
                print(f"   First drill ID: {first_drill.get('question_id', 'N/A')}")
                print(f"   First drill category: {first_drill.get('category', 'N/A')}")
                print(f"   First drill turns: {len(first_drill.get('turns', []))}")
                print(f"   First drill references: {len(first_drill.get('reference', []))}")
        else:
            print("   ❌ Batch generator not available")
    except Exception as e:
        print(f"   ❌ Error parsing batch response: {e}")
    print()
    
    # Test mode comparison
    print("⚖️  Testing Mode Comparison")
    
    test_scenarios = [
        ("Small batch", 2),
        ("Medium batch", 5),
        ("Large batch", 10)
    ]
    
    for scenario_name, count in test_scenarios:
        if generator.batch_generator:
            should_batch = generator.batch_generator.should_use_batch_mode(count)
            batch_size = generator.batch_generator.calculate_batch_size(count)
            mode = "batch" if should_batch else "single"
            print(f"   {scenario_name} ({count} items): {mode} mode (batch_size: {batch_size})")
        else:
            print(f"   {scenario_name} ({count} items): single mode (batch disabled)")
    print()
    
    print("✅ Batch mode testing completed!")
    print()
    print("🚀 To test with real API:")
    print("   export API_KEY='your_real_api_key'")
    print("   export DATA_SIZE='50'")
    print("   python src/generator.py --batch-size 5")
    print()
    print("📊 To compare performance:")
    print("   # Batch mode")
    print("   python src/generator.py --batch-size 5")
    print("   # Single mode") 
    print("   python src/generator.py --no-batch")

if __name__ == "__main__":
    asyncio.run(test_batch_mode()) 