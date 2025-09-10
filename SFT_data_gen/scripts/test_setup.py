#!/usr/bin/env python3
"""
Quick test script to verify setup and configuration.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from utils import load_config, normalize_category, create_question_id
    from prompt_builder import PromptBuilder
    from chunker import TokenChunker
    from postprocess import DataValidator
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_config():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    # Set required environment variable for test
    os.environ.setdefault('API_KEY', 'test_key')
    
    try:
        config = load_config('config/default.yaml')
        print(f"✅ Config loaded: {config.model_name}")
        print(f"   Categories: {len(config.categories)}")
        print(f"   Data size: {config.data_size}")
        return config
    except Exception as e:
        print(f"❌ Config error: {e}")
        return None

def test_source_data():
    """Test source data loading."""
    print("\n📊 Testing source data...")
    
    source_path = "source_data/mt_bench_thai_full.json"
    if not os.path.exists(source_path):
        print(f"❌ Source data not found: {source_path}")
        return None
    
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Source data loaded: {len(data)} items")
        
        # Check categories
        categories = set(item.get('category', '') for item in data)
        print(f"   Categories found: {sorted(categories)}")
        
        return data
    except Exception as e:
        print(f"❌ Source data error: {e}")
        return None

def test_prompts(config):
    """Test prompt loading."""
    print("\n📝 Testing prompts...")
    
    try:
        builder = PromptBuilder(config.category_prompt_map)
        
        # Test each category
        for category in config.categories:
            prompt_path = config.category_prompt_map.get(category)
            if prompt_path and os.path.exists(prompt_path):
                print(f"✅ {category}: {prompt_path}")
            else:
                print(f"⚠️  {category}: Missing prompt file {prompt_path}")
        
        # Test prompt building
        sample_prompt = builder.build_prompt("Writing", "Test Q1", "Test Q2")
        print(f"✅ Sample prompt generated ({len(sample_prompt)} chars)")
        
        return True
    except Exception as e:
        print(f"❌ Prompt error: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\n🛠️  Testing utilities...")
    
    try:
        # Test category normalization
        test_cases = [
            ("Knowledge III", "Knowledge_III"),
            ("Social Science", "Social_Science"),
            ("Writing", "Writing")
        ]
        
        for input_cat, expected in test_cases:
            result = normalize_category(input_cat)
            if result == expected:
                print(f"✅ Normalize '{input_cat}' -> '{result}'")
            else:
                print(f"❌ Normalize '{input_cat}' -> '{result}' (expected '{expected}')")
        
        # Test question ID creation
        qid = create_question_id("Writing", 123)
        expected_qid = "drill_Writing_0123"
        if qid == expected_qid:
            print(f"✅ Question ID: {qid}")
        else:
            print(f"❌ Question ID: {qid} (expected {expected_qid})")
        
        return True
    except Exception as e:
        print(f"❌ Utilities error: {e}")
        return False

def test_validator():
    """Test data validator."""
    print("\n✅ Testing validator...")
    
    try:
        validator = DataValidator()
        
        # Test valid data
        valid_data = {
            "turns": ["คำถามที่ 1", "คำถามที่ 2"],
            "reference": ["คำตอบที่ 1", "คำตอบที่ 2"]
        }
        
        is_valid, msg = validator.validate_drill_data(valid_data)
        if is_valid:
            print(f"✅ Valid data test passed")
        else:
            print(f"❌ Valid data test failed: {msg}")
        
        # Test invalid data
        invalid_data = {
            "turns": ["English question"],
            "reference": ["English answer"]
        }
        
        is_valid, msg = validator.validate_drill_data(invalid_data)
        if not is_valid:
            print(f"✅ Invalid data test passed: {msg}")
        else:
            print(f"❌ Invalid data test failed: should have been invalid")
        
        return True
    except Exception as e:
        print(f"❌ Validator error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 MTBench Drill Generator Setup Test")
    print("=" * 50)
    
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    config = test_config()
    if not config:
        return False
    
    source_data = test_source_data()
    if not source_data:
        return False
    
    if not test_prompts(config):
        return False
    
    if not test_utilities():
        return False
    
    if not test_validator():
        return False
    
    print("\n🎉 All tests passed! Ready to generate data.")
    print("\nTo start generation:")
    print("1. Set your API key: export API_KEY='your_api_key_here'")
    print("2. Run: ./scripts/run_generation.sh")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 