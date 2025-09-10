"""Utility functions for MTBench drill generation."""

import json
import logging
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

def get_project_root() -> str:
    """Get the project root directory (MTbenchtic_data_v4for10000)."""
    # Get the directory containing this utils.py file (src/)
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent
    # Project root is parent of src/
    project_root = src_dir.parent
    return str(project_root)

def resolve_path(path: str) -> str:
    """Convert relative path to absolute path based on project root."""
    if os.path.isabs(path):
        return path
    
    project_root = get_project_root()
    return os.path.join(project_root, path)

@dataclass
class Config:
    """Configuration dataclass."""
    model_name: str
    api_key: str
    api_client: str
    temperature: float
    max_tokens: int
    source_data_path: str
    output_dir: str
    data_size: int
    chunk_size: int
    batch_size: int
    max_retries: int
    retry_delay: float
    categories: List[str]
    category_easy_prompt_map: Dict[str, str]
    category_hard_prompt_map: Dict[str, str]
    log_level: str
    save_frequency: int = 100
    # Quality settings
    quality_mode: bool = True
    validation_strict: bool = True
    min_turn_length: int = 50
    min_reference_length: int = 100
    quality_keywords_required: int = 2
    # Cache settings
    enable_cache: bool = True
    cache_dir: str = "cache"
    cache_cleanup_days: int = 30
    cache_stats_interval: int = 100
    # Batch settings
    enable_batch_mode: bool = True
    max_batch_size: int = 8
    min_batch_size: int = 3
    batch_fallback_single: bool = True
    # Log settings
    log_file: str = "generation.log"

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file with environment variable substitution."""
    # Resolve config path
    config_path = resolve_path(config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace environment variables
    import re
    pattern = r'\$\{([^}]+)\}'
    
    def replace_env_var(match):
        env_expr = match.group(1)
        if ':-' in env_expr:
            var_name, default_value = env_expr.split(':-', 1)
            return os.getenv(var_name, default_value)
        else:
            return os.getenv(env_expr, '')
    
    content = re.sub(pattern, replace_env_var, content)
    config_dict = yaml.safe_load(content)
    
    # Resolve relative paths to absolute paths
    config_dict['source_data_path'] = resolve_path(config_dict['source_data_path'])
    config_dict['output_dir'] = resolve_path(config_dict['output_dir'])
    
    # Resolve category prompt paths
    for category, prompt_path in config_dict['category_easy_prompt_map'].items():
        config_dict['category_easy_prompt_map'][category] = resolve_path(prompt_path)
    for category, prompt_path in config_dict['category_hard_prompt_map'].items():
        config_dict['category_hard_prompt_map'][category] = resolve_path(prompt_path)
    
    return Config(**config_dict)

def setup_logging(level: str) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('generation.log')
        ]
    )
    return logging.getLogger(__name__)

def normalize_category(category: str) -> str:
    """Normalize category name for question ID generation."""
    return category.replace(' III', '_III').replace(' ', '_')

def create_question_id(category: str, index: int) -> str:
    """Create question ID in format: drill_{category}_{index:04d}"""
    normalized_cat = normalize_category(category)
    return f"drill_{normalized_cat}_{index:04d}"

def load_existing_data(output_path: str) -> List[Dict[str, Any]]:
    """Load existing generated data if file exists."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logging.warning(f"Could not load existing data: {e}")
    return []

def save_data_item(output_path: str, item: Dict[str, Any]) -> None:
    """Save a single data item to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_category_counts(data: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count existing data by category."""
    counts = {}
    for item in data:
        category = item.get('category', '')
        # Normalize category name
        normalized_cat = category.replace(' III', '_III').replace(' ', '_')
        counts[normalized_cat] = counts.get(normalized_cat, 0) + 1
    return counts

def calculate_remaining_per_category(
    existing_counts: Dict[str, int], 
    total_size: int, 
    categories: List[str]
) -> Dict[str, int]:
    """Calculate how many items are still needed per category."""
    per_category = total_size // len(categories)
    remainder = total_size % len(categories)
    
    remaining = {}
    for i, category in enumerate(categories):
        target = per_category + (1 if i < remainder else 0)
        current = existing_counts.get(category, 0)
        remaining[category] = max(0, target - current)
    
    return remaining

def create_output_path(output_dir: str, file_name: str) -> str:
    """Create output directory and return JSONL file path (fixed filename for resumable generation)."""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, file_name)

def create_backup_if_exists(output_path: str) -> None:
    """Create backup of existing file before starting new generation."""
    if os.path.exists(output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.replace('.jsonl', f'_backup_{timestamp}.jsonl')
        import shutil
        shutil.copy2(output_path, backup_path)
        logging.getLogger(__name__).info(f"📁 Backup created: {backup_path}") 