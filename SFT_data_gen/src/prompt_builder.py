"""Prompt building utilities for category-specific drill generation."""

import os
from typing import Dict, Any

class PromptBuilder:
    """Handles loading and formatting of category-specific prompts."""
    
    def __init__(self, category_prompt_map: Dict[str, str]):
        self.category_prompt_map = category_prompt_map
        self.prompts_dir = "prompts"
        
    def load_prompt_template(self, category: str) -> str:
        """Load prompt template for the given category."""
        prompt_file = self.category_prompt_map.get(category, "default_v3.txt")
        prompt_path = os.path.join(self.prompts_dir, prompt_file)
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                template = file.read().strip()
        except FileNotFoundError:
            # Fallback to default if specific category prompt not found
            default_path = os.path.join(self.prompts_dir, "default_v3.txt")
            with open(default_path, 'r', encoding='utf-8') as file:
                template = file.read().strip()
        
        return template
    
    def get_base_prompt(self, category: str) -> str:
        """Get the base prompt template for a category (for batch processing)."""
        return self.load_prompt_template(category)
    
    def build_prompt(self, category: str, target_q1: str, target_q2: str, source_question_id: str = "", num_to_generate: int = 1) -> str:
        """Build complete prompt for target question."""
        template = self.load_prompt_template(category)
        
        # Use string replacement instead of format() to avoid JSON curly brace conflicts
        # Convert all values to strings to avoid type errors
        prompt = template.replace("{category}", str(category))
        prompt = prompt.replace("{target_q1}", str(target_q1))
        prompt = prompt.replace("{target_q2}", str(target_q2))
        prompt = prompt.replace("{source_question_id}", str(source_question_id))
        prompt = prompt.replace("{num_to_generate}", str(num_to_generate))
        
        return prompt 