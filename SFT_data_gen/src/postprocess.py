"""Data post-processing and validation utilities."""

from ast import Str
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, ValidationError

# Optional language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

class ThaiExamData(BaseModel):
    """Pydantic model for ThaiExam data validation."""
    question: str
    choices: dict[str, str]
    answer: dict[str, str]

class MTBenchData(BaseModel):
    """Pydantic model for MTBench data validation."""
    category: str
    turns: list[str]
    reference: list[str] | list[list[str]]  # Can be either flat list of strings or nested list of strings


class DataValidator:
    """Validate and post-process drill data."""
    
    def __init__(self):
        self.thai_pattern = re.compile(r'[\u0e00-\u0e7f]')  # Thai Unicode range
        self.vietnamese_pattern = re.compile(r'[\u1ea0-\u1ef9]')  # Vietnamese Unicode range
        self.english_pattern = re.compile(r'[a-zA-Z]')  # English Unicode range
        self.japanese_pattern = re.compile(r'[\u3040-\u30ff]')  # Japanese Unicode range
        self.korean_pattern = re.compile(r'[\uac00-\ud7af]')  # Korean Unicode range
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')  # Chinese Unicode range
        self.indonesian_pattern = re.compile(r'[\u0600-\u06ff]')  # Indonesian Unicode range
        self.malay_pattern = re.compile(r'[\u0600-\u06ff]')  # Malay Unicode range
        self.tagalog_pattern = re.compile(r'[\u1700-\u171f]')  # Tagalog Unicode range
        self.filipino_pattern = re.compile(r'[\u1700-\u171f]')  # Filipino Unicode range
    
    def is_thai_text(self, text: str) -> bool:
        """Check if text contains Thai characters."""
        if not text:
            return False
        return bool(self.thai_pattern.search(text))
    
    def is_vietnamese_text(self, text: str) -> bool:
        """Check if text contains Vietnamese characters."""
        if not text:
            return False
        return bool(self.vietnamese_pattern.search(text))
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        if not LANGDETECT_AVAILABLE:
            return "unknown"
        
        try:
            return detect(text)
        except:
            return "unknown"
    
    def validate_drill_data(self, data: Dict[str, Any], dataset: str) -> Tuple[bool, str]:
        """Validate drill data structure and content."""
        try:
            # Validate structure with Pydantic
            if dataset == "thaiexam":
                thaiexam_drill = ThaiExamData(**data)

                if len(thaiexam_drill.choices) != 5:
                    return False, f"Expected 5 choices, got {len(thaiexam_drill.choices)}"
                
                if thaiexam_drill.answer['label'] not in thaiexam_drill.choices.keys():
                    return False, f"Answer {thaiexam_drill.answer['label']} is not in choices"
                
                # Check for Thai language - only check turns
                all_text = " ".join(thaiexam_drill.question)
            elif "mtbench" in dataset:
                mtbench_drill = MTBenchData(**data)

                # Check turns length
                if len(mtbench_drill.turns) != 2:
                    return False, f"Expected 2 turns, got {len(mtbench_drill.turns)}"
                
                # Check for empty content
                for i, turn in enumerate(mtbench_drill.turns):
                    if not turn.strip():
                        return False, f"Turn {i+1} is empty"

                all_text = " ".join(mtbench_drill.turns)

            
            if not self.is_thai_text(all_text):
                return False, "Text does not contain Thai characters"

            else:
                return False, "Invalid language"
            
            if dataset == "mtbench":
                for i, text in enumerate(mtbench_drill.turns):
                    if len(text) < 20:  # Minimum reasonable length
                        return False, f"Turn {i+1} is too short"        
            return True, "Valid"
            
        except ValidationError as e:
            return False, f"Validation error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted patterns."""
        # Remove <think> tags and content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def validate_drill_structure(self, data: Dict[str, Any], dataset: str) -> bool:
        """Validate basic drill structure without detailed validation."""
        if not isinstance(data, dict):
            return False
        
        if dataset == "thaiexam":
            if 'question' not in data:
                return False
            if 'choices' not in data:
                return False
            if len(data['choices']) != 5:
                return False
        elif "mtbench" in dataset:
            if 'reference' not in data:
                return False
            if data['category'] in ["roleplay", "humanities", "writing"]:
                if len(data['reference']) != 3:
                    return False
            else:
                if len(data['reference']) != 2:
                    return False
                return False
            # Check required fields
            if 'turns' not in data:
                return False
            
            # Check basic structure
            if not isinstance(data['turns'], list):
                return False
            
            # Check basic length requirements
            if len(data['turns']) != 2:
                return False
            
            # Check for empty strings
            for turn in data['turns']:
                if not isinstance(turn, str) or not turn.strip():
                    return False
            
        return True
    
    def validate_drill_quality(self, drill_data: Dict[str, Any], dataset: str) -> bool:
        """Validate drill quality including Thai language and content requirements."""
        is_valid, error_msg = self.validate_drill_data(drill_data, dataset)
        return is_valid
    