"""Batch generator for multiple drill problems at once."""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import aiohttp
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from utils import create_question_id
from postprocess import DataValidator

@dataclass
class BatchRequest:
    """Represents a batch of problems to generate."""
    category: str
    source_items: List[Dict[str, Any]]
    start_index: int
    count: int

class BatchDrillGenerator:
    """Generates multiple drill problems in a single API call."""
    
    def __init__(self, dataset, prompt_builder, config, token_tracker, api_client = "openai", client = None, url = None, headers = None, cache=None):
        self.dataset = dataset
        self.prompt_builder = prompt_builder
        self.client = client
        self.url = url
        self.api_client = api_client
        self.headers = headers
        self.config = config
        self.token_tracker = token_tracker
        self.cache = cache
        self.validator = DataValidator()
        self.logger = logging.getLogger(__name__)
        
        # Set up Jinja2 environment for template rendering
        template_dir = Path(__file__).parent.parent / "templates"
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
    def build_batch_prompt(self, batch_request: BatchRequest) -> str:
        """Build prompt for generating multiple problems at once."""
        category = batch_request.category
        source_items = batch_request.source_items
        count = batch_request.count
        
        # Load base prompt template
        base_prompt = self.prompt_builder.get_base_prompt(category)
        
        # Build multiple target questions section
        target_questions = []
        if "mtbench" in self.dataset:
            for i, item in enumerate(source_items[:count]):
                turns = item.get('turns', [])
                references = item.get('references', [])
                if len(turns) >= 2:
                    target_questions.append(f"""
**Target Question {i+1}:**
- **Source ID:** {item.get('question_id', f'unknown_{i}')}
- **Turn 1:** "{turns[0]}"
- **Turn 2:** "{turns[1]}"
""")        
            # Render batch instructions using Jinja2 template
            template = self.jinja_env.get_template("mtbench_batch_instructions.j2")
            batch_instructions = template.render(
                count=count,
                target_questions=target_questions,
            )
            
            # Combine base prompt with batch instructions
            full_prompt = base_prompt.replace(
                "### **Drill Design Instructions**",
                batch_instructions
            )
            
            # Remove single-question placeholders
            full_prompt = full_prompt.replace("{target_q1}", "See target questions above")
            full_prompt = full_prompt.replace("{target_q2}", "See target questions above") 
            full_prompt = full_prompt.replace("{source_question_id}", "See source IDs above")
            full_prompt = full_prompt.replace("{category}", category)
        elif "thaiexam" in self.dataset:
            for i, item in enumerate(source_items[:count]):
                question = item.get('question', "")
                choices = item.get('choices', {})
                if len(question) >= 1:
                    target_questions.append(f"""
**Target Question {i+1}:**
- **Question:** "{question}"
- **Choices:** {choices}
""")
                template =self.jinja_env.get_template("thaiexam_batch_instructions.j2")
                batch_instructions = template.render(
                    count=count,
                    target_questions=target_questions,
                )
                
            full_prompt = base_prompt.replace(
            "### **Drill Design Instructions**",
            batch_instructions
        )
        
            # Remove single-question placeholders
            full_prompt = full_prompt.replace("{question}", "See target questions above")
            full_prompt = full_prompt.replace("{choices}", "See target choices above") 
            full_prompt = full_prompt.replace("{category}", category)
        
        return full_prompt
    
    def parse_batch_response(self, response_text: str, batch_request: BatchRequest, dataset: str) -> List[Dict[str, Any]]:
        """Parse API response containing multiple drill problems."""
        try:
            # Try to extract JSON array from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                self.logger.warning("No JSON array found in response")
                return []
            
            json_str = response_text[json_start:json_end]
            drill_list = json.loads(json_str)
            
            if not isinstance(drill_list, list):
                self.logger.warning("Response is not a JSON array")
                return []
            
            # Process each drill in the batch
            valid_drills = []
            for i, drill_data in enumerate(drill_list):
                if not isinstance(drill_data, dict):
                    self.logger.warning(f"Drill {i+1} is not a valid object")
                    continue
                
                # Validate drill structure
                if not self.validator.validate_drill_structure(drill_data, dataset):
                    self.logger.warning(f"Drill {i+1} has invalid structure")
                    continue
                
                # Create complete drill item
                question_id = create_question_id(batch_request.category, batch_request.start_index + i)
                
                if "mtbench" in dataset:
                    complete_drill = {
                        "question_id": question_id,
                        "category": batch_request.category.replace('_', ' '),
                        "turns": drill_data["turns"],
                        "reference": drill_data["reference"],
                        "source_question_id": batch_request.source_items[i].get('question_id', f'unknown_{i}') if i < len(batch_request.source_items) else f'unknown_{i}'
                    }
                elif "thaiexam" in dataset:
                    complete_drill = {
                        "category": batch_request.category.replace('_', ' '),
                        "question": drill_data["question"],
                        "choices": drill_data["choices"],
                        "answer": drill_data["answer"]
                    }
                
                # Additional quality validation
                if self.validator.validate_drill_quality(complete_drill, dataset):
                    valid_drills.append(complete_drill)
                else:
                    self.logger.warning(f"Drill {i+1} failed quality validation")
            
            self.logger.info(f"Successfully parsed {len(valid_drills)}/{len(drill_list)} drills from batch")
            return valid_drills
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in batch response: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error parsing batch response: {e}")
            return []
    
    async def generate_batch(self, batch_request: BatchRequest) -> Tuple[List[Dict[str, Any]], int]:
        """
        Generate multiple drill problems in a single API call.
        
        Returns:
            Tuple of (successful_drills, failed_count)
        """
        prompt = self.build_batch_prompt(batch_request)
        category = batch_request.category
        
        # Try batch generation with retries
        for attempt in range(self.config.max_retries):
            try:
                if self.api_client == "azure":  
                    # Map model names to deployment names for Azure
                    deployment_map = {
                        "gpt-4o": "gpt4o-prod",
                        "gpt-4o-mini": "gpt4o-mini-prod",
                        "gpt-4": "gpt4-prod",
                        "gpt-4-turbo": "gpt4-turbo-prod",
                        "gpt-3.5-turbo": "gpt35-turbo-prod",
                        "gpt-5-2025-08-07": "gpt-5-2025-08-07",
                        "gpt-5": "gpt-5"
                    }
                    deployment_name = deployment_map.get(self.config.model_name, self.config.model_name)
                    
                    chat_completion_kwargs = {
                        "model": deployment_name,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if 'gpt-5' in deployment_name:
                        chat_completion_kwargs['reasoning_effort'] = 'high'

                    async with aiohttp.ClientSession() as session:
                        async with session.post(self.url, headers=self.headers, json=chat_completion_kwargs) as result:
                            if result.status != 200:
                                error_text = await result.text()
                                raise Exception(f"API request failed with status {result.status}: {error_text}")
                            
                            if 'application/json' not in result.headers.get('Content-Type', ''):
                                error_text = await result.text()
                                raise Exception(f"Unexpected response type: {result.headers.get('Content-Type')}\nResponse: {error_text}")
                                
                            response = await result.json()
                
                elif self.api_client == "openai":
                    chat_completion_kwargs = {
                        "model": self.config.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if 'gpt-5' in self.config.model_name:
                        chat_completion_kwargs['reasoning_effort'] = 'low'
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(self.url, headers=self.headers, json=chat_completion_kwargs) as result:
                            if result.status != 200:
                                error_text = await result.text()
                                raise Exception(f"API request failed with status {result.status}: {error_text}")
                            
                            if 'application/json' not in result.headers.get('Content-Type', ''):
                                error_text = await result.text()
                                raise Exception(f"Unexpected response type: {result.headers.get('Content-Type')}\nResponse: {error_text}")
                                
                            response = await result.json()
                            # Track as real API call
                
                self.token_tracker.track_api_call(category, response)
                                
                # Parse the batch response
                raw_content = response['choices'][0]['message']['content']
                valid_drills = self.parse_batch_response(raw_content, batch_request, self.dataset)
                
                success_count = len(valid_drills)
                failed_count = batch_request.count - success_count
                
                if success_count > 0:
                    self.logger.info(f"✅ Batch {category}: {success_count}/{batch_request.count} drills successful")
                    return valid_drills, failed_count
                else:
                    self.logger.warning(f"❌ Batch {category} attempt {attempt + 1}: No valid drills generated")
                
            except Exception as e:
                self.logger.error(f"❌ Batch {category} attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        self.logger.error(f"❌ Batch {category} failed after {self.config.max_retries} attempts")
        return [], batch_request.count
    
    def should_use_batch_mode(self, remaining_count: int) -> bool:
        """Determine if batch mode should be used based on remaining count."""
        min_batch_size = getattr(self.config, 'min_batch_size', 3)
        return remaining_count >= min_batch_size
    
    def calculate_batch_size(self, remaining_count: int) -> int:
        """Calculate optimal batch size based on remaining count and configuration."""
        max_batch_size = getattr(self.config, 'max_batch_size', 10)
        min_batch_size = getattr(self.config, 'min_batch_size', 3)
        
        if remaining_count <= min_batch_size:
            return 1  # Use single mode
        
        # Use smaller batches for better error handling
        # Remove hard-coded 8 limit to allow larger batches
        return min(max_batch_size, remaining_count) 