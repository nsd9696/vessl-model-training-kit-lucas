#!/usr/bin/env python3
"""
ThaiExam Dataset Translation Script
Translates Thai ThaiExam dataset to English while preserving Thai-related content
that's relevant to questions or answers.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm
import argparse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('config.env')  # Load from config.env file

def get_endpoint_config(model: str) -> Tuple[str, str, str, str]:
    """
    Get the appropriate endpoint configuration based on the model.
    
    Args:
        model: The model name (e.g., 'gpt-5', 'gpt-4.1', etc.)
    
    Returns:
        Tuple of (api_key, endpoint, deployment_name, api_version)
    """
    # Check if model is gpt-5
    if model.lower().startswith('gpt-5'):
        api_key = os.getenv('GPT5_API_KEY') or os.getenv('AZURE_OPENAI_API_KEY') or ''
        endpoint = os.getenv('GPT5_ENDPOINT') or os.getenv('AZURE_OPENAI_ENDPOINT') or ''
        deployment_name = os.getenv('GPT5_DEPLOYMENT_NAME') or os.getenv('DEPLOYMENT_NAME') or ''
        api_version = os.getenv('GPT5_API_VERSION', '2025-01-01-preview')
        
        if not all([api_key, endpoint, deployment_name]):
            print("⚠️  Warning: GPT-5 configuration incomplete, falling back to legacy environment variables")
            return (
                os.getenv('AZURE_OPENAI_API_KEY', ''),
                os.getenv('AZURE_OPENAI_ENDPOINT', ''),
                os.getenv('DEPLOYMENT_NAME', ''),
                os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')
            )
        
        print(f"📡 Using GPT-5 endpoint configuration")
        return (api_key, endpoint, deployment_name, api_version)
    
    # For any other model (gpt-4.1, gpt-4o, etc.), use GPT-4 endpoint
    else:
        api_key = os.getenv('GPT4_API_KEY') or ''
        endpoint = os.getenv('GPT4_ENDPOINT') or ''
        deployment_name = os.getenv('GPT4_DEPLOYMENT_NAME') or ''
        api_version = os.getenv('GPT4_API_VERSION', '2025-01-01-preview')
        
        if not all([api_key, endpoint, deployment_name]):
            print("⚠️  Warning: GPT-4 configuration not found, falling back to legacy environment variables")
            return (
                os.getenv('AZURE_OPENAI_API_KEY', ''),
                os.getenv('AZURE_OPENAI_ENDPOINT', ''),
                os.getenv('DEPLOYMENT_NAME', ''),
                os.getenv('AZURE_OPENAI_API_VERSION', '2025-01-01-preview')
            )
        
        print(f"📡 Using GPT-4 endpoint configuration for model: {model}")
        return (api_key, endpoint, deployment_name, api_version)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thaiexam_translation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom exception for content filter violations
class ContentFilterViolationError(Exception):
    """Exception raised when content filter violation occurs."""
    pass

class ThaiExamTranslator:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        endpoint: Optional[str] = None, 
        deployment_name: Optional[str] = None, 
        model: Optional[str] = None
    ):
        """Initialize the translator with Azure OpenAI client."""
        # Determine the model to use
        self.model = model or os.getenv("MODEL", "gpt-4o")
        
        # Get endpoint configuration based on model
        config_api_key, config_endpoint, config_deployment, config_api_version = get_endpoint_config(self.model)
        print("Config from config.env get_endpoint_config:")
        print(f"🔑 API Key: {config_api_key}")
        print(f"🔗 Endpoint: {config_endpoint}")
        print(f"🚀 Deployment: {config_deployment}")
        print(f"📅 API Version: {config_api_version}")
        print(f"🤖 Model: {self.model}")
        print("-"*150)

        print("Config from init:")
        print(f"🔑 API Key: {api_key}")
        print(f"🔗 Endpoint: {endpoint}")
        print(f"🚀 Deployment: {deployment_name}")
        print(f"🤖 Model: {self.model}")
        print("-"*150)
        
        # Use provided values or fall back to auto-detected configuration
        final_api_key = config_api_key or api_key
        final_endpoint = config_endpoint or endpoint
        final_deployment = config_deployment or deployment_name
        final_api_version = config_api_version
        
        if not all([final_api_key, final_endpoint, final_deployment]):
            raise ValueError(f"Missing required configuration for model {self.model}. "
                           f"Please ensure the appropriate endpoint configuration is set in config.env")
        
        # Ensure all values are strings (not None) for type safety
        final_api_key = final_api_key or ''
        final_endpoint = final_endpoint or ''
        final_deployment = final_deployment or ''
        
        self.client = AsyncAzureOpenAI(
            azure_endpoint=final_endpoint,
            api_key=final_api_key,
            api_version=final_api_version
        )
        
        self.deployment_name = final_deployment
        self.api_version = final_api_version
        
        # For Azure OpenAI, use deployment_name as the model if available
        if self.deployment_name:
            self.model = self.deployment_name
            
        print(f"🤖 Initialized ThaiExam translator with model: {self.model}")
        print(f"🔗 Using endpoint: {final_endpoint}")
        print(f"🚀 Using deployment: {final_deployment}")
        print(f"📅 Using API version: {final_api_version}")
        
        # Initialize tracking for skipped entries due to content filter
        self.skipped_entries = []
        self.skipped_count = 0
        self.skipped_file = None  # Will be set when first skipped entry occurs
        
        # Initialize tracking for failed entries after all retries
        self.failed_entries = []
        self.failed_count = 0
        self.failed_file = None  # Will be set when first failed entry occurs
        
        # Enhanced academic translation prompt for educational exam content
        self.translation_prompt = """You are a professional academic translator working on educational examination materials for university-level studies. This translation task is part of a legitimate academic research project for educational assessment and AI development.
Translate the following Thai text from ThaiExam dataset to English while following these rules:

**ACADEMIC CONTEXT**: All content is being processed for educational research purposes, including academic examinations, educational assessments, scholarly content analysis, and technical documentation. This is scholarly work conducted under academic research guidelines for educational purposes.

**TRANSLATION PRINCIPLES**:
1. **SCHOLARLY INTEGRITY**: Provide accurate academic translations maintaining the original educational value and assessment context.
2. **PRESERVE CULTURAL ELEMENTS**: If the text contains Thai language content that is relevant to the question, answer, or context, DO NOT translate it. Keep it in Thai. Keep Thai names, places, cultural references, and Thai-specific terminology in Thai when they are central to the educational assessment.
3. **ACADEMIC PRECISION**: Maintain exact meaning for technical terms, scientific concepts, and educational terminology crucial for accurate assessment.
4. **STRUCTURAL PRESERVATION**: Keep all formatting, choice labels (a, b, c, d, e), and examination structure intact.
5. **ASSESSMENT CONTEXT**: All content serves legitimate educational assessment purposes - whether it involves science, literature, social studies, or technical subjects.
6. **EXAMINATION INTEGRITY**: Preserve the exact meaning and difficulty level of examination questions for fair educational assessment.
7. **NO HALLUCINATION**: DO NOT Hallucinate any information that is not in the original text.
8. **NO TRANSLATION TO NONSENSE ENGLISH**: DO NOT Translate to nonsense English.

**EDUCATIONAL DISCLAIMER**: This is a scholarly translation for academic research and educational assessment development. The content is processed under educational research protocols for legitimate academic purposes.

Please provide a professional academic translation suitable for educational assessment:"""

    async def translate_text(self, text: str, max_retries: int = 3, entry_context: Optional[Dict[str, Any]] = None) -> str:
        """Translate a single text using Azure OpenAI."""
        for attempt in range(max_retries):
            try:
                # Use model parameter (which may be deployment_name for Azure OpenAI)
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.translation_prompt},
                        {"role": "user", "content": text}
                    ],
                }
                
                response = await self.client.chat.completions.create(**api_params)
                content = response.choices[0].message.content
                if content is None:
                    return text  # Return original text if response content is None
                return content.strip()
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                
                # Handle content filter violations specifically
                if "content_filter" in error_str.lower() or "content management policy" in error_str.lower() or "ResponsibleAIPolicyViolation" in error_str.lower():
                    logger.warning(f"Content filter violation detected for text: {text[:100]}...")
                    # This will be marked as a content filter violation and skipped at entry level
                    raise ContentFilterViolationError(f"Content filter violation: {e}")
                
                # Handle rate limiting specifically
                elif "429" in error_str or "rate limit" in error_str.lower():
                    # Extract suggested wait time from error message if available
                    suggested_wait = 35  # Default based on common Azure rate limit messages
                    if "retry after" in error_str.lower():
                        try:
                            import re
                            match = re.search(r'retry after (\d+)', error_str.lower())
                            if match:
                                suggested_wait = int(match.group(1))
                        except:
                            pass
                    
                    wait_time = min(max(suggested_wait, 2 ** attempt), 120)  # Use suggested time, cap at 2 minutes
                    logger.info(f"Rate limited, waiting {wait_time} seconds before retry (suggested: {suggested_wait}s)")
                    await asyncio.sleep(wait_time)
                else:
                    # For other errors, use exponential backoff
                    await asyncio.sleep(2 ** attempt)
                
                if attempt == max_retries - 1:
                    logger.error(f"Failed to translate after {max_retries} attempts: {e}")
                    
                    # Save failed entry immediately if we have entry context
                    if entry_context:
                        self.save_failed_entry_immediately(entry_context, str(e), text)
                    
                    return text  # Return original text if all attempts fail
        
        # This line should never be reached, but ensures all code paths return a string
        return text

    async def translate_dataset_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Translate a single ThaiExam dataset entry. Returns None if entry should be skipped due to content filter."""
        try:
            # Create a copy to avoid modifying the original
            translated_entry = entry.copy()
            
            # Translate question
            if "question" in entry:
                translated_entry["question"] = await self.translate_text(entry["question"], entry_context=entry)
            
            # Translate choices (multiple choice options)
            if "choices" in entry:
                translated_choices = {}
                for choice_key, choice_text in entry["choices"].items():
                    translated_choices[choice_key] = await self.translate_text(choice_text, entry_context=entry)
                translated_entry["choices"] = translated_choices
            
            # Translate answer explanation
            if "answer" in entry and "explanation" in entry["answer"]:
                translated_entry["answer"]["explanation"] = await self.translate_text(entry["answer"]["explanation"], entry_context=entry)
            
            # Keep other fields unchanged (category, answer label, etc.)
            return translated_entry
            
        except ContentFilterViolationError as e:
            # Log and track content filter violations
            entry_id = f'thaiexam_{len(self.skipped_entries)}'
            logger.warning(f"Content filter violation for entry {entry_id}: {e}")
            
            # Add to skipped entries with metadata
            skipped_entry = {
                "original_entry": entry,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "reason": "content_filter_violation"
            }
            self.skipped_entries.append(skipped_entry)
            self.skipped_count += 1
            
            # Save skipped entry immediately to file
            self.save_skipped_entry_immediately(skipped_entry)
            
            # Return None to indicate this entry should be skipped
            return None
            
        except Exception as e:
            logger.error(f"Error translating entry: {e}")
            return entry  # Return original entry if translation fails for other reasons

    async def translate_dataset_batch(self, entries: List[Dict[str, Any]], 
                                   semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
        """Translate a batch of dataset entries with concurrency control."""
        async def translate_with_semaphore(entry):
            async with semaphore:
                return await self.translate_dataset_entry(entry)
        
        tasks = [translate_with_semaphore(entry) for entry in entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results (skipped entries)
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Translation failed for entry {i}: {result}")
                successful_results.append(entries[i])  # Keep original entry
            elif result is not None:  # Skip None results (content filter violations)
                successful_results.append(result)
            # If result is None, it was skipped due to content filter - don't add to results
        
        return successful_results

    def load_checkpoint(self, checkpoint_file: str) -> List[Dict[str, Any]]:
        """Load a checkpoint of translated entries."""
        try:
            if not os.path.exists(checkpoint_file):
                return []
            
            entries = []
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded checkpoint: {len(entries)} entries")
            return entries
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return []

    async def translate_full_dataset(self, input_file: str, output_file: str, 
                                  batch_size: int = 10, max_concurrent: int = 5):
        """Translate the entire dataset with progress tracking and checkpointing."""
        logger.info(f"Starting translation of {input_file}")
        
        # Initialize skipped file path
        self.skipped_file = f"{output_file}.skipped.jsonl"
        
        # Initialize failed file path
        self.failed_file = f"{output_file}.failed.jsonl"
        
        # Load existing checkpoint if available
        checkpoint_file = f"{output_file}.checkpoint"
        translated_entries = self.load_checkpoint(checkpoint_file)
        
        if translated_entries:
            logger.info(f"Resuming from checkpoint: {len(translated_entries)} entries already translated")
        
        # Read the input dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_entries = len(lines)
        logger.info(f"Found {total_entries} entries to translate")
        
        # Parse JSON lines
        entries = []
        for line_num, line in enumerate(lines, 1):
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(entries)} entries")
        
        # Skip already translated entries (use a simple index-based approach since ThaiExam doesn't have question_id)
        if translated_entries:
            original_count = len(entries)
            entries = entries[len(translated_entries):]  # Skip already translated entries
            logger.info(f"Skipping {original_count - len(entries)} already translated entries")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process in batches
        # Save checkpoint after EVERY batch for maximum safety
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(entries) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} entries)")
            
            # Translate batch
            translated_batch = await self.translate_dataset_batch(batch, semaphore)
            translated_entries.extend(translated_batch)
            
            # Save checkpoint after EVERY batch for maximum safety
            self.save_checkpoint(translated_entries, checkpoint_file)
            logger.info(f"Checkpoint saved: {len(translated_entries)} entries processed")
            
            # Progress update
            progress = (len(translated_entries) / total_entries) * 100
            logger.info(f"Progress: {progress:.1f}% ({len(translated_entries)}/{total_entries})")
            
            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.01)
        
        # Save final result
        self.save_final_result(translated_entries, output_file)
        logger.info(f"Translation completed! Saved {len(translated_entries)} entries to {output_file}")

    def save_checkpoint(self, entries: List[Dict[str, Any]], checkpoint_file: str):
        """Save a checkpoint of translated entries."""
        try:
            # Create backup of existing checkpoint
            if os.path.exists(checkpoint_file):
                backup_file = f"{checkpoint_file}.backup"
                os.rename(checkpoint_file, backup_file)
                logger.info(f"Backed up existing checkpoint to {backup_file}")
            
            # Save new checkpoint
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Verify the file was written correctly
            if os.path.exists(checkpoint_file):
                actual_lines = sum(1 for _ in open(checkpoint_file, 'r', encoding='utf-8'))
                expected_lines = len(entries)
                if actual_lines != expected_lines:
                    logger.error(f"Checkpoint verification failed: expected {expected_lines} entries, got {actual_lines}")
                    # Restore backup if verification fails
                    if os.path.exists(backup_file):
                        os.rename(backup_file, checkpoint_file)
                        logger.info("Restored backup checkpoint due to verification failure")
                else:
                    logger.info(f"Checkpoint saved successfully: {actual_lines} entries verified")
                    # Remove backup if everything is OK
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
            else:
                logger.error("Checkpoint file was not created")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            # Try to restore backup
            backup_file = f"{checkpoint_file}.backup"
            if os.path.exists(backup_file):
                try:
                    os.rename(backup_file, checkpoint_file)
                    logger.info("Restored backup checkpoint after save error")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup checkpoint: {restore_error}")

    def save_skipped_entry_immediately(self, skipped_entry: Dict[str, Any]):
        """Save a single skipped entry immediately to the skipped file."""
        if not self.skipped_file:
            logger.error("Skipped file path not initialized")
            return
            
        try:
            # Append to skipped file (create if doesn't exist)
            with open(self.skipped_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(skipped_entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Skipped entry saved immediately to {self.skipped_file} (total skipped: {self.skipped_count})")
            
        except Exception as e:
            logger.error(f"Error saving skipped entry immediately: {e}")

    def save_failed_entry_immediately(self, entry: Dict[str, Any], error_message: str, failed_text: str):
        """Save a single failed entry immediately to the failed file."""
        if not self.failed_file:
            logger.error("Failed file path not initialized")
            return
            
        try:
            # Create failed entry with metadata
            failed_entry = {
                "original_entry": entry,
                "error": error_message,
                "failed_text": failed_text,
                "timestamp": datetime.now().isoformat(),
                "reason": "translation_max_retries_exceeded"
            }
            
            # Track failed entry
            self.failed_entries.append(failed_entry)
            self.failed_count += 1
            
            # Append to failed file (create if doesn't exist)
            with open(self.failed_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failed_entry, ensure_ascii=False) + '\n')
            
            logger.warning(f"Failed entry saved immediately to {self.failed_file} (total failed: {self.failed_count})")
            
        except Exception as e:
            logger.error(f"Error saving failed entry immediately: {e}")

    def save_skipped_entries(self, output_file: str):
        """Create summary report for skipped entries (entries already saved immediately)."""
        if not self.skipped_entries:
            logger.info("No entries were skipped due to content filter violations")
            return
        
        skipped_file = f"{output_file}.skipped.jsonl"
        try:
            # Entries are already saved immediately, just create the summary report
            summary_file = f"{output_file}.skipped_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"# Skipped ThaiExam Entries Summary\n\n")
                f.write(f"**Total Skipped**: {len(self.skipped_entries)}\n")
                f.write(f"**Reason**: Content filter violations\n")
                f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
                f.write(f"## Files Created\n")
                f.write(f"- **Skipped Entries**: `{skipped_file}`\n")
                f.write(f"- **Summary Report**: `{summary_file}`\n\n")
                f.write(f"## Next Steps\n")
                f.write(f"1. Review the skipped entries in `{skipped_file}`\n")
                f.write(f"2. Manually process problematic exam content if needed\n")
                f.write(f"3. Consider alternative translation approaches for sensitive educational content\n")
            
            logger.info(f"Skipped entries summary saved to {summary_file} (entries already saved to {skipped_file})")
            
        except Exception as e:
            logger.error(f"Error creating skipped entries summary: {e}")

    def save_failed_entries_summary(self, output_file: str):
        """Create summary report for failed entries (entries already saved immediately)."""
        if not self.failed_entries:
            logger.info("No entries failed after maximum retries")
            return
        
        failed_file = f"{output_file}.failed.jsonl"
        try:
            # Entries are already saved immediately, just create the summary report
            summary_file = f"{output_file}.failed_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"# Failed Entries Summary\n\n")
                f.write(f"**Total Failed**: {len(self.failed_entries)}\n")
                f.write(f"**Reason**: Translation failed after maximum retries (rate limits, API errors, etc.)\n")
                f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
                f.write(f"## Files Created\n")
                f.write(f"- **Failed Entries**: `{failed_file}`\n")
                f.write(f"- **Summary Report**: `{summary_file}`\n\n")
                f.write(f"## Error Analysis\n")
                
                # Analyze error types
                error_types = {}
                for failed_entry in self.failed_entries:
                    error = failed_entry.get('error', 'Unknown')
                    if '429' in error:
                        error_type = 'Rate Limit (429)'
                    elif '400' in error:
                        error_type = 'Bad Request (400)'
                    elif '404' in error:
                        error_type = 'Resource Not Found (404)'
                    elif '500' in error:
                        error_type = 'Server Error (500)'
                    else:
                        error_type = 'Other'
                    
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                for error_type, count in error_types.items():
                    f.write(f"- **{error_type}**: {count} entries\n")
                
                f.write(f"\n## Next Steps\n")
                f.write(f"1. Review the failed entries in `{failed_file}`\n")
                f.write(f"2. For rate limit errors: Re-run with lower concurrency\n")
                f.write(f"3. For API errors: Check endpoint configuration\n")
                f.write(f"4. For other errors: Manual review and processing\n")
            
            logger.info(f"Failed entries summary saved to {summary_file} (entries already saved to {failed_file})")
            
        except Exception as e:
            logger.error(f"Error creating failed entries summary: {e}")

    def save_final_result(self, entries: List[Dict[str, Any]], output_file: str):
        """Save the final translated dataset."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logger.info(f"Final dataset saved to {output_file}")
            
            # Save skipped entries if any
            self.save_skipped_entries(output_file)
            
            # Save failed entries summary if any
            self.save_failed_entries_summary(output_file)
            
            # Log summary
            total_processed = len(entries) + len(self.skipped_entries) + len(self.failed_entries)
            logger.info(f"Translation Summary: {len(entries)} translated, {len(self.skipped_entries)} skipped, {len(self.failed_entries)} failed, {total_processed} total")
            
        except Exception as e:
            logger.error(f"Error saving final result: {e}")

async def main():
    """Main function to run the translation."""
    parser = argparse.ArgumentParser(description='Translate ThaiExam dataset from Thai to English')
    parser.add_argument('--input', default="/root/data/vessl-ai-kt-api-models/vessl-ai-kt/generated_datasets/thaiexam_hard_training.jsonl", help='Input JSONL file path')
    parser.add_argument('--output', default="/root/data/vessl-ai-kt-api-models/vessl-ai-kt/generated_datasets/thaiexam_hard_training_english.jsonl", help='Output JSONL file path')
    parser.add_argument('--batch-size', type=int, default=int(os.getenv('BATCH_SIZE', '10')), help='Batch size for processing')
    parser.add_argument('--max-concurrent', type=int, default=int(os.getenv('MAX_CONCURRENT', '5')), help='Maximum concurrent API calls')
    parser.add_argument('--api-key', default=os.getenv('AZURE_OPENAI_API_KEY'), help='Azure OpenAI API key')
    parser.add_argument('--endpoint', default=os.getenv('AZURE_OPENAI_ENDPOINT'), help='Azure OpenAI endpoint')
    parser.add_argument('--deployment-name', default=os.getenv('DEPLOYMENT_NAME'), help='Azure OpenAI deployment name')
    parser.add_argument('--model', default=os.getenv('MODEL', 'gpt-4o'), help='Azure OpenAI model to use (defaults to env var MODEL)')
    
    args = parser.parse_args()
    
    # Note: API key and endpoint validation is now handled in ThaiExamTranslator constructor
    # The constructor will automatically select the appropriate endpoint based on the model
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize translator
    translator = ThaiExamTranslator(
        api_key=args.api_key,
        endpoint=args.endpoint,
        deployment_name=args.deployment_name,
        model=args.model
    )
    
    # Start translation
    start_time = time.time()
    try:
        await translator.translate_full_dataset(
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Translation completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
