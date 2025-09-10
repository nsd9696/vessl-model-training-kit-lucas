"""Chunking utilities for efficient batch processing."""

import tiktoken
from typing import List, Dict, Any

class TokenChunker:
    """Handle chunking of data based on token limits."""
    
    def __init__(self, model_name: str, chunk_size: int):
        self.chunk_size = chunk_size
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for given text."""
        return len(self.encoding.encode(text))
    
    def create_chunks(self, source_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create chunks of source data based on chunk_size."""
        chunks = []
        current_chunk = []
        
        for item in source_data:
            current_chunk.append(item)
            
            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add remaining items as final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_by_category(
        self, 
        source_data: List[Dict[str, Any]], 
        remaining_per_category: Dict[str, int]
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        """Create chunks organized by category with needed counts."""
        category_chunks = {}
        
        for category, needed_count in remaining_per_category.items():
            if needed_count <= 0:
                continue
            
            # Filter source data by category
            category_data = [
                item for item in source_data 
                if item.get('category', '').replace(' III', '_III').replace(' ', '_').lower() == category.lower()
            ]
            
            if not category_data:
                continue
            
            # Calculate how many source items we need to generate required drills
            # We'll cycle through source data if needed
            needed_source_items = min(needed_count, len(category_data))
            selected_items = category_data[:needed_source_items]
            
            # If we need more than available, cycle through the data
            if needed_count > len(category_data):
                cycles_needed = (needed_count // len(category_data)) + 1
                extended_items = (category_data * cycles_needed)[:needed_count]
                selected_items = extended_items
            
            # Create chunks for this category
            chunks = self.create_chunks(selected_items)
            category_chunks[category] = chunks
        
        return category_chunks 