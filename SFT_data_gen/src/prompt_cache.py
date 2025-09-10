"""Prompt caching system for API calls."""

import json
import hashlib
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

class PromptCache:
    """Caches API responses based on prompt hash."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "prompt_cache.json"
        self.stats_file = self.cache_dir / "cache_stats.json"
        
        # Load existing cache
        self.cache = self._load_cache()
        self.stats = self._load_stats()
        
        self.logger = logging.getLogger(__name__)
        
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load cache statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {"hits": 0, "misses": 0, "total_calls": 0}
        return {"hits": 0, "misses": 0, "total_calls": 0}
    
    def _save_stats(self):
        """Save cache statistics."""
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
    
    def _generate_cache_key(self, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate cache key from request parameters."""
        # Create a consistent hash from request parameters
        request_data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Convert to JSON string with sorted keys for consistency
        request_str = json.dumps(request_data, sort_keys=True, ensure_ascii=False)
        
        # Generate MD5 hash
        return hashlib.md5(request_str.encode('utf-8')).hexdigest()
    
    def get_cached_response(self, model: str, prompt: str, temperature: float, max_tokens: int) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        cache_key = self._generate_cache_key(model, prompt, temperature, max_tokens)
        
        if cache_key in self.cache:
            self.stats["hits"] += 1
            self.stats["total_calls"] += 1
            self.logger.debug(f"Cache HIT for key: {cache_key[:8]}...")
            
            # Convert cached response back to a format that mimics ChatCompletion
            cached_entry = self.cache[cache_key]
            serializable_response = cached_entry["response"]
            
            # Create a mock response object that has the same interface
            class MockResponse:
                def __init__(self, data):
                    self.choices = []
                    for choice_data in data["choices"]:
                        choice = type('Choice', (), {})()
                        choice.message = type('Message', (), {})()
                        choice.message.content = choice_data["message"]["content"]
                        choice.message.role = choice_data["message"]["role"]
                        choice.finish_reason = choice_data["finish_reason"]
                        choice.index = choice_data["index"]
                        self.choices.append(choice)
                    
                    if data["usage"]:
                        self.usage = type('Usage', (), {})()
                        self.usage.prompt_tokens = data["usage"]["prompt_tokens"]
                        self.usage.completion_tokens = data["usage"]["completion_tokens"]
                        self.usage.total_tokens = data["usage"]["total_tokens"]
                    else:
                        self.usage = None
                    
                    self.model = data["model"]
                    self.id = data["id"]
                    self.object = data["object"]
                    self.created = data["created"]
            
            return {"response": MockResponse(serializable_response)}
        
        self.stats["misses"] += 1
        self.stats["total_calls"] += 1
        self.logger.debug(f"Cache MISS for key: {cache_key[:8]}...")
        return None
    
    def cache_response(self, model: str, prompt: str, temperature: float, max_tokens: int, response):
        """Cache API response."""
        cache_key = self._generate_cache_key(model, prompt, temperature, max_tokens)
        
        # Convert ChatCompletion object to serializable format
        serializable_response = {
            "choices": [
                {
                    "message": {
                        "content": choice.message.content,
                        "role": choice.message.role
                    },
                    "finish_reason": choice.finish_reason,
                    "index": choice.index
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            "model": response.model,
            "id": response.id,
            "object": response.object,
            "created": response.created
        }
        
        # Store response with metadata
        cache_entry = {
            "response": serializable_response,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt)
        }
        
        self.cache[cache_key] = cache_entry
        self._save_cache()
        self.logger.debug(f"Cached response for key: {cache_key[:8]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (self.stats["hits"] / self.stats["total_calls"]) * 100 if self.stats["total_calls"] > 0 else 0
        return {
            **self.stats,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self.cache = {}
        self._save_cache()
        self.stats = {"hits": 0, "misses": 0, "total_calls": 0}
        self._save_stats()
        self.logger.info("Cache cleared")
    
    def cleanup_old_entries(self, days_old: int = 30):
        """Remove cache entries older than specified days."""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        keys_to_remove = []
        for key, entry in self.cache.items():
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if entry_time < cutoff_date:
                    keys_to_remove.append(key)
            except (KeyError, ValueError):
                # Remove entries with invalid timestamps
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        if keys_to_remove:
            self._save_cache()
            self.logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
    
    def save_stats(self):
        """Save current statistics."""
        self._save_stats()
        
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_stats()
        print(f"\n💾 Prompt Cache Statistics:")
        print(f"   Cache Size: {stats['cache_size']} entries")
        print(f"   Total Calls: {stats['total_calls']}")
        print(f"   Cache Hits: {stats['hits']}")
        print(f"   Cache Misses: {stats['misses']}")
        print(f"   Hit Rate: {stats['hit_rate']}") 