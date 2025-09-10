"""Token usage tracking and cost estimation."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    cached_calls: int = 0  # Track cached calls separately
    
    def update(self, response_usage: Any) -> None:
        """Update token counts from API response."""
        if hasattr(response_usage, 'prompt_tokens'):
            self.prompt_tokens += response_usage.prompt_tokens
        if hasattr(response_usage, 'completion_tokens'):
            self.completion_tokens += response_usage.completion_tokens
        if hasattr(response_usage, 'total_tokens'):
            self.total_tokens += response_usage.total_tokens
        self.api_calls += 1
    
    def update_cached(self) -> None:
        """Update cached call count without adding token usage."""
        self.cached_calls += 1
    
    def get_estimated_cost(self, model_name: str) -> float:
        """Estimate cost based on model pricing."""
        # Pricing per 1M tokens (as of 2024)
        pricing = {
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
            "gpt-4o": {"prompt": 5.00, "completion": 15.00},
            "gpt-4": {"prompt": 30.00, "completion": 60.00},
            "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        }
        
        model_pricing = pricing.get(model_name, pricing["gpt-4o-mini"])
        
        prompt_cost = (self.prompt_tokens / 1_000_000) * model_pricing["prompt"]
        completion_cost = (self.completion_tokens / 1_000_000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/reporting."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "cached_calls": self.cached_calls
        }

class TokenTracker:
    """Track token usage across generation process."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.total_usage = TokenUsage()
        self.category_usage: Dict[str, TokenUsage] = {}
        self.logger = logging.getLogger(__name__)
    
    def track_api_call(self, category: str, response: Any, is_cached: bool = False) -> None:
        """Track token usage from API response."""
        if is_cached:
            # For cached responses, only count the call but not the tokens
            self.total_usage.update_cached()
            if category not in self.category_usage:
                self.category_usage[category] = TokenUsage()
            self.category_usage[category].update_cached()
            return
        
        if not hasattr(response, 'usage') or not response.usage:
            return
        
        # Update total usage
        self.total_usage.update(response.usage)
        
        # Update category usage
        if category not in self.category_usage:
            self.category_usage[category] = TokenUsage()
        self.category_usage[category].update(response.usage)
    
    def log_progress(self, completed: int, total: int) -> None:
        """Log current progress with token usage."""
        cost = self.total_usage.get_estimated_cost(self.model_name)
        
        total_calls = self.total_usage.api_calls + self.total_usage.cached_calls
        cached_ratio = (self.total_usage.cached_calls / total_calls * 100) if total_calls > 0 else 0
        
        self.logger.info(
            f"Progress: {completed}/{total} | "
            f"Tokens: {self.total_usage.total_tokens:,} | "
            f"API Calls: {self.total_usage.api_calls} | "
            f"Cached: {self.total_usage.cached_calls} ({cached_ratio:.1f}%) | "
            f"Est. Cost: ${cost:.4f}"
        )
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        total_cost = self.total_usage.get_estimated_cost(self.model_name)
        
        total_calls = self.total_usage.api_calls + self.total_usage.cached_calls
        cached_ratio = (self.total_usage.cached_calls / total_calls * 100) if total_calls > 0 else 0
        
        summary = {
            "model": self.model_name,
            "total_usage": self.total_usage.to_dict(),
            "total_cost_usd": round(total_cost, 4),
            "cache_efficiency": f"{cached_ratio:.1f}%",
            "category_breakdown": {}
        }
        
        for category, usage in self.category_usage.items():
            category_cost = usage.get_estimated_cost(self.model_name)
            summary["category_breakdown"][category] = {
                **usage.to_dict(),
                "cost_usd": round(category_cost, 4)
            }
        
        return summary
    
    def print_final_report(self) -> None:
        """Print detailed final usage report."""
        summary = self.get_summary_report()
        
        print("\n" + "="*60)
        print("🎯 GENERATION COMPLETE - TOKEN USAGE REPORT")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Total API Calls: {self.total_usage.api_calls:,}")
        print(f"Cached Calls: {self.total_usage.cached_calls:,}")
        print(f"Cache Efficiency: {summary['cache_efficiency']}")
        print(f"Total Tokens: {self.total_usage.total_tokens:,}")
        print(f"  ├─ Prompt Tokens: {self.total_usage.prompt_tokens:,}")
        print(f"  └─ Completion Tokens: {self.total_usage.completion_tokens:,}")
        print(f"Estimated Cost: ${summary['total_cost_usd']:.4f}")
        
        if self.category_usage:
            print("\n📊 Category Breakdown:")
            for category, data in summary["category_breakdown"].items():
                print(f"  {category}:")
                print(f"    ├─ API Calls: {data['api_calls']}")
                print(f"    ├─ Cached Calls: {data['cached_calls']}")
                print(f"    ├─ Tokens: {data['total_tokens']:,}")
                print(f"    └─ Cost: ${data['cost_usd']:.4f}")
        
        print("="*60) 