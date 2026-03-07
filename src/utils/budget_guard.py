"""
Budget guard for cost management across extraction strategies.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path


class BudgetGuard:
    """
    Guard for managing extraction costs and preventing budget overruns.
    """
    
    def __init__(
        self,
        daily_budget_usd: float = 10.0,
        per_document_budget_usd: float = 0.50,
        warning_threshold: float = 0.8
    ):
        """
        Initialize budget guard.
        
        Args:
            daily_budget_usd: Maximum daily spend in USD
            per_document_budget_usd: Maximum per-document spend
            warning_threshold: Threshold for warning (0.0-1.0)
        """
        self.daily_budget = daily_budget_usd
        self.per_document_budget = per_document_budget_usd
        self.warning_threshold = warning_threshold
        
        # Cost models (simplified estimates)
        self.strategy_costs = {
            "fast_text": 0.001,  # $0.001 per page
            "layout_aware": 0.01,  # $0.01 per page
            "vision_augmented": 0.05,  # $0.05 per page
            "vlm_per_request": 0.002,  # $0.002 per VLM request
        }
        
        # Token costs for OpenRouter (Gemma 3 27B)
        self.token_costs = {
            "input": 0.0000005,  # $0.50 per 1M tokens
            "output": 0.0000015,  # $1.50 per 1M tokens
        }
        
        self.daily_spend = self._load_daily_spend()
    
    def _load_daily_spend(self) -> float:
        """Load today's spend from persistent storage"""
        spend_file = Path(".refinery/budget/daily_spend.json")
        if spend_file.exists():
            try:
                with open(spend_file) as f:
                    data = json.load(f)
                    # Reset if not today
                    if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
                        return data.get("spend", 0.0)
            except:
                pass
        
        return 0.0
    
    def _save_daily_spend(self, spend: float):
        """Save daily spend to persistent storage"""
        spend_file = Path(".refinery/budget/daily_spend.json")
        spend_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(spend_file, "w") as f:
            json.dump({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "spend": spend
            }, f)
    
    def estimate_cost(
        self,
        strategy: str,
        page_count: int,
        estimated_tokens: Optional[int] = None
    ) -> float:
        """
        Estimate cost for extraction.
        
        Args:
            strategy: Extraction strategy
            page_count: Number of pages
            estimated_tokens: Estimated tokens (for VLM)
            
        Returns:
            Estimated cost in USD
        """
        base_cost = self.strategy_costs.get(strategy, 0.0) * page_count
        
        if strategy == "vision_augmented" and estimated_tokens:
            token_cost = (
                estimated_tokens * self.token_costs["input"] +
                estimated_tokens * self.token_costs["output"] // 10  # Assume 10% output
            )
            base_cost += token_cost
        
        return base_cost
    
    def check_budget(
        self,
        strategy: str,
        page_count: int,
        estimated_tokens: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Check if operation is within budget.
        
        Returns:
            Dict with budget status information
        """
        estimated_cost = self.estimate_cost(strategy, page_count, estimated_tokens)
        new_total = self.daily_spend + estimated_cost
        
        status = {
            "approved": True,
            "estimated_cost": estimated_cost,
            "daily_spend_after": new_total,
            "daily_budget": self.daily_budget,
            "per_document_cost": estimated_cost,
            "per_document_budget": self.per_document_budget,
            "warnings": []
        }
        
        # Check daily budget
        if new_total > self.daily_budget:
            status["approved"] = False
            status["warnings"].append(
                f"Daily budget exceeded: ${new_total:.3f} > ${self.daily_budget:.3f}"
            )
        elif new_total > self.daily_budget * self.warning_threshold:
            status["warnings"].append(
                f"Daily budget warning: ${new_total:.3f} > "
                f"${self.daily_budget * self.warning_threshold:.3f}"
            )
        
        # Check per-document budget
        if estimated_cost > self.per_document_budget:
            status["warnings"].append(
                f"Per-document budget exceeded: ${estimated_cost:.3f} > "
                f"${self.per_document_budget:.3f}"
            )
        
        return status
    
    def record_spend(self, cost_usd: float):
        """Record actual spend"""
        self.daily_spend += cost_usd
        self._save_daily_spend(self.daily_spend)
    
    def get_strategy_recommendation(
        self,
        profile_recommendation: str,
        page_count: int
    ) -> str:
        """
        Get budget-aware strategy recommendation.
        
        Args:
            profile_recommendation: Strategy recommended by profile
            page_count: Number of pages
            
        Returns:
            Recommended strategy (may be downgraded if budget constrained)
        """
        # Check if we can afford the recommended strategy
        cost_check = self.check_budget(profile_recommendation, page_count)
        
        if cost_check["approved"]:
            return profile_recommendation
        
        # Try downgrading
        downgrade_path = {
            "vision_augmented": "layout_aware",
            "layout_aware": "fast_text",
            "fast_text": "fast_text"  # Already at minimum
        }
        
        downgraded = downgrade_path.get(profile_recommendation, "fast_text")
        
        if downgraded != profile_recommendation:
            # Check if downgraded strategy is within budget
            downgrade_check = self.check_budget(downgraded, page_count)
            if downgrade_check["approved"]:
                return downgraded
        
        return "fast_text"  # Fallback to cheapest