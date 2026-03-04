"""Comprehensive budget management with tracking, alerts, and enforcement."""

from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List, Callable
from datetime import datetime, timedelta, date
from enum import Enum
import json
from pathlib import Path
import threading
import time
from collections import deque


class BudgetPeriod(str, Enum):
    """Time periods for budget tracking."""
    PER_DOCUMENT = "per_document"
    PER_PAGE = "per_page"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    PROJECT = "project"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class BudgetAlert(BaseModel):
    """Budget alert notification."""
    level: AlertLevel
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    current_spend: float
    limit: float
    period: BudgetPeriod
    action_taken: Optional[str] = None


class TokenUsage(BaseModel):
    """Detailed token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def cost(self, model_pricing: Dict[str, float]) -> float:
        """Calculate cost based on model pricing."""
        prompt_cost = self.prompt_tokens * model_pricing.get('prompt_per_token', 0)
        completion_cost = self.completion_tokens * model_pricing.get('completion_per_token', 0)
        return prompt_cost + completion_cost


class UsageRecord(BaseModel):
    """Record of a single usage event."""
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    document_id: str
    strategy: str
    model: Optional[str] = None
    pages_processed: int
    tokens: Optional[TokenUsage] = None
    cost: float
    metadata: Dict = Field(default_factory=dict)


class BudgetLimits(BaseModel):
    """Comprehensive budget limits configuration."""
    per_document_cap: float = Field(0.05, ge=0)
    per_page_cap: float = Field(0.001, ge=0)
    hourly_cap: float = Field(1.0, ge=0)
    daily_cap: float = Field(5.0, ge=0)
    weekly_cap: float = Field(25.0, ge=0)
    monthly_cap: float = Field(100.0, ge=0)
    project_cap: float = Field(1000.0, ge=0)
    
    # Grace periods (seconds after hitting limit before blocking)
    grace_periods: Dict[BudgetPeriod, int] = Field(default_factory=lambda: {
        BudgetPeriod.PER_DOCUMENT: 0,
        BudgetPeriod.PER_PAGE: 0,
        BudgetPeriod.HOURLY: 300,
        BudgetPeriod.DAILY: 3600,
        BudgetPeriod.WEEKLY: 86400,
        BudgetPeriod.MONTHLY: 172800,
        BudgetPeriod.PROJECT: 259200
    })
    
    @validator('*', pre=True, each_item=True)
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError(f'Budget limit must be non-negative, got {v}')
        return v


class ModelPricing(BaseModel):
    """Pricing configuration for different models."""
    prompt_per_token: float
    completion_per_token: float
    image_cost_per_page: float = 0.0
    
    @classmethod
    def from_openrouter(cls, model_name: str) -> 'ModelPricing':
        """Get pricing for common OpenRouter models."""
        pricing_map = {
            "google/gemini-flash-1.5": {
                "prompt_per_token": 0.075 / 1_000_000,
                "completion_per_token": 0.30 / 1_000_000,
                "image_cost_per_page": 0.00014
            },
            "openai/gpt-4o-mini": {
                "prompt_per_token": 0.15 / 1_000_000,
                "completion_per_token": 0.60 / 1_000_000,
                "image_cost_per_page": 0.00014
            },
            "mistralai/pixtral-12b": {
                "prompt_per_token": 0.10 / 1_000_000,
                "completion_per_token": 0.40 / 1_000_000,
                "image_cost_per_page": 0.00014
            }
        }
        return cls(**pricing_map.get(model_name, pricing_map["google/gemini-flash-1.5"]))


class BudgetGuard:
    """Production-grade budget manager with multi-period tracking and alerts."""
    
    def __init__(self, 
                 limits: BudgetLimits,
                 model_pricing: Dict[str, ModelPricing],
                 storage_path: Optional[Path] = None,
                 alert_callbacks: Optional[List[Callable[[BudgetAlert], None]]] = None):
        
        self.limits = limits
        self.model_pricing = model_pricing
        self.storage_path = storage_path or Path(".refinery/budget")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.alert_callbacks = alert_callbacks or []
        self.lock = threading.RLock()
        
        # Usage tracking
        self.usage_history: List[UsageRecord] = []
        self.blocked_until: Dict[BudgetPeriod, datetime] = {}
        self.alerts: List[BudgetAlert] = []
        
        # Load historical data
        self._load_history()
    
    def _load_history(self):
        """Load historical usage from storage."""
        history_file = self.storage_path / "usage_history.jsonl"
        if history_file.exists():
            with open(history_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        self.usage_history.append(UsageRecord(**data))
                    except:
                        continue
    
    def _save_record(self, record: UsageRecord):
        """Save usage record to persistent storage."""
        history_file = self.storage_path / "usage_history.jsonl"
        with open(history_file, 'a') as f:
            f.write(json.dumps(record.dict()) + '\n')
        
        # Also update current session
        self.usage_history.append(record)
    
    def _get_spend_for_period(self, period: BudgetPeriod, now: datetime) -> float:
        """Calculate total spend for a specific time period."""
        if not self.usage_history:
            return 0.0
        
        if period == BudgetPeriod.PER_DOCUMENT:
            # Handled at call time
            return 0.0
        
        if period == BudgetPeriod.PER_PAGE:
            # Handled at call time
            return 0.0
        
        # Time-based periods
        period_starts = {
            BudgetPeriod.HOURLY: now - timedelta(hours=1),
            BudgetPeriod.DAILY: now - timedelta(days=1),
            BudgetPeriod.WEEKLY: now - timedelta(weeks=1),
            BudgetPeriod.MONTHLY: now - timedelta(days=30),
            BudgetPeriod.PROJECT: datetime.min
        }
        
        start_time = period_starts.get(period, datetime.min)
        
        return sum(
            record.cost 
            for record in self.usage_history 
            if record.timestamp >= start_time
        )
    
    def _is_blocked(self, period: BudgetPeriod, now: datetime) -> bool:
        """Check if a period is currently blocked."""
        if period in self.blocked_until:
            if now < self.blocked_until[period]:
                return True
            else:
                # Clear expired block
                del self.blocked_until[period]
        return False
    
    def _check_and_alert(self, 
                         period: BudgetPeriod,
                         current_spend: float,
                         limit: float,
                         now: datetime) -> bool:
        """Check limits and trigger alerts if needed."""
        if current_spend >= limit:
            # Determine alert level
            if current_spend >= limit * 1.5:
                level = AlertLevel.CRITICAL
                action = "blocked"
                # Set block with grace period
                grace = self.limits.grace_periods.get(period, 0)
                self.blocked_until[period] = now + timedelta(seconds=grace)
            elif current_spend >= limit * 1.2:
                level = AlertLevel.WARNING
                action = "warning"
            else:
                level = AlertLevel.INFO
                action = "limit_reached"
            
            alert = BudgetAlert(
                level=level,
                message=f"{period.value} budget limit reached: ${current_spend:.4f} / ${limit:.4f}",
                current_spend=current_spend,
                limit=limit,
                period=period,
                action_taken=action
            )
            
            self.alerts.append(alert)
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except:
                    pass
            
            return True
        
        return False
    
    def can_process(self, 
                   document_id: str,
                   pages: int,
                   strategy: str,
                   model: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check if a document can be processed within budget."""
        with self.lock:
            now = datetime.utcnow()
            
            # Check per-page cap if applicable
            if strategy == "vision" and pages > 0:
                page_cost = pages * self.model_pricing.get(model, {}).get('image_cost_per_page', 0.001)
                if page_cost > self.limits.per_page_cap * pages:
                    return False, f"Per-page cost ${page_cost:.4f} exceeds cap ${self.limits.per_page_cap * pages:.4f}"
            
            # Check time-based periods
            for period in [BudgetPeriod.HOURLY, BudgetPeriod.DAILY, 
                          BudgetPeriod.WEEKLY, BudgetPeriod.MONTHLY, BudgetPeriod.PROJECT]:
                
                if self._is_blocked(period, now):
                    return False, f"{period.value} budget is blocked until {self.blocked_until[period]}"
                
                current_spend = self._get_spend_for_period(period, now)
                limit = getattr(self.limits, f"{period.value}_cap", float('inf'))
                
                # Estimate additional cost (conservative)
                estimated_cost = self._estimate_cost(pages, strategy, model)
                
                if current_spend + estimated_cost > limit:
                    self._check_and_alert(period, current_spend + estimated_cost, limit, now)
                    return False, f"{period.value} budget would be exceeded (${current_spend:.4f} + ${estimated_cost:.4f} > ${limit:.4f})"
            
            return True, None
    
    def _estimate_cost(self, pages: int, strategy: str, model: Optional[str]) -> float:
        """Estimate cost for processing."""
        if strategy != "vision" or not model:
            return 0.0
        
        pricing = self.model_pricing.get(model, ModelPricing.from_openrouter(model))
        
        # Rough estimate: 750 tokens per page for vision
        tokens_per_page = 750
        total_tokens = pages * tokens_per_page
        
        prompt_cost = total_tokens * pricing.prompt_per_token
        image_cost = pages * pricing.image_cost_per_page
        
        return prompt_cost + image_cost
    
    def record_usage(self, 
                    document_id: str,
                    strategy: str,
                    pages_processed: int,
                    model: Optional[str] = None,
                    tokens: Optional[TokenUsage] = None,
                    actual_cost: Optional[float] = None,
                    metadata: Optional[Dict] = None) -> UsageRecord:
        """Record actual usage after processing."""
        with self.lock:
            # Calculate cost if not provided
            if actual_cost is None and model and tokens:
                pricing = self.model_pricing.get(model, ModelPricing.from_openrouter(model))
                actual_cost = tokens.cost(pricing.dict())
            elif actual_cost is None:
                actual_cost = 0.0
            
            record = UsageRecord(
                id=f"{document_id}_{datetime.utcnow().timestamp()}",
                document_id=document_id,
                strategy=strategy,
                model=model,
                pages_processed=pages_processed,
                tokens=tokens,
                cost=actual_cost,
                metadata=metadata or {}
            )
            
            self._save_record(record)
            
            # Check limits after recording
            now = datetime.utcnow()
            for period in [BudgetPeriod.HOURLY, BudgetPeriod.DAILY, 
                          BudgetPeriod.WEEKLY, BudgetPeriod.MONTHLY, BudgetPeriod.PROJECT]:
                
                current_spend = self._get_spend_for_period(period, now)
                limit = getattr(self.limits, f"{period.value}_cap", float('inf'))
                self._check_and_alert(period, current_spend, limit, now)
            
            return record
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget summary."""
        with self.lock:
            now = datetime.utcnow()
            summary = {
                "total_spend": sum(r.cost for r in self.usage_history),
                "total_documents": len(set(r.document_id for r in self.usage_history)),
                "total_pages": sum(r.pages_processed for r in self.usage_history),
                "by_strategy": {},
                "by_model": {},
                "period_spend": {},
                "alerts": [a.dict() for a in self.alerts[-10:]],  # Last 10 alerts
                "blocked_periods": {
                    p.value: dt.isoformat() for p, dt in self.blocked_until.items()
                }
            }
            
            # Spending by strategy
            for record in self.usage_history:
                summary["by_strategy"][record.strategy] = \
                    summary["by_strategy"].get(record.strategy, 0) + record.cost
            
            # Spending by model
            for record in self.usage_history:
                if record.model:
                    summary["by_model"][record.model] = \
                        summary["by_model"].get(record.model, 0) + record.cost
            
            # Period spending
            for period in BudgetPeriod:
                if period in [BudgetPeriod.PER_DOCUMENT, BudgetPeriod.PER_PAGE]:
                    continue
                spend = self._get_spend_for_period(period, now)
                limit = getattr(self.limits, f"{period.value}_cap", float('inf'))
                summary["period_spend"][period.value] = {
                    "spend": spend,
                    "limit": limit,
                    "remaining": max(0, limit - spend),
                    "percent_used": (spend / limit * 100) if limit > 0 else 0
                }
            
            return summary
    
    def reset_period(self, period: BudgetPeriod):
        """Reset tracking for a specific period."""
        with self.lock:
            if period in self.blocked_until:
                del self.blocked_until[period]
            
            # For time-based periods, we just clear blocks - history remains
            if period == BudgetPeriod.PROJECT:
                # For project reset, clear all history
                self.usage_history = []
                history_file = self.storage_path / "usage_history.jsonl"
                if history_file.exists():
                    history_file.unlink()