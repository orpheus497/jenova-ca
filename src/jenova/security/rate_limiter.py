"""
JENOVA Rate Limiter - Token bucket rate limiting per user.

Implements per-user rate limiting to prevent resource exhaustion and abuse.

Fixes: BUG-M4 - No rate limiting on cognitive operations
Implements: FEATURE-C3 - Input validation framework (rate limiting component)

Copyright (c) 2024-2025, orpheus497. All rights reserved.
Licensed under the MIT License.
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Tokens are added at a fixed rate. Each operation consumes one token.
    If no tokens available, operation is rate-limited.
    """

    capacity: int  # Maximum tokens
    refill_rate: float  # Tokens added per second
    tokens: float  # Current tokens
    last_refill: float  # Last refill timestamp

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient tokens
        """
        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate time to wait until tokens available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """
    Multi-user rate limiter using token bucket algorithm.

    Each user has separate buckets for different operation types.
    Prevents resource exhaustion and abuse.
    """

    def __init__(
        self,
        default_capacity: int = 60,
        default_refill_rate: float = 1.0,
    ):
        """
        Initialize the RateLimiter.

        Args:
            default_capacity: Default bucket capacity (max burst)
            default_refill_rate: Default token refill rate (ops/second)
        """
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate

        # Per-user buckets: {user: {operation: TokenBucket}}
        self.buckets: Dict[str, Dict[str, TokenBucket]] = {}
        self.lock = threading.RLock()

    def _get_bucket(
        self,
        user: str,
        operation: str,
        capacity: Optional[int] = None,
        refill_rate: Optional[float] = None,
    ) -> TokenBucket:
        """
        Get or create token bucket for user and operation.

        Args:
            user: Username
            operation: Operation type (e.g., 'query', 'memory', 'file')
            capacity: Bucket capacity (None = use default)
            refill_rate: Refill rate (None = use default)

        Returns:
            TokenBucket for user and operation
        """
        with self.lock:
            if user not in self.buckets:
                self.buckets[user] = {}

            if operation not in self.buckets[user]:
                cap = capacity if capacity is not None else self.default_capacity
                rate = (
                    refill_rate if refill_rate is not None else self.default_refill_rate
                )

                self.buckets[user][operation] = TokenBucket(
                    capacity=cap,
                    refill_rate=rate,
                    tokens=float(cap),  # Start full
                    last_refill=time.time(),
                )

            return self.buckets[user][operation]

    def check_rate_limit(
        self,
        user: str,
        operation: str,
        tokens: int = 1,
        capacity: Optional[int] = None,
        refill_rate: Optional[float] = None,
    ) -> bool:
        """
        Check if operation is allowed under rate limit.

        Args:
            user: Username
            operation: Operation type
            tokens: Number of tokens to consume
            capacity: Override default capacity
            refill_rate: Override default refill rate

        Returns:
            True if allowed, False if rate-limited
        """
        bucket = self._get_bucket(user, operation, capacity, refill_rate)
        allowed = bucket.consume(tokens)

        if not allowed:
            wait_time = bucket.get_wait_time(tokens)
            logger.warning(
                f"Rate limit exceeded for user '{user}' operation '{operation}'. "
                f"Wait {wait_time:.1f}s"
            )

        return allowed

    def get_wait_time(
        self,
        user: str,
        operation: str,
        tokens: int = 1,
    ) -> float:
        """
        Get wait time until operation allowed.

        Args:
            user: Username
            operation: Operation type
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        bucket = self._get_bucket(user, operation)
        return bucket.get_wait_time(tokens)

    def reset_user(self, user: str) -> None:
        """
        Reset all rate limits for a user.

        Args:
            user: Username
        """
        with self.lock:
            if user in self.buckets:
                del self.buckets[user]
                logger.info(f"Reset rate limits for user '{user}'")

    def get_stats(self, user: str) -> Dict[str, Dict[str, float]]:
        """
        Get rate limit stats for a user.

        Args:
            user: Username

        Returns:
            Dictionary of operation stats:
            {
                'operation': {
                    'tokens': float,
                    'capacity': int,
                    'refill_rate': float,
                    'utilization': float  # 0.0 to 1.0
                }
            }
        """
        with self.lock:
            if user not in self.buckets:
                return {}

            stats = {}
            for operation, bucket in self.buckets[user].items():
                # Trigger refill to get current state
                bucket.consume(0)

                stats[operation] = {
                    "tokens": bucket.tokens,
                    "capacity": bucket.capacity,
                    "refill_rate": bucket.refill_rate,
                    "utilization": 1.0 - (bucket.tokens / bucket.capacity),
                }

            return stats


# Singleton instance for convenience
_default_rate_limiter: Optional[RateLimiter] = None


def get_default_rate_limiter() -> RateLimiter:
    """Get the default singleton RateLimiter instance."""
    global _default_rate_limiter
    if _default_rate_limiter is None:
        _default_rate_limiter = RateLimiter()
    return _default_rate_limiter
