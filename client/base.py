from abc import ABC, abstractmethod
from typing import Any
import time
import httpx

from config.client import ClientConfig


class BaseClient(ABC):
    """Abstract base class for API clients."""

    def __init__(self, config: ClientConfig):
        """Initialize base client.

        Args:
            config: Client configuration instance
        """
        self.config = config
        self.session = httpx.Client(
            headers={"User-Agent": config.user_agent},
            timeout=config.request_timeout,
            follow_redirects=True,
        )
        self.async_session: httpx.AsyncClient | None = None

    @abstractmethod
    def fetch(self, **kwargs) -> Any:
        """Fetch data from the API.

        Returns:
            API-specific data structure
        """
        pass

    def _get_async_session(self) -> httpx.AsyncClient:
        """Lazy initialization of async session."""
        if self.async_session is None:
            self.async_session = httpx.AsyncClient(
                headers={"User-Agent": self.config.user_agent},
                timeout=self.config.request_timeout,
            )
        return self.async_session

    def close(self) -> None:
        """Close HTTP sessions."""
        self.session.close()
        if self.async_session:
            # NOTE: async_session.aclose() needs to be awaited
            pass

    async def aclose(self) -> None:
        """Close async session."""
        if self.async_session:
            await self.async_session.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


class RateLimiter:
    """Simple rate limiter to ensure minimum time between requests."""

    def __init__(self, min_interval: float):
        """Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between requests
        """
        self.min_interval = min_interval
        self.last_request_time: float = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        if self.min_interval <= 0:
            return

        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
