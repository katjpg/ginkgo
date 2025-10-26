import time
from typing import Any
import httpx

from client.base import BaseClient, RateLimiter
from config.client import GROBIDConfig
from models.grobid import Form, Response


class GROBIDClientError(Exception):
    """GROBID client error."""

    pass


class GROBIDClient(BaseClient):
    """GROBID API client for PDF to TEI XML conversion."""

    def __init__(self, config: GROBIDConfig | None = None):
        """Initialize with configuration."""
        config = config or GROBIDConfig()
        super().__init__(config)
        self.config: GROBIDConfig = config
        self.rate_limiter = RateLimiter(config.rate_limit)

    def fetch(self, form: Form) -> bytes:
        """Extract TEI XML from PDF."""
        response = self.process_pdf(form)
        return response.content

    def process_pdf(self, form: Form) -> Response:
        """Process PDF and return TEI XML response."""
        self.rate_limiter.wait()

        # separate file and data fields for multipart encoding
        files, data = form.to_files_and_data()

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    self.config.full_url,
                    files=files,
                    data=data,  # scalar fields go here, not in files
                )
                return self._build_response(response)

            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
            except httpx.HTTPError as exc:
                raise GROBIDClientError(f"HTTP error: {exc}")

        raise GROBIDClientError(
            f"Request failed after {self.config.max_retries} attempts: {last_error}"
        )

    async def process_pdf_async(self, form: Form) -> Response:
        """Process PDF asynchronously."""
        self.rate_limiter.wait()

        # separate file and data fields for multipart encoding
        files, data = form.to_files_and_data()

        async_client = self._get_async_session()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = await async_client.post(
                    self.config.full_url,
                    files=files,
                    data=data,  # scalar fields go here, not in files
                )
                return self._build_response(response)

            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
            except httpx.HTTPError as exc:
                raise GROBIDClientError(f"HTTP error: {exc}")

        raise GROBIDClientError(
            f"Request failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _build_response(self, response: httpx.Response) -> Response:
        """Build and validate Response object."""
        res = Response(
            status_code=response.status_code,
            content=response.content,
            headers=response.headers,
        )

        try:
            res.raise_for_status()
        except httpx.HTTPError as exc:
            raise GROBIDClientError(f"GROBID processing failed: {exc}")

        return res
