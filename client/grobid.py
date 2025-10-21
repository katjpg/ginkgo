import time
from typing import Any
import httpx

from client.base import BaseClient, RateLimiter
from config.client import GROBIDConfig
from models.grobid import Form, Response


class GROBIDClientError(Exception):
    """Exception for GROBID client errors."""

    pass


class GROBIDClient(BaseClient):
    """
    GROBID API client for converting PDFs to TEI XML.

    Supports:
    - Synchronous and asynchronous PDF processing
    - Full document structure extraction
    - Citation consolidation
    - Configurable processing options

    Reference: https://grobid.readthedocs.io/
    """

    def __init__(self, config: GROBIDConfig | None = None):
        """Initialize GROBID client.

        Args:
            config: GROBIDConfig instance, uses defaults if None
        """
        config = config or GROBIDConfig()
        super().__init__(config)
        self.config: GROBIDConfig = config
        self.rate_limiter = RateLimiter(config.rate_limit)

    def fetch(self, form: Form) -> bytes:
        """Fetch TEI XML for a PDF document.

        Args:
            form: Form object containing PDF and processing options

        Returns:
            TEI XML content as bytes
        """
        response = self.process_pdf(form)
        return response.content

    def process_pdf(self, form: Form) -> Response:
        """Process PDF synchronously and return Response object.

        Args:
            form: Form object with PDF payload

        Returns:
            Response object with TEI XML content

        Raises:
            GROBIDClientError: If request fails after retries
        """
        self.rate_limiter.wait()

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(self.config.full_url, files=form.to_dict())
                return self._build_response(response)

            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
            except httpx.HTTPError as exc:
                raise GROBIDClientError(f"HTTP error: {exc}")

        # at this point, all retries failed
        raise GROBIDClientError(
            f"Request failed after {self.config.max_retries} attempts: {last_error}"
        )

    async def process_pdf_async(self, form: Form) -> Response:
        """Process PDF asynchronously and return Response object.

        Args:
            form: Form object with PDF payload

        Returns:
            Response object with TEI XML content

        Raises:
            GROBIDClientError: If request fails after retries
        """
        self.rate_limiter.wait()

        async_client = self._get_async_session()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = await async_client.post(
                    self.config.full_url, files=form.to_dict()
                )
                return self._build_response(response)

            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
            except httpx.HTTPError as exc:
                raise GROBIDClientError(f"HTTP error: {exc}")

        # at this point, all retries failed
        raise GROBIDClientError(
            f"Request failed after {self.config.max_retries} attempts: {last_error}"
        )

    def _build_response(self, response: httpx.Response) -> Response:
        """Build Response object and validate status.

        Args:
            response: httpx Response object

        Returns:
            Validated Response object

        Raises:
            GROBIDClientError: If response indicates error
        """
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
