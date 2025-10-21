from pydantic import BaseModel, Field


class ClientConfig(BaseModel):
    """General configuration for API clients."""

    base_url: str
    rate_limit: float = Field(default=3.0, description="Seconds between requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    user_agent: str = "GinkgoClient/1.0"


class ArXivConfig(ClientConfig):
    """Configuration for arXiv API client."""

    base_url: str = "http://export.arxiv.org/api/query"
    pdf_base_url: str = "http://arxiv.org/pdf"
    namespaces: dict[str, str] = Field(
        default_factory=lambda: {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
    )


class GROBIDConfig(ClientConfig):
    """Configuration for GROBID API client."""

    base_url: str = "https://kermitt2-grobid.hf.space"
    rate_limit: float = 1.0  # NOTE: GROBID has stricter rate limits
    request_timeout: int = 60  # pdf processing tends to take longer
    api_endpoint: str = "/api/processFulltextDocument"

    @property
    def full_url(self) -> str:
        """Construct full API URL."""
        return f"{self.base_url}{self.api_endpoint}"
