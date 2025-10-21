import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import httpx

from client.base import BaseClient, RateLimiter
from config.client import ArXivConfig


class ArXivClientError(Exception):
    """Exception for arXiv client errors."""
    pass


class ArXivClient(BaseClient):
    """
    ArXiv API client for fetching paper metadata and downloading PDFs.
    
    Supports:
    - PDF downloads by arXiv ID
    - Metadata extraction including title, abstract, authors, categories, DOI
    - Both synchronous and asynchronous operations
    
    Reference: https://info.arxiv.org/help/api/user-manual.html
    """
    
    def __init__(self, config: ArXivConfig | None = None):
        """Initialize ArXiv client.
        
        Args:
            config: ArXivConfig instance, uses defaults if None
        """
        config = config or ArXivConfig()
        super().__init__(config)
        self.config: ArXivConfig = config
        self.rate_limiter = RateLimiter(config.rate_limit)
    
    def fetch(self, arxiv_id: str) -> dict[str, Any]:
        """Fetch metadata for a paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv identifier (e.g., "2103.15348")
        
        Returns:
            Dictionary containing paper metadata
        """
        return self.get_metadata(arxiv_id)
    
    def get_metadata(self, arxiv_id: str) -> dict[str, Any]:
        """Fetch complete metadata for arXiv paper.
        
        Args:
            arxiv_id: arXiv identifier (e.g., "2103.15348")
        
        Returns:
            Dictionary with fields: arxiv_id, title, url, abstract, authors,
            published, updated, doi, external_doi, primary_category, categories,
            comment, journal_ref, pdf_url
        
        Raises:
            ArXivClientError: If request fails after retries
        """
        self.rate_limiter.wait()
        
        params = {"id_list": arxiv_id}
        last_error: Exception | None = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    self.config.base_url,
                    params=params
                )
                response.raise_for_status()
                return self._parse_entry(response.content, arxiv_id)
                
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise ArXivClientError(
            f"Request failed after {self.config.max_retries} attempts: {last_error}"
        )
    
    async def get_metadata_async(self, arxiv_id: str) -> dict[str, Any]:
        """Fetch metadata asynchronously.
        
        Args:
            arxiv_id: arXiv identifier
        
        Returns:
            Metadata dictionary
        
        Raises:
            ArXivClientError: If request fails after retries
        """
        self.rate_limiter.wait()
        
        params = {"id_list": arxiv_id}
        async_client = self._get_async_session()
        last_error: Exception | None = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await async_client.get(
                    self.config.base_url,
                    params=params
                )
                response.raise_for_status()
                return self._parse_entry(response.content, arxiv_id)
                
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise ArXivClientError(
            f"Request failed after {self.config.max_retries} attempts: {last_error}"
        )
    
    def _parse_entry(self, content: bytes, arxiv_id: str) -> dict[str, Any]:
        """Parse Atom XML response to extract metadata.
        
        Args:
            content: Raw XML response bytes
            arxiv_id: arXiv ID for error messages
        
        Returns:
            Parsed metadata dictionary
        
        Raises:
            ValueError: If required fields are missing
        """
        root = ET.fromstring(content)
        ns = self.config.namespaces
        
        entry = root.find('atom:entry', ns)
        if entry is None:
            raise ValueError(f"No paper found for arXiv ID: {arxiv_id}")
        
        # extract + validate required fields individually for type narrowing
        id_elem = entry.find('atom:id', ns)
        if id_elem is None or id_elem.text is None:
            raise ValueError(f"Missing ID in arXiv response for {arxiv_id}")
        
        title_elem = entry.find('atom:title', ns)
        if title_elem is None:
            raise ValueError(f"Missing title in arXiv response for {arxiv_id}")
        
        summary_elem = entry.find('atom:summary', ns)
        if summary_elem is None:
            raise ValueError(f"Missing summary in arXiv response for {arxiv_id}")
        
        published_elem = entry.find('atom:published', ns)
        if published_elem is None:
            raise ValueError(f"Missing published date in arXiv response for {arxiv_id}")
        
        updated_elem = entry.find('atom:updated', ns)
        if updated_elem is None:
            raise ValueError(f"Missing updated date in arXiv response for {arxiv_id}")
        
        # now, it's safe to access .text
        id_url = id_elem.text
        extracted_id = id_url.split('/abs/')[-1]
        
        # extract authors (<author>)
        authors = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None and name_elem.text:
                author_data = {'name': name_elem.text}
                affiliation = author.find('arxiv:affiliation', ns)
                if affiliation is not None and affiliation.text:
                    author_data['affiliation'] = affiliation.text
                authors.append(author_data)
        
        # extract categories (<category>)
        categories = [cat.get('term') for cat in entry.findall('atom:category', ns)]
        
        # primary category (<arxiv:primary_category>)
        primary_cat = entry.find('arxiv:primary_category', ns)
        primary_category = primary_cat.get('term') if primary_cat is not None else None
        
        # optional fields (<arxiv:comment>, <arxiv:journal_ref>, <arxiv:doi>)
        comment_elem = entry.find('arxiv:comment', ns)
        journal_ref_elem = entry.find('arxiv:journal_ref', ns)
        external_doi_elem = entry.find('arxiv:doi', ns)
        
        # PDF link
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break
        if pdf_url is None:
            pdf_url = f"{self.config.pdf_base_url}/{extracted_id}.pdf"
        
        return {
            'arxiv_id': extracted_id,
            'title': (title_elem.text or "").strip(),
            'url': id_url,
            'abstract': (summary_elem.text or "").strip(),
            'authors': authors,
            'published': published_elem.text or "",
            'updated': updated_elem.text or "",
            'doi': self.arxiv_id_to_doi(extracted_id),
            'external_doi': external_doi_elem.text if external_doi_elem is not None else None,
            'primary_category': primary_category,
            'categories': categories,
            'comment': comment_elem.text if comment_elem is not None else None,
            'journal_ref': journal_ref_elem.text if journal_ref_elem is not None else None,
            'pdf_url': pdf_url
        }
    
    def download_pdf(self, arxiv_id: str, output_path: str) -> None:
        """Download PDF for given arXiv ID.
        
        Args:
            arxiv_id: arXiv identifier
            output_path: Path to save PDF file
        """
        self.rate_limiter.wait()
        
        pdf_url = f"{self.config.pdf_base_url}/{arxiv_id}.pdf"
        response = self.session.get(pdf_url)
        response.raise_for_status()
        
        Path(output_path).write_bytes(response.content)
    
    async def download_pdf_async(self, arxiv_id: str, output_path: str) -> None:
        """Download PDF asynchronously.
        
        Args:
            arxiv_id: arXiv identifier
            output_path: Path to save PDF file
        """
        self.rate_limiter.wait()
        
        pdf_url = f"{self.config.pdf_base_url}/{arxiv_id}.pdf"
        async_client = self._get_async_session()
        response = await async_client.get(pdf_url)
        response.raise_for_status()
        
        Path(output_path).write_bytes(response.content)
    
    @staticmethod
    def arxiv_id_to_doi(arxiv_id: str) -> str:
        """Convert arXiv ID to arXiv-assigned DOI."""
        clean_id = arxiv_id.replace("arXiv:", "").replace(":", ".")
        return f"10.48550/arXiv.{clean_id}"
    
    @staticmethod
    def doi_to_arxiv_id(doi: str) -> str:
        """Extract arXiv ID from arXiv-assigned DOI."""
        if doi.startswith("10.48550/arXiv."):
            return doi.replace("10.48550/arXiv.", "")
        elif doi.startswith("10.48550/arxiv."):
            return doi.replace("10.48550/arxiv.", "")
        raise ValueError(f"Not an arXiv DOI: {doi}")