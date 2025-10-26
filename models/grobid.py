from typing import Any
from enum import Enum
from httpx import Headers, HTTPError
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PageRange(BaseModel):
    """Page boundaries from biblScope XML."""

    from_page: int
    to_page: int


class Scope(BaseModel):
    """Bibliographic scope from biblScope tags."""

    volume: int | None = None
    pages: PageRange | None = None

    def is_empty(self) -> bool:
        """Check if all fields are None."""
        return self.volume is None and self.pages is None


class Date(BaseModel):
    """Date from 'when' attribute."""

    year: str
    month: str | None = None
    day: str | None = None


class PersonName(BaseModel):
    """Person name from persName tag."""

    surname: str
    first_name: str | None = None

    def to_string(self) -> str:
        """Format as readable name."""
        if self.first_name:
            return f"{self.first_name} {self.surname}"
        return self.surname


class Affiliation(BaseModel):
    """Author affiliation data."""

    department: str | None = None
    institution: str | None = None
    laboratory: str | None = None

    def is_empty(self) -> bool:
        """Check if all fields are None."""
        return all(getattr(self, field) is None for field in self.model_fields)


class Author(BaseModel):
    """Author with affiliations."""

    person_name: PersonName
    affiliations: list[Affiliation] = Field(default_factory=list)
    email: str | None = None


class CitationIDs(BaseModel):
    """External identifiers from idno tags."""

    DOI: str | None = None
    arXiv: str | None = None

    def is_empty(self) -> bool:
        """Check if all identifiers are None."""
        return self.DOI is None and self.arXiv is None


class Citation(BaseModel):
    """Bibliography entry from biblStruct."""

    title: str
    authors: list[Author] = Field(default_factory=list)
    date: Date | None = None
    ids: CitationIDs | None = None
    target: str | None = None
    publisher: str | None = None
    journal: str | None = None
    series: str | None = None
    scope: Scope | None = None


class Marker(str, Enum):
    """Reference marker types."""

    bibr = "bibr"
    figure = "figure"
    table = "table"
    box = "box"
    formula = "formula"


class Ref(BaseModel):
    """Reference position in text."""

    start: int
    end: int
    marker: Marker | None = None
    target: str | None = None


class RefText(BaseModel):
    """Paragraph with embedded references."""

    text: str
    refs: list[Ref] = Field(default_factory=list)

    @property
    def plain_text(self) -> str:
        """Extract text without references."""
        if len(self.refs) == 0:
            return self.text

        ranges = [(ref.start, ref.end) for ref in self.refs]
        text = ""
        left_bound = 0
        for start, end in ranges:
            text += self.text[left_bound:start].rstrip()
            left_bound = end
        text += self.text[ranges[-1][1] :].rstrip()
        return text


class Section(BaseModel):
    """Document section with paragraphs."""

    title: str
    paragraphs: list[RefText] = Field(default_factory=list)

    def to_str(self) -> str:
        """Concatenate paragraph text."""
        text = ""
        for paragraph in self.paragraphs:
            text += paragraph.plain_text
        return text


class Table(BaseModel):
    """Table from figure tag."""

    heading: str
    description: str | None = None
    rows: list[list[str]] = Field(default_factory=list)


class Article(BaseModel):
    """Parsed scholarly article."""

    bibliography: Citation
    keywords: set[str]
    citations: dict[str, Citation]
    sections: list[Section]
    tables: dict[str, Table]
    abstract: Section | None = None


class File(BaseModel):
    """PDF file for GROBID processing."""

    payload: bytes
    file_name: str | None = None
    mime_type: str | None = None

    def to_tuple(self) -> tuple[str | None, bytes, str | None]:
        """Convert to httpx multipart tuple."""
        return self.file_name, self.payload, self.mime_type


class Form(BaseModel):
    """GROBID processFulltextDocument form data."""

    file: File
    segment_sentences: bool | None = None
    consolidate_header: int | None = None
    consolidate_citations: int | None = None
    include_raw_citations: bool | None = None
    include_raw_affiliations: bool | None = None
    tei_coordinates: str | None = None

    @field_validator("consolidate_header", "consolidate_citations")
    @classmethod
    def validate_consolidate_values(cls, v: int | None) -> int | None:
        """Validate consolidation levels are 0, 1, or 2."""
        if v is not None and v not in (0, 1, 2):
            raise ValueError("must be 0, 1, or 2")
        return v

    def to_files_and_data(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split into files dict and data dict for httpx.post()."""
        # file content goes in files parameter
        files = {"input": self.file.to_tuple()}

        # scalar values go in data parameter as strings
        data = {}

        if self.segment_sentences is not None:
            data["segmentSentences"] = "1" if self.segment_sentences else "0"

        if self.consolidate_header is not None:
            data["consolidateHeader"] = str(self.consolidate_header)

        if self.consolidate_citations is not None:
            data["consolidateCitations"] = str(self.consolidate_citations)

        if self.include_raw_citations is not None:
            data["includeRawCitations"] = "1" if self.include_raw_citations else "0"

        if self.include_raw_affiliations is not None:
            data["includeRawAffiliations"] = (
                "1" if self.include_raw_affiliations else "0"
            )

        if self.tei_coordinates is not None:
            data["teiCoordinates"] = self.tei_coordinates

        return files, data


class Response(BaseModel):
    """GROBID API response."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status_code: int
    content: bytes
    headers: Headers

    def raise_for_status(self) -> None:
        """Raise exception for GROBID-specific error codes."""
        match self.status_code:
            case 203:
                error_msg = "Content couldn't be extracted"
            case 400:
                error_msg = "Wrong request, missing parameters, missing header"
            case 500:
                error_msg = "Internal service error"
            case 503:
                error_msg = "Service not available"
            case _:
                return

        raise HTTPError(f"{self.status_code}: {error_msg}")
