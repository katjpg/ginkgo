from typing import Any
from enum import Enum
from httpx import Headers, HTTPError
from pydantic import BaseModel, Field, field_validator, ConfigDict


# 1. CITATIONS
class PageRange(BaseModel):
    """Represents the 'to' and 'from' attributes in <biblScope/> XML tag."""

    from_page: int
    to_page: int


class Scope(BaseModel):
    """Represents the <biblScope/> XML tag."""

    volume: int | None = None
    pages: PageRange | None = None

    def is_empty(self) -> bool:
        """Return True if all fields are None."""
        return self.volume is None and self.pages is None


class Date(BaseModel):
    """Represents the 'when' attribute in the <date/> XML tag."""

    year: str
    month: str | None = None
    day: str | None = None


class PersonName(BaseModel):
    """Represents the <persName/> XML tag."""

    surname: str
    first_name: str | None = None
    # middle_name: str | None = None
    # title: str | None = None

    def to_string(self) -> str:
        """Return string representation of object."""
        if self.first_name:
            return f"{self.first_name} {self.surname}"
        return self.surname


class Affiliation(BaseModel):
    """Represents the <affiliation> XML tag."""

    department: str | None = None
    institution: str | None = None
    laboratory: str | None = None
    # address: Address | None = None

    def is_empty(self) -> bool:
        """Return True if all fields are None."""
        return all(getattr(self, field) is None for field in self.model_fields)


class Author(BaseModel):
    """Represents the <author> XML tag."""

    person_name: PersonName
    affiliations: list[Affiliation] = Field(default_factory=list)
    email: str | None = None


class CitationIDs(BaseModel):
    """Represents the <idno> XML tag."""

    DOI: str | None = None
    arXiv: str | None = None
    # issn: str | None = None
    # pii: str | None = None
    # other: str | None = None

    def is_empty(self) -> bool:
        """Return True if all fields are None."""
        return self.DOI is None and self.arXiv is None


class Citation(BaseModel):
    """Represents the <biblStruct> XML tag."""

    title: str
    authors: list[Author] = Field(default_factory=list)
    date: Date | None = None
    ids: CitationIDs | None = None
    target: str | None = None
    publisher: str | None = None
    journal: str | None = None
    series: str | None = None
    scope: Scope | None = None
    # meeting: str | None = None
    # phone: str | None = None


# 2. SECTION
class Marker(str, Enum):
    """Represents the callouts to structures.

    <https://grobid.readthedocs.io/en/latest/training/fulltext/#markers-callouts-to-structures>
    """

    bibr = "bibr"
    figure = "figure"
    table = "table"
    box = "box"
    formula = "formula"


class Ref(BaseModel):
    """Represents <ref> XML tag.

    Stores the start and end positions of the reference rather than the text.
    """

    start: int
    end: int
    marker: Marker | None = None
    target: str | None = None


class RefText(BaseModel):
    """Represents the <p> XML tag.

    Supports embedded <ref> XML tags.
    """

    text: str
    refs: list[Ref] = Field(default_factory=list)

    @property
    def plain_text(self) -> str:
        """Return text without any references.

        Trailing whitespace is removed.
        """
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
    """Represents <div> tag with <head> tag."""

    title: str
    paragraphs: list[RefText] = Field(default_factory=list)

    def to_str(self) -> str:
        """Return paragraphs in plain text format."""
        text = ""
        for paragraph in self.paragraphs:
            text += paragraph.plain_text

        return text


# 3. ARTICLE
class Table(BaseModel):
    """Represents the <figure> XML tag of type table."""

    heading: str
    description: str | None = None
    rows: list[list[str]] = Field(default_factory=list)


class Article(BaseModel):
    """Represents the scholarly article."""

    bibliography: Citation
    keywords: set[str]
    citations: dict[str, Citation]
    sections: list[Section]
    tables: dict[str, Table]
    abstract: Section | None = None


# 4. FORM
class File(BaseModel):
    """Represents the PDF file used as input."""

    payload: bytes
    file_name: str | None = None
    mime_type: str | None = None

    def to_tuple(self) -> tuple[str | None, bytes, str | None]:
        """Return a tuple for httpx multipart/form-data encoding."""
        return self.file_name, self.payload, self.mime_type


class Form(BaseModel):
    """Represents form data accepted by GROBID's processFulltextDocument endpoint."""

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
        """Validate consolidate fields are 0, 1, or 2."""
        if v is not None and v not in (0, 1, 2):
            raise ValueError("must be 0, 1, or 2")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary for multipart/form-data."""
        form_dict: dict[str, Any] = {"input": self.file.to_tuple()}

        if self.segment_sentences:
            form_dict["segmentSentences"] = "1"

        if self.consolidate_header is not None:
            form_dict["consolidateHeader"] = str(self.consolidate_header)

        if self.consolidate_citations is not None:
            form_dict["consolidateCitations"] = str(self.consolidate_citations)

        if self.include_raw_citations is not None:
            form_dict["includeRawCitations"] = self.include_raw_citations

        if self.include_raw_affiliations is not None:
            form_dict["includeRawAffiliations"] = self.include_raw_affiliations

        if self.tei_coordinates is not None:
            form_dict["teiCoordinates"] = self.tei_coordinates

        return form_dict


# 5. RESPONSE
class Response(BaseModel):
    """Represents the response from GROBID's processFulltextDocument endpoint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status_code: int
    content: bytes
    headers: Headers

    def raise_for_status(self) -> None:
        """Only considers GROBID's documented status codes as HTTP errors."""
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
