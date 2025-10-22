from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for individual text chunks."""

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    section_id: UUID
    position: int = Field(ge=0, description="Sequential position within section")
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    token_count: int = Field(ge=0)
    reference_ids: list[str] = Field(
        default_factory=list, description="Citation IDs appearing in chunk"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )  # FIX: utcnow is deprecated


class Chunk(BaseModel):
    """Text chunk with optional embedding for retrieval."""

    content: str = Field(min_length=1)
    embedding: list[float] | None = Field(default=None)
    metadata: ChunkMetadata


class SectionMetadata(BaseModel):
    """Metadata for document sections."""

    section_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    title: str
    position: int = Field(ge=0, description="Sequential position in document")
    char_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    reference_count: int = Field(ge=0, description="Total inline references")
    citation_ids: set[str] = Field(
        default_factory=set, description="Unique citation IDs in section"
    )


class Section(BaseModel):
    """Logical section from parsed document."""

    content: str
    chunks: list[Chunk] = Field(default_factory=list)
    metadata: SectionMetadata


class Citation(BaseModel):
    """Bibliography entry from GROBID."""

    citation_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: str | None = None
    doi: str | None = None
    journal: str | None = None
    venue: str | None = None


class DocumentMetadata(BaseModel):
    """Core document metadata."""

    document_id: UUID = Field(default_factory=uuid4)
    title: str
    authors: list[str] = Field(default_factory=list)
    publication_year: str | None = None
    keywords: set[str] = Field(default_factory=set)
    doi: str | None = None
    filename: str
    source_path: str | None = None
    file_size: int = Field(ge=0, description="Size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    """Complete processed document with sections and citations."""

    metadata: DocumentMetadata
    abstract: str | None = None
    sections: list[Section] = Field(default_factory=list)
    citations: dict[str, Citation] = Field(
        default_factory=dict, description="Bibliography indexed by citation ID"
    )

    def get_all_chunks(self) -> list[Chunk]:
        """Retrieve all chunks across all sections."""
        return [chunk for section in self.sections for chunk in section.chunks]

    def get_section_by_title(self, title: str) -> Section | None:
        """Find section by exact title match."""
        for section in self.sections:
            if section.metadata.title == title:
                return section
        return None

    def get_citation_context(self, citation_id: str) -> list[tuple[str, str]]:
        """Get all (section_title, chunk_content) pairs that cite this work."""
        contexts = []
        for section in self.sections:
            if citation_id in section.metadata.citation_ids:
                for chunk in section.chunks:
                    if citation_id in chunk.metadata.reference_ids:
                        contexts.append((section.metadata.title, chunk.content))
        return contexts
