from enum import Enum
from typing import TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Entity types for academic literature."""

    CONCEPT = "concept"
    METHOD = "method"
    PROBLEM = "problem"
    CLAIM = "claim"
    FINDING = "finding"
    METRIC = "metric"


class ConceptAttributes(TypedDict, total=False):
    """Attributes for CONCEPT entities."""

    domain: str  # "graph-based RAG", "NLP"
    introduces: bool  # True if novel in paper


class MethodAttributes(TypedDict, total=False):
    """Attributes for METHOD entities."""

    purpose: str  # "detect communities", "retrieve information"
    applied_to: str  # concept/problem it addresses


class ProblemAttributes(TypedDict, total=False):
    """Attributes for PROBLEM entities."""

    scope: str  # "computational", "quality", "scalability"
    affects: str  # method/concept impacted


class ClaimAttributes(TypedDict, total=False):
    """Attributes for CLAIM entities."""

    evidence_type: str  # "empirical", "theoretical", "none"
    supports: str  # method/concept/finding it asserts


class FindingAttributes(TypedDict, total=False):
    """Attributes for FINDING entities."""

    comparison: str  # "outperforms", "matches", "underperforms"
    baseline: str  # compared method/value


class MetricAttributes(TypedDict, total=False):
    """Attributes for METRIC entities."""

    evaluates: str  # method/concept measured
    units: str  # "tokens", "accuracy", "%"
    direction: str  # "higher_better", "lower_better"


class Entity(BaseModel):
    """Extracted entity from scientific text."""

    entity_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType
    text: str = Field(min_length=1, description="Extracted text span")
    attributes: dict[str, str | bool] = Field(default_factory=dict)
    context: str | None = Field(default=None, description="Surrounding text")
    document_id: UUID | None = None
    section_id: UUID | None = None

    def to_langextract_format(self) -> dict[str, str | dict]:
        """Convert to LangExtract Extraction format."""
        return {
            "extraction_class": self.entity_type.value,
            "extraction_text": self.text,
            "attributes": self.attributes,
        }


# type hints
EntityAttributes = (
    ConceptAttributes
    | MethodAttributes
    | ProblemAttributes
    | ClaimAttributes
    | FindingAttributes
    | MetricAttributes
)
