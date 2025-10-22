from enum import Enum
from typing import Any, TypedDict
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

    type: str  # "model", "framework", "theory", "algorithm", "technique"
    based_on: str  # name of the concept it builds on
    novelty: str  # "novel" or "existing"


class MethodAttributes(TypedDict, total=False):
    """Attributes for METHOD entities."""

    purpose: str  # "detect communities", "retrieve information"
    applied_to: str  # concept/problem it addresses


class ProblemAttributes(TypedDict, total=False):
    """Attributes for PROBLEM entities."""

    scope: str  # "computational", "quality", "scalability", "generalization"
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
    units: str  # "tokens", "accuracy", "%", "score"
    direction: str  # "higher_better", "lower_better"


class Entity(BaseModel):
    """Extracted entity from scientific text."""

    entity_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType
    text: str = Field(min_length=1, max_length=200, description="Extracted text span")
    attributes: dict[str, str | bool] = Field(default_factory=dict)
    context: str | None = Field(default=None, description="Surrounding text")
    document_id: UUID | None = None
    section_id: UUID | None = None

    def to_langextract_format(self) -> dict[str, str | dict]:
        """Convert to LangExtract Extraction format.
        
        Returns:
            Dictionary with extraction_class, extraction_text, and attributes.
        """
        return {
            "extraction_class": self.entity_type.value,
            "extraction_text": self.text,
            "attributes": self.attributes,
        }

    @classmethod
    def from_langextract_format(
        cls,
        extraction: dict[str, Any],
        document_id: UUID | None = None,
        section_id: UUID | None = None,
    ) -> "Entity":
        """Create Entity from LangExtract extraction result.
        
        Args:
            extraction: Dictionary from LangExtract with extraction_class, 
                       extraction_text, and attributes.
            document_id: Optional UUID of source document.
            section_id: Optional UUID of source section.
            
        Returns:
            Entity instance.
        """
        return cls(
            entity_type=EntityType(str(extraction["extraction_class"])),
            text=str(extraction["extraction_text"]),
            attributes=dict(extraction.get("attributes", {})),
            document_id=document_id,
            section_id=section_id,
        )


# type hints
EntityAttributes = (
    ConceptAttributes
    | MethodAttributes
    | ProblemAttributes
    | ClaimAttributes
    | FindingAttributes
    | MetricAttributes
)