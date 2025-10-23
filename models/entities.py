"""
Two-pass system:
- pass 1: Extract entities with intrinsic attributes only
- pass 2: Extract relationships between entities
"""

from enum import Enum
from typing import Any, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Entity types for academic literature."""

    CONTRIBUTION = "contribution"
    PROBLEM = "problem"
    CLAIM = "claim"
    FINDING = "finding"


class RelationType(str, Enum):
    """Relationship types between entities."""

    DERIVED_FROM = "derived_from"  # (Contribution -> Contribution)
    ADDRESSES = "addresses"  # (Contribution -> Problem)
    EVALUATES = "evaluates"  # (Finding -> Contribution)
    COMPARES_TO = "compares_to"  # (Finding -> Contribution)
    SUPPORTS = "supports"  # (Claim -> Contribution) or (Finding -> Claim)
    USES = "uses"  # (Contribution -> Contribution)
    MEASURED_BY = "measured_by"  # (Finding -> Contribution[category=metric])


# pass 1: intrinsic attributes only
class ContributionAttributes(TypedDict, total=False):
    """Intrinsic attributes for CONTRIBUTION entities."""

    category: str  # "method", "model", "framework", "metric", "technique", "algorithm"
    novelty: str  # "novel", "existing", "adaptation"
    purpose: str  # "retrieve information", "measure quality"
    type: str  # "retrieval", "generation", "preprocessing", "evaluation"


class ProblemAttributes(TypedDict, total=False):
    """Intrinsic attributes for PROBLEM entities."""

    scope: str  # "computational", "quality", "scalability", "generalization"


class ClaimAttributes(TypedDict, total=False):
    """Intrinsic attributes for CLAIM entities."""

    evidence_type: str  # "empirical", "theoretical", "none"


class FindingAttributes(TypedDict, total=False):
    """Intrinsic attributes for FINDING entities."""

    comparison: str  # "outperforms", "matches", "underperforms", "achieves"
    value: str  # "28.4 BLEU", "3% to 6%", "O(n^1.5)"


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
        """Convert to LangExtract Extraction format."""
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
        """Create Entity from LangExtract extraction result."""
        return cls(
            entity_type=EntityType(str(extraction["extraction_class"])),
            text=str(extraction["extraction_text"]),
            attributes=dict(extraction.get("attributes", {})),
            document_id=document_id,
            section_id=section_id,
        )


class Relationship(BaseModel):
    """Relationship between entities (extracted in Pass 2)."""

    relationship_id: UUID = Field(default_factory=uuid4)
    source_entity_id: UUID
    target_entity_id: UUID
    relation_type: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    document_id: UUID | None = None


EntityAttributes = (
    ContributionAttributes | ProblemAttributes | ClaimAttributes | FindingAttributes
)
