from enum import Enum
from typing import Any, TypedDict
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    CONTRIBUTION = "contribution"
    PROBLEM = "problem"
    CLAIM = "claim"
    FINDING = "finding"
    RELATION = "relation"


class RelationType(str, Enum):
    DERIVED_FROM = "derived_from"
    ADDRESSES = "addresses"
    EVALUATES = "evaluates"
    COMPARES_TO = "compares_to"
    SUPPORTS = "supports"
    USES = "uses"
    MEASURED_BY = "measured_by"


class ContributionAttributes(TypedDict, total=False):
    category: str
    novelty: str
    purpose: str
    type: str


class ProblemAttributes(TypedDict, total=False):
    scope: str


class ClaimAttributes(TypedDict, total=False):
    evidence_type: str


class FindingAttributes(TypedDict, total=False):
    comparison: str
    value: str


class RelationAttributes(TypedDict, total=False):
    source: str  # entity text reference
    target: str  # entity text reference
    type: str  # relation type
    context: str  # semantic context


class Entity(BaseModel):
    entity_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType
    text: str = Field(min_length=1, max_length=200)
    attributes: dict[str, str | bool] = Field(default_factory=dict)
    document_id: UUID | None = None

    def to_langextract_format(self) -> dict[str, str | dict]:
        return {
            "extraction_class": self.entity_type.value,
            "extraction_text": self.text,
            "attributes": self.attributes,
        }

    @classmethod
    def from_langextract_format(
        cls, extraction: dict[str, Any], document_id: UUID | None = None
    ) -> "Entity":
        return cls(
            entity_type=EntityType(str(extraction["extraction_class"])),
            text=str(extraction["extraction_text"]),
            attributes=dict(extraction.get("attributes", {})),
            document_id=document_id,
        )


class Relationship(BaseModel):
    """Resolved relationship after post-processing RELATION entities."""

    relationship_id: UUID = Field(default_factory=uuid4)
    source_entity_id: UUID
    target_entity_id: UUID
    relation_type: RelationType
    original_span: str  # original RELATION extraction text
    document_id: UUID | None = None


EntityAttributes = (
    ContributionAttributes
    | ProblemAttributes
    | ClaimAttributes
    | FindingAttributes
    | RelationAttributes
)
