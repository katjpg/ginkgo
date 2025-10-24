from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    TASK = "task"
    METHOD = "method"
    DATASET = "dataset"
    OBJECT = "object"
    METRIC = "metric"
    GENERIC = "generic"
    OTHER = "other"


class Entity(BaseModel):
    entity_id: UUID = Field(default_factory=uuid4)
    entity_type: EntityType
    text: str = Field(min_length=1, max_length=200)
    document_id: UUID | None = None

    def to_langextract_format(self) -> dict[str, str]:
        return {
            "extraction_class": self.entity_type.value,
            "extraction_text": self.text,
        }

    @classmethod
    def from_langextract_format(
        cls, extraction: dict[str, Any], document_id: UUID | None = None
    ) -> "Entity":
        return cls(
            entity_type=EntityType(str(extraction["extraction_class"])),
            text=str(extraction["extraction_text"]),
            document_id=document_id,
        )
