from pydantic import BaseModel, Field


class SectionConfig(BaseModel):
    """Extraction parameters for a section type."""

    extraction_passes: int = Field(ge=1, le=10)
    max_char_buffer: int = Field(ge=100, le=5000)


class NLPConfig(BaseModel):
    """NLP pipeline configuration."""

    sections: dict[str, SectionConfig] = Field(
        default_factory=lambda: {
            "introduction": SectionConfig(extraction_passes=2, max_char_buffer=2000),
            "related_work": SectionConfig(extraction_passes=2, max_char_buffer=2000),
            "methods": SectionConfig(extraction_passes=3, max_char_buffer=2000),
            "experiments": SectionConfig(extraction_passes=3, max_char_buffer=2000),
            "results": SectionConfig(extraction_passes=3, max_char_buffer=2000),
            "discussion": SectionConfig(extraction_passes=2, max_char_buffer=2000),
            "conclusion": SectionConfig(extraction_passes=2, max_char_buffer=2000),
            "default": SectionConfig(extraction_passes=2, max_char_buffer=2000),
        }
    )

    patterns: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "introduction": ["introduction", "intro", "background", "motivation"],
            "related_work": [
                "related work",
                "related works",
                "literature review",
                "prior work",
                "previous work",
            ],
            "methods": [
                "method",
                "methods",
                "methodology",
                "approach",
                "our approach",
                "proposed method",
                "technique",
                "framework",
                "system",
                "architecture",
            ],
            "experiments": [
                "experiment",
                "experiments",
                "experimental setup",
                "evaluation",
                "setup",
            ],
            "results": [
                "result",
                "results",
                "findings",
                "analysis",
                "detailed analysis",
                "overall results",
                "performance",
            ],
            "discussion": ["discussion", "interpretation"],
            "conclusion": [
                "conclusion",
                "conclusions",
                "summary",
                "concluding remarks",
                "future work",
            ],
        }
    )

    max_workers: int = Field(default=5, ge=1, le=20)


def normalize_section(heading: str, patterns: dict[str, list[str]]) -> str:
    """Normalize section heading to standard category."""
    heading_lower = heading.lower().strip()

    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            if pattern in heading_lower:
                return category

    return "default"
