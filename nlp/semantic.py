from typing import Any

import langextract as lx
from spacy.tokens import Doc, Span

from config.llm import LangExtractConfig
from config.nlp import SectionConfig
from llm.prompts.langextract import PROMPT, EXAMPLES


class SemanticExtractor:
    """Extract entities from scientific text."""

    def __init__(self, langextract_config: LangExtractConfig):
        self.langextract_config = langextract_config

    # TODO: match relation.py
    # extract_entities -> extract 
    # include the sentence
    def extract_entities(
        self, text: str, section_config: SectionConfig
    ) -> list[dict[str, Any]]:
        """Execute multi-pass entity extraction."""
        result = lx.extract(
            text_or_documents=text,
            prompt_description=PROMPT,
            examples=EXAMPLES,
            model_id=self.langextract_config.model_id,
            api_key=self.langextract_config.api_key,
            extraction_passes=section_config.extraction_passes,
            max_workers=self.langextract_config.max_workers,
            max_char_buffer=section_config.max_char_buffer,
        )

        entities = []
        for extraction in result.extractions:
            entity_dict = {
                "text": extraction.extraction_text,
                "type": extraction.extraction_class,
                "char_interval": (
                    {
                        "start_pos": extraction.char_interval.start_pos,
                        "end_pos": extraction.char_interval.end_pos,
                    }
                    if extraction.char_interval
                    else None
                ),
                "attributes": extraction.attributes if extraction.attributes else {},
            }
            entities.append(entity_dict)

        return entities

    def convert_entity(self, entity: dict, doc: Doc) -> Span | None:
        """Map character positions to spaCy token spans."""
        if entity["char_interval"] is None:
            return None

        start_char = entity["char_interval"]["start_pos"]
        end_char = entity["char_interval"]["end_pos"]

        indices = []
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)

            if not (token_end <= start_char or token_start >= end_char):
                indices.append(token.i)

        if not indices:
            return None

        return doc[min(indices) : max(indices) + 1]

    def convert_to_spans(
        self, entities: list[dict], doc: Doc
    ) -> list[tuple[dict, Span]]:
        """Convert entities to spaCy spans."""
        entity_spans = []

        for entity in entities:
            span = self.convert_entity(entity, doc)
            if span is not None:
                entity_spans.append((entity, span))

        return entity_spans

    def process_section(
        self, text: str, doc: Doc, section_config: SectionConfig
    ) -> dict[str, Any]:
        """Process section for entity extraction."""
        entities = self.extract_entities(text, section_config)
        entity_spans = self.convert_to_spans(entities, doc)

        return {
            "entities": entities,
            "metadata": {
                "total_entities": len(entities),
                "span_conversions": len(entity_spans),
            },
        }


def extract_semantic(
    text: str,
    doc: Doc,
    section_config: SectionConfig,
    langextract_config: LangExtractConfig,
) -> dict[str, Any]:
    """Extract entities from text section."""
    extractor = SemanticExtractor(langextract_config)
    return extractor.process_section(text, doc, section_config)
