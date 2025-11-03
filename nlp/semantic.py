from typing import Any

import langextract as lx
from spacy.tokens import Doc, Span

from config.llm import LangExtractConfig
from config.nlp import SectionConfig
from llm.prompts.langextract import PROMPT, EXAMPLES


class EntityExtractor:
    """Extract entities from scientific text."""

    def __init__(self, config: LangExtractConfig):
        self.config = config
        self.model_id = config.model_id
        self.api_key = config.api_key
        self.max_workers = config.max_workers

    def extract(self, text: str, section_config: SectionConfig) -> list[dict[str, Any]]:
        """Execute multi-pass entity extraction."""
        result = lx.extract(
            text_or_documents=text,
            prompt_description=PROMPT,
            examples=EXAMPLES,
            model_id=self.model_id,
            api_key=self.api_key,
            extraction_passes=section_config.extraction_passes,
            max_workers=self.max_workers,
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

    def get_context(self, span: Span, context_size: int = 2) -> str:
        """Get surrounding sentences for entity span."""
        sents = list(span.doc.sents)
        
        # find sentence index containing the span
        sent_idx = None
        for i, sent in enumerate(sents):
            if span.start >= sent.start and span.end <= sent.end:
                sent_idx = i
                break
        
        if sent_idx is None:
            return span.sent.text  # fallback to just the sentence
        
        start = max(0, sent_idx - context_size // 2)
        end = min(len(sents), sent_idx + context_size // 2 + 1)
        
        return " ".join(s.text for s in sents[start:end])

    def convert_to_spans(
        self, entities: list[dict], doc: Doc, context_size: int = 2
    ) -> list[dict[str, Any]]:
        """Convert entities to dictionaries with spans and context."""
        results = []
        sents = list(doc.sents)

        for entity in entities:
            span = self.convert_entity(entity, doc)
            if span is not None:
                # find sentence index
                sent_idx = 0
                for i, sent in enumerate(sents):
                    if span.start >= sent.start and span.end <= sent.end:
                        sent_idx = i
                        break
                
                result = dict(entity)
                result["span"] = span
                result["sentence_context"] = self.get_context(span, context_size)
                result["sentence_index"] = sent_idx
                results.append(result)

        return results

    def process_section(
        self, text: str, doc: Doc, section_config: SectionConfig, context_size: int = 2
    ) -> dict[str, Any]:
        """Process section for entity extraction."""
        entities = self.extract(text, section_config)
        processed = self.convert_to_spans(entities, doc, context_size)

        return {
            "entities": processed,
            "metadata": {
                "total_entities": len(entities),
                "span_conversions": len(processed),
            },
        }


def extract_entities(
    text: str,
    doc: Doc,
    section_config: SectionConfig,
    langextract_config: LangExtractConfig,
    context_size: int = 2,
) -> dict[str, Any]:
    """Extract entities from text section."""
    extractor = EntityExtractor(langextract_config)
    return extractor.process_section(text, doc, section_config, context_size)