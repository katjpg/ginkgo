from typing import Any

import langextract as lx
from spacy.tokens import Doc, Span

from config.llm import LangExtractConfig
from config.nlp import SectionConfig
from llm.prompts.langextract import PROMPT, EXAMPLES
from nlp.syntactic import find_relations


class SemanticExtractor:
    """Extract entities from scientific text."""
    
    def __init__(self, langextract_config: LangExtractConfig):
        self.langextract_config = langextract_config
    
    def extract_entities(self, text: str, section_config: SectionConfig) -> list[dict[str, Any]]:
        """Execute multi-pass entity extraction."""
        result = lx.extract(
            text_or_documents=text,
            prompt_description=PROMPT,
            examples=EXAMPLES,
            model_id=self.langextract_config.model_id,
            api_key=self.langextract_config.api_key,
            extraction_passes=section_config.extraction_passes,
            max_workers=self.langextract_config.max_workers,
            max_char_buffer=section_config.max_char_buffer
        )
        
        entities = []
        for extraction in result.extractions:
            entity_dict = {
                "text": extraction.extraction_text,
                "type": extraction.extraction_class,
                "char_interval": {
                    "start_pos": extraction.char_interval.start_pos,
                    "end_pos": extraction.char_interval.end_pos,
                } if extraction.char_interval else None,
                "attributes": extraction.attributes if extraction.attributes else {}
            }
            entities.append(entity_dict)
        
        return entities
    
    def convert_entity(self, entity: dict, doc: Doc) -> Span | None:
        """Map character positions to spaCy token spans."""
        if entity["char_interval"] is None:
            return None
            
        start_char = entity["char_interval"]["start_pos"]
        end_char = entity["char_interval"]["end_pos"]
        
        # find overlapping tokens
        indices = []
        for token in doc:
            token_start = token.idx
            token_end = token.idx + len(token.text)
            
            if not (token_end <= start_char or token_start >= end_char):
                indices.append(token.i)
        
        if not indices:
            return None
        
        return doc[min(indices):max(indices) + 1]
    
    def convert_to_spans(self, entities: list[dict], doc: Doc) -> list[tuple[dict, Span]]:
        """Convert entities to spaCy spans."""
        entity_spans = []
        
        for entity in entities:
            span = self.convert_entity(entity, doc)
            if span is not None:
                entity_spans.append((entity, span))
        
        return entity_spans
    
    def analyze_syntactic_density(self, entity_spans: list[tuple[dict, Span]]) -> dict[str, Any]:
        """Compute syntactic connection density between entities."""
        syntactic_pairs = 0
        total_pairs = 0
        
        for i, (_, s1) in enumerate(entity_spans):
            for _, s2 in entity_spans[i+1:]:
                total_pairs += 1
                if find_relations(s1, s2):
                    syntactic_pairs += 1
        
        return {
            "total_pairs": total_pairs,
            "connected_pairs": syntactic_pairs,
            "density": syntactic_pairs / total_pairs if total_pairs > 0 else 0.0
        }
    
    def process_section(self, text: str, doc: Doc, section_config: SectionConfig) -> dict[str, Any]:
        """Process section for entity extraction and syntactic analysis."""
        entities = self.extract_entities(text, section_config)
        entity_spans = self.convert_to_spans(entities, doc)
        syntactic_metrics = self.analyze_syntactic_density(entity_spans)
        
        return {
            "entities": entities,
            "metadata": {
                "total_entities": len(entities),
                "span_conversions": len(entity_spans),
                "syntactic_density": syntactic_metrics["density"],
                "connected_pairs": syntactic_metrics["connected_pairs"],
                "total_pairs": syntactic_metrics["total_pairs"]
            }
        }


def extract_semantic(
    text: str, 
    doc: Doc, 
    section_config: SectionConfig,
    langextract_config: LangExtractConfig
) -> dict[str, Any]:
    """Extract entities from text section."""
    extractor = SemanticExtractor(langextract_config)
    return extractor.process_section(text, doc, section_config)