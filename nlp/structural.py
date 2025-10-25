from typing import Any
from spacy.tokens import Doc

from models.grobid import Section
from config.nlp import NLPConfig, normalize_section
from utils.clean_text import preprocess_section
from nlp.syntactic import parse


class SectionProcessor:
    """Process paper sections through structural and syntactic pipeline."""
    
    def __init__(self, nlp_config: NLPConfig):
        self.nlp_config = nlp_config
    
    def process(self, section: Section) -> dict[str, Any]:
        """Preprocess and parse section structure."""
        normalized = normalize_section(section.title, self.nlp_config.patterns)
        section_config = self.nlp_config.sections[normalized]
        
        section_text = section.to_str()
        clean_text = preprocess_section(section_text)
        doc = parse(clean_text)
        
        return {
            'section': section.title,
            'normalized': normalized,
            'text': clean_text,
            'doc': doc,
            'config': section_config
        }
