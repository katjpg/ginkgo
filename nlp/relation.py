from typing import Any
import json

from spacy.tokens import Doc
from google import genai
from google.genai import types

from nlp.entity_pairs import create_pairs, filter_by_type, aggregate_mentions
from llm.prompts.relation import build_prompt, EXAMPLES, RESPONSE_SCHEMA
from config.llm import GeminiConfig


class RelationExtractor:
    """Extract relations via Gemini LLM."""
    
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self.examples = EXAMPLES
    
    def extract(self, entities: list[dict], doc: Doc) -> list[dict]:
        """Extract relations from entity pairs."""
        pairs = create_pairs(entities, doc)
        pairs = filter_by_type(pairs)
        
        if not pairs:
            return []
        
        relations = []
        for pair in pairs:
            result = self._classify(pair)
            if result['relation'] != 'NONE':
                relations.append({
                    'head': pair['head']['text'],
                    'tail': pair['tail']['text'],
                    'relation': result['relation'],
                    'confidence': result['confidence'],
                    'evidence': pair['sentence'],
                    'syntax': pair.get('syntax', ''),
                    'reasoning': result.get('reasoning', '')
                })
        
        return relations
    
    def _classify(self, pair: dict) -> dict:
        """Classify single entity pair via Gemini."""
        prompt = build_prompt(pair, self.examples)
        
        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_output_tokens=self.config.max_output_tokens,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.config.model_id,
                contents=prompt,
                config=config
            )
            
            # check if response has text content
            if response.text is None:
                return {
                    'relation': 'NONE',
                    'confidence': 'LOW',
                    'reasoning': 'No response generated from model'
                }
            
            result = json.loads(response.text)
            
            # validate response structure
            if 'relation' not in result:
                result['relation'] = 'NONE'
            if 'confidence' not in result:
                result['confidence'] = 'LOW'
            if 'reasoning' not in result:
                result['reasoning'] = 'No reasoning provided'
                
            return result
            
        except json.JSONDecodeError as e:
            return {
                'relation': 'NONE',
                'confidence': 'LOW',
                'reasoning': f'JSON parsing error: {str(e)}'
            }
        except Exception as e:
            return {
                'relation': 'NONE',
                'confidence': 'LOW',
                'reasoning': f'Classification error: {str(e)}'
            }

def extract_with_aggregation(entities: list[dict], doc: Doc, extractor: RelationExtractor) -> list[dict]:
    """Extract relations with mention aggregation."""
    pairs = create_pairs(entities, doc)
    pairs = filter_by_type(pairs)
    
    if not pairs:
        return []
    
    aggregated = aggregate_mentions(pairs)
    
    relations = []
    for entity_pair in aggregated:
        # combine evidence from multiple mentions
        sentences = []
        syntaxes = []
        
        for mention in entity_pair.mentions[:3]:  # limit to 3 mentions for context length
            sentences.append(mention['sentence'])
            if mention.get('syntax'):
                syntaxes.append(mention['syntax'])
        
        # create enriched pair dictionary
        pair_dict = {
            'head': entity_pair.head,
            'tail': entity_pair.tail,
            'sentence': ' | '.join(sentences),
            'syntax': syntaxes[0] if syntaxes else 'no pattern'
        }
        
        result = extractor._classify(pair_dict)
        
        if result['relation'] != 'NONE':
            relations.append({
                'head': entity_pair.head['text'],
                'tail': entity_pair.tail['text'],
                'relation': result['relation'],
                'confidence': result['confidence'],
                'mention_count': len(entity_pair.mentions),
                'evidence': sentences[0],  # primary evidence
                'all_evidence': sentences,  # all supporting sentences
                'reasoning': result.get('reasoning', '')
            })
    
    return relations