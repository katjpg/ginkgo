"""Type-aware entity pair filtering by semantic category."""

from itertools import combinations
from typing import Sequence

from models.entities import EntityType


class TypeConfig:
    """Configuration for type-aware filtering."""
    
    def __init__(self):
        self.meaningful_pairs: set[tuple[EntityType, EntityType]] = {
            (EntityType.TASK, EntityType.METHOD),
            (EntityType.METHOD, EntityType.DATASET),
            (EntityType.TASK, EntityType.DATASET),
            (EntityType.METHOD, EntityType.METHOD),
            (EntityType.OTHER, EntityType.TASK),
            (EntityType.METHOD, EntityType.METRIC),
            
            (EntityType.METHOD, EntityType.OTHER),    
            (EntityType.OTHER, EntityType.OTHER),      
            (EntityType.OTHER, EntityType.METRIC),     
            (EntityType.OTHER, EntityType.DATASET),  
            (EntityType.OTHER, EntityType.TASK),      
        }
        self.skip_generic = True



class PairFilter:
    """Filter entity pairs by meaningful type combinations."""
    
    def __init__(self, config: TypeConfig | None = None):
        self.config = config or TypeConfig()
    
    def generate(self, entities: Sequence[dict]) -> list[tuple[str, str]]:
        """Generate candidate pairs respecting type constraints."""
        candidates = []
        
        for e1, e2 in combinations(entities, 2):
            t1 = EntityType(e1['type'])
            t2 = EntityType(e2['type'])
            
            if self.config.skip_generic:
                if t1 == EntityType.GENERIC or t2 == EntityType.GENERIC:
                    continue
            
            if (t1, t2) in self.config.meaningful_pairs or \
               (t2, t1) in self.config.meaningful_pairs:
                candidates.append((e1['text'], e2['text']))
        
        return candidates
    
    def filter_relations(self, 
                        relations: Sequence, 
                        entity_types: dict[str, EntityType]) -> list:
        """Filter relations by meaningful type pairs."""
        filtered = []
        
        for rel in relations:
            e1_key = getattr(rel, 'e1', None) or getattr(rel, 'e_i', None)
            e2_key = getattr(rel, 'e2', None) or getattr(rel, 'e_j', None)
            
            if e1_key and e2_key: 
                t1 = entity_types.get(e1_key)
                t2 = entity_types.get(e2_key)
                
                if t1 and t2:
                    if (t1, t2) in self.config.meaningful_pairs or \
                    (t2, t1) in self.config.meaningful_pairs:
                        filtered.append(rel)
        
        return filtered

