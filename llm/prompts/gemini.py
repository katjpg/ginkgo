RELATION_PROMPT = '''Classify the semantic relation between [ENT1] and [ENT2] in this sentence:

"{exemplar}"

Relation types:
- uses: entity1 uses/employs/applies/utilizes entity2
- improves: entity1 improves/enhances/optimizes/boosts entity2
- evaluates: entity1 tests/benchmarks/evaluates/assesses entity2
- enables: entity1 enables/allows/facilitates/supports entity2
- proposes: entity1 proposes/introduces/presents entity2
- related: no clear specific relation

Return ONLY the relation type name, nothing else.'''
