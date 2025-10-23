RELATIONSHIP_PROMPT = """Given the following text and extracted entities, identify relationships between entities.

Relationship Types:
- DERIVED_FROM: (Contribution → Contribution) - one contribution builds on another
- ADDRESSES: (Contribution → Problem) - contribution solves/addresses problem
- EVALUATES: (Finding → Contribution) - finding measures contribution performance
- COMPARES_TO: (Finding → Contribution) - finding compares contributions
- SUPPORTS: (Claim → Contribution) or (Finding → Claim) - evidence relationships
- USES: (Contribution → Contribution) - one contribution uses another
- MEASURED_BY: (Finding → Contribution[metric]) - finding uses specific metric

Rules:
- Only create relationships between entities that appear in the entity list
- Use exact entity IDs from the provided list
- Each relationship needs: source_id, target_id, relation_type
- Provide confidence score (0.0-1.0) for each relationship

Example Input:
Text: "Our HippoRAG method achieves 28.4 BLEU, outperforming ColBERTv2."
Entities:
- E1: HippoRAG (contribution, method)
- E2: BLEU (contribution, metric)
- E3: ColBERTv2 (contribution, method)
- E4: "achieves 28.4 BLEU" (finding)

Example Output:
Relationships:
- source: E4, target: E1, type: EVALUATES, confidence: 1.0
- source: E4, target: E3, type: COMPARES_TO, confidence: 1.0
- source: E4, target: E2, type: MEASURED_BY, confidence: 1.0
"""
