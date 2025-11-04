RELATION_PROMPT = '''Classify the semantic relation between [ENT1] and [ENT2].

Sentence: "{exemplar}"

Relation types:

1. uses: [ENT1] uses [ENT2] as a component or internal mechanism
   Example: Transformer uses multi-head attention, LSTM uses gating mechanism
   Do NOT use for: hardware, infrastructure, or tasks
   Do NOT use if: [ENT1] and [ENT2] are listed in parallel (e.g., "A, B, and C all...")

2. improves: [ENT1] achieves better performance on [ENT2]
   Example: Our model improves F1 score, BERT improves accuracy
   Do NOT use for: methods evaluated on datasets
   Do NOT use if: [ENT1] is a metric or dataset

3. evaluates: [ENT1] is tested or assessed on [ENT2]
   Example: We evaluate on ImageNet, model tested on SQuAD
   Do NOT use for: metrics measuring performance
   Do NOT use if: [ENT1] is a metric or dataset

4. enables: [ENT1] facilitates or makes possible [ENT2]
   Example: Attention enables parallel computation, transformers enable scaling
   Do NOT use for: tasks that [ENT1] solves (use this only for technical capabilities)

5. proposes: [ENT1] introduces or presents [ENT2] as a contribution
   Example: We propose BERT, this paper introduces a new optimizer

6. related: Unclear or non-specific connection. Use this for coordinated alternatives
   Example: If [ENT1] and [ENT2] appear in parallel lists or are competing approaches

CRITICAL RULES:
- If [ENT1] appears in a list like "X, Y, and Z all...", those are typically related not uses
- If [ENT1] is a metric or dataset, return related
- If [ENT1] is infrastructure or hardware, return related

Return ONLY the relation type name in lowercase, nothing else.'''
