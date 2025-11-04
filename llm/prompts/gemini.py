RELATION_PROMPT = '''Classify the semantic relation between ENT1 and ENT2.

Sentence: "{exemplar}"

Relation types:

1. uses: A method uses another method as a component or internal mechanism
   Example: Transformer uses multi-head attention, LSTM uses gating mechanism
   Do NOT use for: hardware, infrastructure, or tasks that a method solves
   Valid: method to method (compositional only)

2. improves: A method achieves better performance on a specific metric or benchmark
   Example: Our model improves F1 score, BERT improves accuracy
   Do NOT use for: methods evaluated on datasets, datasets improved by methods
   Valid: method to metric

3. evaluates: A method is tested or assessed on a dataset or benchmark
   Example: We evaluate on ImageNet, model tested on SQuAD, parser benchmarked on WSJ
   Do NOT use for: metrics measuring performance, methods measuring metrics
   Valid: method to dataset

4. enables: A method facilitates, supports, or makes possible a capability or task
   Example: Attention enables parallel computation, transformers enable scaling
   Do NOT use for: methods that solve tasks (use enables only for technical capabilities)
   Valid: method to other or method to task

5. proposes: Introduces or presents something as a novel contribution
   Example: We propose BERT, this paper introduces a new optimizer
   Valid: any

6. related: Unclear connection or no specific relation identified

CRITICAL RULES:
- If ENT1 is a metric or dataset, return related
- If ENT1 solves a task, use enables or proposes, not uses
- If ENT1 runs on hardware infrastructure, use related
- Reverse the relation if it appears backwards in the sentence

Return ONLY the relation type name, nothing else.'''


