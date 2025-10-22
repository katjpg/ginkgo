"""LangExtract prompts and examples for academic entity extraction.

Provides PROMPT and EXAMPLES for extracting scientific entities from academic papers.

Example Sources
---------------
Text excerpts are from:

1. Bernal et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory
   for Large Language Models. arXiv:2405.14831

2. Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762

3. Rengarajan et al. (2025). PANER: A Paraphrase-Augmented Framework for
   Low-Resource Named Entity Recognition. arXiv:2510.17720

4. Wu et al. (2024). Medical Graph RAG: Towards Safe Medical Large Language Model
   via Graph Retrieval-Augmented Generation. arXiv:2408.04187

5. Guo et al. (2025). LightRAG: Simple and Fast Retrieval-Augmented Generation.
   arXiv:2410.05779

Notes
-----
All excerpts used for educational purposes to demonstrate entity extraction patterns.
"""

import langextract as lx


PROMPT = """Extract entities using EXACT text spans from the source:
CONCEPT, METHOD, PROBLEM, CLAIM, FINDING, METRIC

Rules:
- Copy text verbatim - zero paraphrasing
- Max 80 chars for concepts/methods, 150 for claims/findings
- Extract only central contributions
- Attributes: concise phrases (<50 chars)
- No overlapping spans

Examples of good extraction length:
CONCEPT: "Transformer" (11 chars)
METHOD: "multi-hop retrieval" (19 chars)  
CLAIM: "outperforms baselines on multi-hop QA" (38 chars)
"""


EXAMPLES = [
    lx.data.ExampleData(
        text="A major advantage of HippoRAG over conventional RAG methods in multi-hop QA is its ability to perform multi-hop retrieval in a single step. We demonstrate this by measuring the percentage of queries where all the supporting passages are retrieved successfully, a feat that can only be accomplished through successful multi-hop reasoning. Table 6 below shows that the gap between our method and ColBERTv2, using the top-5 passages, increases even more from 3% to 6% on MuSiQue and from 20% to 38% on 2WikiMultiHopQA, suggesting that large improvements come from obtaining all supporting documents rather than achieving partially retrieval on more questions.",
        extractions=[
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="HippoRAG",
                attributes={
                    "purpose": "multi-hop retrieval",
                    "applied_to": "multi-hop QA",
                },
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="multi-hop retrieval in a single step",
                attributes={
                    "purpose": "retrieve supporting passages",
                    "applied_to": "QA",
                },
            ),
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="multi-hop reasoning",
                attributes={"type": "technique", "novelty": "existing"},
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="ColBERTv2",
                attributes={"purpose": "baseline retrieval"},
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="percentage of queries",
                attributes={
                    "evaluates": "retrieval completeness",
                    "direction": "higher_better",
                },
            ),
            lx.data.Extraction(
                extraction_class="claim",
                extraction_text="major advantage of HippoRAG over conventional RAG methods",
                attributes={"evidence_type": "empirical", "supports": "HippoRAG"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="gap increases from 3% to 6% on MuSiQue",
                attributes={"comparison": "outperforms", "baseline": "ColBERTv2"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="from 20% to 38% on 2WikiMultiHopQA",
                attributes={"comparison": "outperforms", "baseline": "ColBERTv2"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.",
        extractions=[
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="Transformer",
                attributes={
                    "type": "model",
                    "based_on": "attention mechanisms",
                    "novelty": "novel",
                },
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="Transformer",
                attributes={
                    "purpose": "machine translation",
                    "applied_to": "WMT tasks",
                },
            ),
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="attention mechanisms",
                attributes={"type": "technique", "novelty": "existing"},
            ),
            lx.data.Extraction(
                extraction_class="claim",
                extraction_text="Transformer based solely on attention mechanisms",
                attributes={"evidence_type": "theoretical", "supports": "Transformer"},
            ),
            lx.data.Extraction(
                extraction_class="claim",
                extraction_text="superior in quality while more parallelizable",
                attributes={"evidence_type": "empirical", "supports": "Transformer"},
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="BLEU",
                attributes={
                    "evaluates": "translation quality",
                    "units": "score",
                    "direction": "higher_better",
                },
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="achieves 28.4 BLEU on WMT 2014 English-to-German",
                attributes={
                    "comparison": "outperforms",
                    "baseline": "existing best results",
                },
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="state-of-the-art BLEU score of 41.0",
                attributes={
                    "comparison": "outperforms",
                    "baseline": "literature best models",
                },
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="training costs",
                attributes={
                    "evaluates": "efficiency",
                    "units": "cost",
                    "direction": "lower_better",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="While zero-shot and instruction-tuned approaches have made progress, they often fail to generalize to domain-specific entities and do not effectively utilize limited available data. We present a lightweight few-shot NER framework that addresses these challenges through two key innovations: (1) a new instruction tuning template with a simplified output format that combines principles from prior IT approaches to leverage the large context window of recent state-of-the-art LLMs; (2) introducing a strategic data augmentation technique that preserves entity information while paraphrasing the surrounding context, thereby expanding our training data without compromising semantic relationships. Experiments on benchmark datasets show that our method achieves performance comparable to state-of-the-art models on few-shot and zero-shot tasks, with our few-shot approach attaining an average F1 score of 80.1 on the CrossNER datasets.",
        extractions=[
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="fail to generalize to domain-specific entities",
                attributes={
                    "scope": "generalization",
                    "affects": "zero-shot approaches",
                },
            ),
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="do not effectively utilize limited available data",
                attributes={
                    "scope": "data efficiency",
                    "affects": "instruction-tuned approaches",
                },
            ),
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="lightweight few-shot NER framework",
                attributes={"type": "framework", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="lightweight few-shot NER framework",
                attributes={
                    "purpose": "address generalization",
                    "applied_to": "domain-specific NER",
                },
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="instruction tuning template",
                attributes={
                    "purpose": "leverage context window",
                    "applied_to": "few-shot NER",
                },
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="data augmentation technique",
                attributes={
                    "purpose": "preserve entities while paraphrasing",
                    "applied_to": "NER training",
                },
            ),
            lx.data.Extraction(
                extraction_class="claim",
                extraction_text="addresses challenges through two key innovations",
                attributes={
                    "evidence_type": "theoretical",
                    "supports": "few-shot NER framework",
                },
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="F1 score",
                attributes={
                    "evaluates": "NER performance",
                    "units": "percentage",
                    "direction": "higher_better",
                },
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="comparable to state-of-the-art on few-shot tasks",
                attributes={
                    "comparison": "matches",
                    "baseline": "state-of-the-art models",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Graph-based RAG (GraphRAG) leverages LLMs to organize RAG data into graphs, showing strong potential for gaining holistic insights from long-form documents. However, its standard implementation is overly complex for general use and lacks the ability to generate evidence-based responses, limiting its effectiveness in the medical field. To extend the capabilities of GraphRAG to the medical domain, we propose unique Triple Graph Construction and U-Retrieval techniques over it",
        extractions=[
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="GraphRAG",
                attributes={
                    "purpose": "organize RAG data into graphs",
                    "applied_to": "long-form documents",
                },
            ),
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="overly complex for general use",
                attributes={"scope": "complexity", "affects": "GraphRAG"},
            ),
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="lacks evidence-based responses",
                attributes={"scope": "quality", "affects": "GraphRAG"},
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="Triple Graph Construction",
                attributes={
                    "purpose": "extend GraphRAG",
                    "applied_to": "medical domain",
                },
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="U-Retrieval techniques",
                attributes={
                    "purpose": "extend GraphRAG",
                    "applied_to": "medical domain",
                },
            ),
            lx.data.Extraction(
                extraction_class="claim",
                extraction_text="extend GraphRAG to medical domain",
                attributes={
                    "evidence_type": "theoretical",
                    "supports": "Triple Graph Construction",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Retrieval-Augmented Generation (RAG) integrates user queries with a collection of pertinent documents sourced from an external knowledge database, incorporating two essential elements: the Retrieval Component and the Generation Component. 1) The retrieval component is responsible for fetching relevant documents or information from the external knowledge database. It identifies and retrieves the most pertinent data based on the input query. 2) After the retrieval process, the generation component takes the retrieved information and generates coherent, contextually relevant responses",
        extractions=[
            lx.data.Extraction(
                extraction_class="concept",
                extraction_text="Retrieval-Augmented Generation (RAG)",
                attributes={"type": "framework", "novelty": "existing"},
            ),
        ],
    ),
]