"""LangExtract prompts for joint entity and relationship extraction.

Provides PROMPT and EXAMPLES for extracting scientific entities AND relationships
from academic papers in a single pass.

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
"""

import langextract as lx


PROMPT = """Extract entities AND relationships using EXACT text spans:
CONTRIBUTION, PROBLEM, CLAIM, FINDING, RELATION

Entity Types:
- CONTRIBUTION: Named methods, models, frameworks, metrics, algorithms
- PROBLEM: Challenges, limitations, gaps
- CLAIM: Assertions about contributions
- FINDING: Empirical results, measurements
- RELATION: Relationship statements between entities

RELATION Attributes:
- source: source entity text
- target: target entity text  
- type: relationship category (derived_from, addresses, evaluates, compares_to, supports, uses, measured_by)
- context: optional context

Rules:
- Copy text verbatim - zero paraphrasing
- Extract entities first, then relations
- RELATION spans must contain both source and target
- Max 80 chars for entities, 150 for relations
- NO generic categories ("deep learning model", "lower-tier LLM")

Examples:
CONTRIBUTION: "Transformer" → category="model"
RELATION: "Transformer based on attention mechanisms" → source="Transformer", target="attention mechanisms", type="derived_from"
"""


EXAMPLES = [
    lx.data.ExampleData(
        text="A major advantage of HippoRAG over conventional RAG methods in multi-hop QA is its ability to perform multi-hop retrieval in a single step. We demonstrate this by measuring the percentage of queries where all the supporting passages are retrieved successfully, a feat that can only be accomplished through successful multi-hop reasoning. Table 6 below shows that the gap between our method and ColBERTv2, using the top-5 passages, increases even more from 3% to 6% on MuSiQue and from 20% to 38% on 2WikiMultiHopQA, suggesting that large improvements come from obtaining all supporting documents rather than achieving partially retrieval on more questions.",
        extractions=[
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="HippoRAG",
                attributes={
                    "category": "method",
                    "novelty": "novel",
                    "type": "retrieval",
                },
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="ColBERTv2",
                attributes={
                    "category": "method",
                    "novelty": "existing",
                    "type": "retrieval",
                },
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="percentage of queries",
                attributes={"category": "metric"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="gap increases from 3% to 6% on MuSiQue",
                attributes={"comparison": "outperforms", "value": "3% to 6%"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="from 20% to 38% on 2WikiMultiHopQA",
                attributes={"comparison": "outperforms", "value": "20% to 38%"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="advantage of HippoRAG over conventional RAG methods",
                attributes={
                    "source": "HippoRAG",
                    "target": "conventional RAG methods",
                    "type": "compares_to",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="gap between our method and ColBERTv2",
                attributes={
                    "source": "HippoRAG",
                    "target": "ColBERTv2",
                    "type": "compares_to",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="measuring the percentage of queries",
                attributes={
                    "source": "finding",
                    "target": "percentage of queries",
                    "type": "measured_by",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.",
        extractions=[
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="Transformer",
                attributes={
                    "category": "model",
                    "novelty": "novel",
                    "purpose": "machine translation",
                },
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="attention mechanisms",
                attributes={"category": "technique", "novelty": "existing"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="BLEU",
                attributes={
                    "category": "metric",
                    "purpose": "measure translation quality",
                },
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="achieves 28.4 BLEU on WMT 2014 English-to-German",
                attributes={"comparison": "outperforms", "value": "28.4"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="state-of-the-art BLEU score of 41.0",
                attributes={"comparison": "outperforms", "value": "41.0"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="Transformer, based solely on attention mechanisms",
                attributes={
                    "source": "Transformer",
                    "target": "attention mechanisms",
                    "type": "derived_from",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="achieves 28.4 BLEU",
                attributes={
                    "source": "Transformer",
                    "target": "BLEU",
                    "type": "measured_by",
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
                attributes={"scope": "generalization"},
            ),
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="do not effectively utilize limited available data",
                attributes={"scope": "data efficiency"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="lightweight few-shot NER framework",
                attributes={"category": "framework", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="instruction tuning template",
                attributes={"category": "method", "type": "preprocessing"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="data augmentation technique",
                attributes={"category": "method", "type": "augmentation"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="F1 score",
                attributes={"category": "metric", "purpose": "measure NER performance"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="comparable to state-of-the-art on few-shot tasks",
                attributes={"comparison": "matches", "value": "80.1"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="framework that addresses these challenges",
                attributes={
                    "source": "lightweight few-shot NER framework",
                    "target": "fail to generalize",
                    "type": "addresses",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="attaining an average F1 score",
                attributes={
                    "source": "finding",
                    "target": "F1 score",
                    "type": "measured_by",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Graph-based RAG (GraphRAG) leverages LLMs to organize RAG data into graphs, showing strong potential for gaining holistic insights from long-form documents. However, its standard implementation is overly complex for general use and lacks the ability to generate evidence-based responses, limiting its effectiveness in the medical field. To extend the capabilities of GraphRAG to the medical domain, we propose unique Triple Graph Construction and U-Retrieval techniques over it",
        extractions=[
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="GraphRAG",
                attributes={
                    "category": "method",
                    "novelty": "existing",
                    "type": "retrieval",
                },
            ),
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="overly complex for general use",
                attributes={"scope": "complexity"},
            ),
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="lacks evidence-based responses",
                attributes={"scope": "quality"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="Triple Graph Construction",
                attributes={"category": "method", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="U-Retrieval techniques",
                attributes={"category": "method", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="Triple Graph Construction and U-Retrieval techniques over it",
                attributes={
                    "source": "Triple Graph Construction",
                    "target": "GraphRAG",
                    "type": "derived_from",
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Retrieval-Augmented Generation (RAG) integrates user queries with a collection of pertinent documents sourced from an external knowledge database, incorporating two essential elements: the Retrieval Component and the Generation Component.",
        extractions=[
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="Retrieval-Augmented Generation (RAG)",
                attributes={"category": "framework", "novelty": "existing"},
            ),
        ],
    ),
]
