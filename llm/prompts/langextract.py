import langextract as lx


PROMPT = """Extract research information relevant to scientific discourse including problem, claim, contribution, finding, and relation in the order they appear in the text. Do NOT extract generic entities. Do NOT paraphrase or overlap entities. Provide meaningful attributes for each entity to add context."""


EXAMPLES = [
    lx.data.ExampleData(
        text="Deep learning models have revolutionized computer vision in recent years. Our proposed ResNet architecture uses residual connections to solve the degradation problem in very deep networks. We train ResNet on ImageNet and achieve 93.1% top-5 accuracy.",
        extractions=[
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="ResNet",
                attributes={"category": "model", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="residual connections",
                attributes={"category": "technique", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="degradation problem in very deep networks",
                attributes={"scope": "training"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="ImageNet",
                attributes={"category": "dataset"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="achieve 93.1% top-5 accuracy",
                attributes={"value": "93.1%"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="ResNet uses residual connections",
                attributes={
                    "source": "ResNet",
                    "target": "residual connections",
                    "type": "uses",
                },
            ),
        ],
    ),
    
    lx.data.ExampleData(
        text="Transformer-based models suffer from quadratic memory complexity with respect to sequence length. We propose Linformer to address this limitation. Linformer approximates self-attention using low-rank decomposition, reducing complexity from O(n²) to O(n).",
        extractions=[
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="quadratic memory complexity with respect to sequence length",
                attributes={"scope": "efficiency"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="Linformer",
                attributes={"category": "model", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="low-rank decomposition",
                attributes={"category": "technique"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="reducing complexity from O(n²) to O(n)",
                attributes={"comparison": "reduces", "value": "O(n²) to O(n)"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="Linformer to address this limitation",
                attributes={
                    "source": "Linformer",
                    "target": "quadratic memory complexity",
                    "type": "addresses",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="Linformer approximates self-attention using low-rank decomposition",
                attributes={
                    "source": "Linformer",
                    "target": "low-rank decomposition",
                    "type": "uses",
                },
            ),
        ],
    ),
    
    lx.data.ExampleData(
        text="We claim that attention mechanisms enable models to capture long-range dependencies more effectively. To test this hypothesis, we train Transformer on WMT 2014 English-to-German. Our model achieves 28.4 BLEU, outperforming previous models by 2.0 points. These results support our claim.",
        extractions=[
            lx.data.Extraction(
                extraction_class="claim",
                extraction_text="attention mechanisms enable models to capture long-range dependencies",
                attributes={"evidence_type": "hypothesis"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="Transformer",
                attributes={"category": "model", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="WMT 2014 English-to-German",
                attributes={"category": "dataset"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="BLEU",
                attributes={"category": "metric"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="achieves 28.4 BLEU",
                attributes={"value": "28.4"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="outperforming previous models by 2.0 points",
                attributes={"comparison": "outperforms", "value": "2.0"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="These results support our claim",
                attributes={
                    "source": "Transformer",
                    "target": "attention mechanisms",
                    "type": "supports",
                },
            ),
        ],
    ),
    
    lx.data.ExampleData(
        text="Building upon GCN, we develop GraphSAINT with importance sampling to scale training to large graphs. GraphSAINT is trained on Reddit dataset. Compared to GCN, GraphSAINT achieves 97.1% F1 versus 93.2%.",
        extractions=[
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="GCN",
                attributes={"category": "model", "novelty": "existing"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="GraphSAINT",
                attributes={"category": "model", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="importance sampling",
                attributes={"category": "technique"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="Reddit dataset",
                attributes={"category": "dataset"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="GraphSAINT achieves 97.1% F1 versus 93.2%",
                attributes={"comparison": "outperforms", "value": "97.1% vs 93.2%"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="Building upon GCN, we develop GraphSAINT",
                attributes={
                    "source": "GraphSAINT",
                    "target": "GCN",
                    "type": "derived_from",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="GraphSAINT with importance sampling",
                attributes={
                    "source": "GraphSAINT",
                    "target": "importance sampling",
                    "type": "uses",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="trained on Reddit dataset",
                attributes={
                    "source": "GraphSAINT",
                    "target": "Reddit dataset",
                    "type": "evaluates",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="Compared to GCN, GraphSAINT achieves 97.1%",
                attributes={
                    "source": "GraphSAINT",
                    "target": "GCN",
                    "type": "compares_to",
                },
            ),
        ],
    ),
    
    lx.data.ExampleData(
        text="Reinforcement learning agents struggle with sparse reward environments. We hypothesize that curiosity-driven exploration improves learning efficiency. Our PPO experiments on Montezuma's Revenge show agents reach 8,500 versus 400 for baseline. This validates our hypothesis.",
        extractions=[
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="sparse reward environments",
                attributes={"scope": "exploration"},
            ),
            lx.data.Extraction(
                extraction_class="claim",
                extraction_text="curiosity-driven exploration improves learning efficiency",
                attributes={"evidence_type": "hypothesis"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="PPO",
                attributes={"category": "algorithm"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="Montezuma's Revenge",
                attributes={"category": "environment"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="agents reach 8,500 versus 400",
                attributes={"comparison": "outperforms", "value": "8,500 vs 400"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="PPO experiments on Montezuma's Revenge",
                attributes={
                    "source": "PPO",
                    "target": "Montezuma's Revenge",
                    "type": "evaluates",
                },
            ),
        ],
    ),
    
    lx.data.ExampleData(
        text="Existing diffusion models require hundreds of sampling steps. DDIM addresses this through deterministic sampling trajectories. DDIM reduces steps from 1000 to 50.",
        extractions=[
            lx.data.Extraction(
                extraction_class="problem",
                extraction_text="require hundreds of sampling steps",
                attributes={"scope": "efficiency"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="DDIM",
                attributes={"category": "method", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="deterministic sampling trajectories",
                attributes={"category": "technique"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="DDIM reduces steps from 1000 to 50",
                attributes={"comparison": "reduces", "value": "1000 to 50"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="DDIM addresses this through deterministic sampling",
                attributes={
                    "source": "DDIM",
                    "target": "deterministic sampling trajectories",
                    "type": "uses",
                },
            ),
        ],
    ),
    
    lx.data.ExampleData(
        text="CLIP learns by aligning images with text. Zero-shot evaluation on ImageNet yields 76.2% accuracy. CLIP demonstrates strong transfer capabilities.",
        extractions=[
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="CLIP",
                attributes={"category": "model", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="ImageNet",
                attributes={"category": "dataset"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="Zero-shot evaluation on ImageNet yields 76.2%",
                attributes={"value": "76.2%"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="evaluation on ImageNet",
                attributes={
                    "source": "CLIP",
                    "target": "ImageNet",
                    "type": "evaluates",
                },
            ),
        ],
    ),
    
    lx.data.ExampleData(
        text="CodeBERT extends BERT by pretraining on programming languages. We evaluate CodeBERT on CodeSearchNet and achieve 86.9 mean reciprocal rank. Compared to RoBERTa, CodeBERT shows 14% improvement.",
        extractions=[
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="CodeBERT",
                attributes={"category": "model", "novelty": "novel"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="BERT",
                attributes={"category": "model", "novelty": "existing"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="CodeSearchNet",
                attributes={"category": "dataset"},
            ),
            lx.data.Extraction(
                extraction_class="contribution",
                extraction_text="RoBERTa",
                attributes={"category": "model", "novelty": "existing"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="achieve 86.9 mean reciprocal rank",
                attributes={"value": "86.9"},
            ),
            lx.data.Extraction(
                extraction_class="finding",
                extraction_text="CodeBERT shows 14% improvement",
                attributes={"comparison": "improves", "value": "14%"},
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="CodeBERT extends BERT",
                attributes={
                    "source": "CodeBERT",
                    "target": "BERT",
                    "type": "derived_from",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="evaluate CodeBERT on CodeSearchNet",
                attributes={
                    "source": "CodeBERT",
                    "target": "CodeSearchNet",
                    "type": "evaluates",
                },
            ),
            lx.data.Extraction(
                extraction_class="relation",
                extraction_text="Compared to RoBERTa",
                attributes={
                    "source": "CodeBERT",
                    "target": "RoBERTa",
                    "type": "compares_to",
                },
            ),
        ],
    ),
]