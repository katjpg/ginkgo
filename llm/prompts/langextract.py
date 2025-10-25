import textwrap
import langextract as lx


PROMPT = textwrap.dedent(
    """\
    Extract scientific entities from research abstracts using exact text spans.
    
    Entity types:
    
    task: specific research problems (e.g., "image classification", "named entity recognition")
    method: named systems/algorithms (e.g., "BERT", "ResNet", "Belief propagation algorithm")
    dataset: named benchmark collections (e.g., "ImageNet", "CoNLL2003")
    object: domain-specific entities being studied (e.g., "proteins", "biomedical entities")
    metric: performance measures (e.g., "F1 score", "accuracy", "BLEU")
    generic: anaphoric references with determiners (e.g., "this approach", "our model")
    other: technical concepts providing context (e.g., "neural architecture", "attention mechanism")
    
    Extraction rules:
    
    No overlapping spans.
    - Extract first occurrence only and skip repeated mentions of same entity.
    
    Span boundaries:
    - Extract entity mentions WITHOUT surrounding punctuation (quotes, parentheses)
    - For acronyms with expansions like "Retrieval-Augmented Generation (RAG)", extract ONLY the primary form
    - For methods with years like "BERT (2018)", extract ONLY the method name
    - Strip leading/trailing punctuation from extracted spans
    
    Acronym handling:
    - If full form appears: extract full form on first mention (e.g., "Retrieval-Augmented Generation")
    - If acronym appears: extract acronym only (e.g., "RAG")
    - Do NOT extract "X (Y)" format - choose one or the other based on context
    
    Do NOT extract:
    - Meta-discourse: "literature review", "related works", "prior work"
    - Meta-linguistic: "abbreviation", "definition", "term", "word", "full name"
    - Generic processes: "manual review", "rules", "normalization", "clean-up work"
    - Gerund processes: "normalized entities", "extracted features"
    - Entity type lists: when describing what a system/dataset extracts (e.g., "extracts person and location" Do NOT extract "person" and "location")
    - Standalone task verbs: "evaluation", "analysis", "testing", "method", without specific context
    - Vague references: "the machine", "methods" without determiner
    
    Generic type: Only "this/our/the/these" + noun referring to specific prior contribution."""
)

EXAMPLES = [
    lx.data.ExampleData(
        text="Image classification remains a fundamental challenge in computer vision. We propose ResNet to address this problem. ResNet uses residual connections to enable training of very deep networks. We evaluate our approach on ImageNet and achieve 93.1% top-5 accuracy.",
        extractions=[
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="Image classification",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="computer vision",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="ResNet",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="this problem",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="residual connections",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="our approach",
            ),
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="ImageNet",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="top-5 accuracy",
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Machine translation systems struggle with long sequences due to memory constraints. We introduce Linformer to overcome this limitation. The model approximates attention using low-rank matrix decomposition. Experiments on WMT 2014 demonstrate 28.4 BLEU score while reducing computational complexity.",
        extractions=[
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="Machine translation",
            ),
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="long sequences",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="Linformer",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="this limitation",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="The model",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="attention",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="low-rank matrix decomposition",
            ),
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="WMT 2014",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="BLEU score",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="computational complexity",
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Graph neural networks analyze structured data by message passing between nodes. However, over-smoothing limits their expressiveness in deep architectures. We propose GraphSAINT using importance sampling for scalable training. Our method is evaluated on Reddit social network with 233,000 nodes, achieving 97.1% F1 score.",
        extractions=[
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="Graph neural networks",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="structured data",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="message passing",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="nodes",
            ),
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="over-smoothing",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="deep architectures",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="GraphSAINT",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="importance sampling",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="Our method",
            ),
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="Reddit social network",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="F1 score",
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Object detection in autonomous vehicles requires real-time processing. YOLO achieves this through single-pass neural architecture. The system processes frames at 45 FPS on COCO dataset. This approach balances speed and accuracy for practical deployment.",
        extractions=[
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="Object detection",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="autonomous vehicles",
            ),
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="real-time processing",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="YOLO",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="single-pass neural architecture",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="The system",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="FPS",
            ),
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="COCO dataset",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="This approach",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="speed",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="accuracy",
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Protein structure prediction determines three-dimensional configurations from amino acid sequences. AlphaFold2 uses transformer architecture with attention mechanisms operating on residue pairs. The model achieves median GDT score of 92.4 on CASP14 benchmark, approaching experimental accuracy.",
        extractions=[
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="Protein structure prediction",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="three-dimensional configurations",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="amino acid sequences",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="AlphaFold2",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="transformer architecture",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="attention mechanisms",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="residue pairs",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="The model",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="GDT score",
            ),
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="CASP14 benchmark",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="experimental accuracy",
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Sentiment analysis classifies emotional tone in text. We fine-tune BERT on Twitter corpus containing 1.6 million tweets. Our system achieves 89.3% accuracy on test set, outperforming previous approaches by 3.2 percentage points.",
        extractions=[
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="Sentiment analysis",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="emotional tone",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="text",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="BERT",
            ),
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="Twitter corpus",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="tweets",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="Our system",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="accuracy",
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Speech recognition converts audio signals into text transcriptions. DeepSpeech employs recurrent neural networks with connectionist temporal classification loss. The model is trained on LibriSpeech dataset and evaluated using word error rate metric.",
        extractions=[
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="Speech recognition",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="audio signals",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="text transcriptions",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="DeepSpeech",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="recurrent neural networks",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="connectionist temporal classification loss",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="The model",
            ),
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="LibriSpeech dataset",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="word error rate",
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Reinforcement learning enables agents to learn optimal policies through environmental interaction. PPO stabilizes training using clipped surrogate objectives. We test this algorithm on Atari games, measuring performance by average episode reward across 49 environments.",
        extractions=[
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="Reinforcement learning",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="agents",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="optimal policies",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="environmental interaction",
            ),
            lx.data.Extraction(
                extraction_class="method",
                extraction_text="PPO",
            ),
            lx.data.Extraction(
                extraction_class="other",
                extraction_text="clipped surrogate objectives",
            ),
            lx.data.Extraction(
                extraction_class="generic",
                extraction_text="this algorithm",
            ),
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="Atari games",
            ),
            lx.data.Extraction(
                extraction_class="metric",
                extraction_text="episode reward",
            ),
        ],
    ),
    lx.data.ExampleData(
        text="CoNLL2003 only requires the extraction of person, organizations, and locations from the news. We propose biomedicine NER for academic papers.",
        extractions=[
            lx.data.Extraction(
                extraction_class="dataset",
                extraction_text="CoNLL2003",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="news",
            ),
            lx.data.Extraction(
                extraction_class="task",
                extraction_text="biomedicine NER",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="academic papers",
            ),
        ],
    ),
]