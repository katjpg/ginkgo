import textwrap
import langextract as lx

PROMPT = textwrap.dedent(
    """\
    You are a PhD researcher that extracts contextually rich scientific entities from research text. Return each entity with its type and exact text span.
    
    ENTITY TYPES:
    
    1. method - Specific named algorithms, models, or techniques that can be trained, executed, or implemented as complete standalone systems.
       Do NOT extract descriptive method phrases including graph-based RAG approach, neural network-based method, traditional methods.
    
    2. dataset - Named benchmark collections or corpora.
    
    3. metric - Specific performance measures used to evaluate the research, including named metrics with established definitions or clear task-specific formulations.
    
    4. task - Specific research problems or well-defined objectives, including domain-specific problem formulations.
    
    5. object - Domain-specific subject matter, phenomena, or entities that are being investigated, studied, or are the main focus of the research. 
    Not generic computational infrastructure.
    
    6. generic - Anaphoric references that clearly refer to a specific previously mentioned named entity.
       Must have determiner AND reference a specific named entity mentioned earlier.
    
    7. other - Specific theory, frameworks, or domain concepts.
    
    EXTRACTION RULES:
    - Extract ALL occurrences of each entity.
    - Extract exact text spans without surrounding punctuation, quotes, or parentheses.
    - For acronyms with expansions like "Retrieval-Augmented Generation (RAG)", extract ONLY the full form OR acronym NOT both.
    - When an entity fits multiple types, use the top priority type from the list above.
    
    
    Do NOT extract:
    - Generic objects: question, answer, node, entities, graph, layer, prompt, text, information.
    - Generic plurals: models, methods, datasets, features, parameters, vectors, architectures.
    - Vague descriptors: relevant information, high-level, state-of-the-art.
    - Abstract nouns: efficiency, reusability, performance, infrastructure, speed.
    - Section labels, abbreviations, or references: VR, AR, C1, C2, Section 3, Table 1, Figure 2a.
    - Generic metrics: Comprehensiveness, Diversity, Empowerment, Overall.
    - Discourse markers: the following, an example, overall workflow, design rationale, each component.
    - Descriptive method phrases: graph-based RAG approach, neural network-based method, LLM-based methods.
    - Variables or symbols: x, y, threshold, weight, alpha, beta.
    - Process descriptions: training process, evaluation phase, indexing stage.
    - Meta-discourse: prior work, related work, literature review.
    - Implementation variables: candidate expansion queue Q, dynamic neighbor set K.

    
    OUTPUT FORMAT:
    
    Return entities in order of appearance with type and exact text.
    """
)

EXAMPLES = [
    lx.data.ExampleData(
        text="Image classification remains a fundamental challenge in computer vision. We propose ResNet to address this problem. ResNet uses residual connections to enable training of very deep networks. We evaluate our approach on ImageNet and achieve 93.1% top-5 accuracy.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Image classification"),
            lx.data.Extraction(extraction_class="method", extraction_text="ResNet"),
            lx.data.Extraction(extraction_class="generic", extraction_text="this problem"),
            lx.data.Extraction(extraction_class="method", extraction_text="ResNet"),
            lx.data.Extraction(extraction_class="other", extraction_text="residual connections"),
            lx.data.Extraction(extraction_class="object", extraction_text="deep networks"),
            lx.data.Extraction(extraction_class="generic", extraction_text="our approach"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="ImageNet"),
            lx.data.Extraction(extraction_class="metric", extraction_text="top-5 accuracy"),
        ],
    ),
    lx.data.ExampleData(
        text="Named entity recognition extracts entities from text. We introduce BiLSTM-CRF using contextualized embeddings. The model achieves 91.2% F1 on CoNLL2003.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Named entity recognition"),
            lx.data.Extraction(extraction_class="method", extraction_text="BiLSTM-CRF"),
            lx.data.Extraction(extraction_class="other", extraction_text="contextualized embeddings"),
            lx.data.Extraction(extraction_class="generic", extraction_text="The model"),
            lx.data.Extraction(extraction_class="metric", extraction_text="F1"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="CoNLL2003"),
        ],
    ),
    lx.data.ExampleData(
        text="Graph neural networks process relational data through message passing. GCN aggregates features using spectral convolution. Our experiments on Cora dataset show 81.5% classification accuracy.",
        extractions=[
            lx.data.Extraction(extraction_class="method", extraction_text="Graph neural networks"),
            lx.data.Extraction(extraction_class="object", extraction_text="relational data"),
            lx.data.Extraction(extraction_class="other", extraction_text="message passing"),
            lx.data.Extraction(extraction_class="method", extraction_text="GCN"),
            lx.data.Extraction(extraction_class="other", extraction_text="spectral convolution"),
            lx.data.Extraction(extraction_class="generic", extraction_text="Our experiments"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="Cora dataset"),
            lx.data.Extraction(extraction_class="metric", extraction_text="classification accuracy"),
        ],
    ),
    lx.data.ExampleData(
        text="Question answering systems retrieve information to answer queries. BERT-QA uses transformer encoders for passage ranking. We evaluate on SQuAD 2.0 achieving 89.8% exact match and 92.3% F1.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Question answering"),
            lx.data.Extraction(extraction_class="method", extraction_text="BERT-QA"),
            lx.data.Extraction(extraction_class="other", extraction_text="transformer encoders"),
            lx.data.Extraction(extraction_class="task", extraction_text="passage ranking"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="SQuAD 2.0"),
            lx.data.Extraction(extraction_class="metric", extraction_text="exact match"),
            lx.data.Extraction(extraction_class="metric", extraction_text="F1"),
        ],
    ),
    lx.data.ExampleData(
        text="Protein folding predicts three-dimensional structures from amino acid sequences. AlphaFold2 leverages attention mechanisms and multiple sequence alignments. The system obtains median GDT-TS of 92.4 on CASP14.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Protein folding"),
            lx.data.Extraction(extraction_class="object", extraction_text="three-dimensional structures"),
            lx.data.Extraction(extraction_class="object", extraction_text="amino acid sequences"),
            lx.data.Extraction(extraction_class="method", extraction_text="AlphaFold2"),
            lx.data.Extraction(extraction_class="other", extraction_text="attention mechanisms"),
            lx.data.Extraction(extraction_class="other", extraction_text="multiple sequence alignments"),
            lx.data.Extraction(extraction_class="generic", extraction_text="The system"),
            lx.data.Extraction(extraction_class="metric", extraction_text="GDT-TS"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="CASP14"),
        ],
    ),
    lx.data.ExampleData(
        text="Object detection localizes and classifies objects in images. YOLO performs single-stage detection using convolutional layers. This approach processes frames at 45 FPS on COCO achieving 33.0 mAP.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Object detection"),
            lx.data.Extraction(extraction_class="method", extraction_text="YOLO"),
            lx.data.Extraction(extraction_class="task", extraction_text="single-stage detection"),
            lx.data.Extraction(extraction_class="other", extraction_text="convolutional layers"),
            lx.data.Extraction(extraction_class="generic", extraction_text="This approach"),
            lx.data.Extraction(extraction_class="metric", extraction_text="FPS"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="COCO"),
            lx.data.Extraction(extraction_class="metric", extraction_text="mAP"),
        ],
    ),
    lx.data.ExampleData(
        text="Machine translation converts text between languages. Transformer architecture uses self-attention for sequence modeling. We train on WMT 2014 English-German obtaining 28.4 BLEU score.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Machine translation"),
            lx.data.Extraction(extraction_class="other", extraction_text="Transformer architecture"),
            lx.data.Extraction(extraction_class="other", extraction_text="self-attention"),
            lx.data.Extraction(extraction_class="task", extraction_text="sequence modeling"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="WMT 2014 English-German"),
            lx.data.Extraction(extraction_class="metric", extraction_text="BLEU score"),
        ],
    ),
    lx.data.ExampleData(
        text="Semantic segmentation assigns class labels to each pixel. U-Net employs encoder-decoder architecture with skip connections. The model is trained on Cityscapes and evaluated using mean IoU metric.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Semantic segmentation"),
            lx.data.Extraction(extraction_class="object", extraction_text="class labels"),
            lx.data.Extraction(extraction_class="object", extraction_text="pixel"),
            lx.data.Extraction(extraction_class="method", extraction_text="U-Net"),
            lx.data.Extraction(extraction_class="other", extraction_text="encoder-decoder architecture"),
            lx.data.Extraction(extraction_class="other", extraction_text="skip connections"),
            lx.data.Extraction(extraction_class="generic", extraction_text="The model"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="Cityscapes"),
            lx.data.Extraction(extraction_class="metric", extraction_text="mean IoU"),
        ],
    ),
    lx.data.ExampleData(
        text="Drug discovery identifies compounds that bind to target proteins. We apply DockNet for molecular docking predictions. Our method outperforms AutoDock Vina on PDBbind dataset with RMSD below 2 angstroms.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Drug discovery"),
            lx.data.Extraction(extraction_class="object", extraction_text="compounds"),
            lx.data.Extraction(extraction_class="object", extraction_text="target proteins"),
            lx.data.Extraction(extraction_class="method", extraction_text="DockNet"),
            lx.data.Extraction(extraction_class="task", extraction_text="molecular docking predictions"),
            lx.data.Extraction(extraction_class="generic", extraction_text="Our method"),
            lx.data.Extraction(extraction_class="method", extraction_text="AutoDock Vina"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="PDBbind dataset"),
            lx.data.Extraction(extraction_class="metric", extraction_text="RMSD"),
        ],
    ),
]
