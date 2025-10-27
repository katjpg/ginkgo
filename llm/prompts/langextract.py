import textwrap
import langextract as lx

PROMPT = textwrap.dedent(
    """\
    Extract scientific entities from research text. Return each entity with its type and exact text span.
    
    ENTITY TYPES:
    
    1. method - Named algorithms, models, systems with proper names, or model categories
       Extract: BERT, ResNet-50, Mask R-CNN, AlphaFold2, GPT-4, BiLSTM-CRF, LLM, Graph neural networks
       Note: LLM is METHOD not OBJECT (it is a category of models)
    
    2. dataset - Named benchmark collections or corpora
       Extract: ImageNet, CoNLL2003, HotpotQA, MNIST, WMT 2014, COCO, Cityscapes
    
    3. metric - Specific performance measures
       Extract: F1 score, accuracy, BLEU, precision, recall, mAP, top-5 accuracy, GDT-TS
    
    4. task - Specific research problems or objectives
       Extract: named entity recognition, image classification, question answering, protein folding
    
    5. object - Domain entities being studied or manipulated
       Extract: proteins, knowledge graphs, neural networks, amino acid sequences, queries, entities
    
    6. generic - Anaphoric references that clearly refer to a specific previously mentioned named entity
       Extract: this approach, our method, The model, these results, Our experiments
       Must have determiner AND reference a specific named entity mentioned earlier
       Do NOT extract: overall workflow, design rationale, each component, the following, an example
    
    7. other - Named technical concepts with proper nouns or well-defined technical terms
       Extract: transformer architecture, attention mechanisms, residual connections, Adam optimizer, C-HNSW index
       Must contain specific identifiers or be well-defined named concepts
       Do NOT extract: graph-based RAG approach, neural network-based method (descriptive phrases)
    
    EXTRACTION RULES:
    
    Extract ALL occurrences of each entity to enable frequency-based filtering.
    
    Extract exact text spans without surrounding punctuation, quotes, or parentheses.
    
    For acronyms with expansions like "Retrieval-Augmented Generation (RAG)", extract ONLY the full form OR acronym, not both.
    
    When an entity fits multiple types, use the highest priority type from the list above.
    
    SKIP THESE:
    
    Abstract nouns: efficiency, reusability, performance, infrastructure, speed
    Generic plurals: models, methods, datasets, features, parameters, vectors, architectures
    Section labels or abbreviations: VR, LR, AR, C1, C2, Section 3
    Figure or table references: Figure 3, Table 1, Figure 2a
    Evaluation dimensions: Comprehensiveness, Diversity, Empowerment, Overall
    Discourse markers: the following, an example, overall workflow, design rationale, each component
    Descriptive method phrases: graph-based RAG approach, neural network-based method, traditional methods
    Variables or symbols: x, y, threshold, weight, alpha, beta
    Process descriptions: training process, evaluation phase, indexing stage
    Adjective plus generic noun: traditional methods, deep learning models, pre-trained models
    Meta-discourse: prior work, related work, literature review
    Vague descriptors: relevant information, high-level, state-of-the-art
    Generic technical terms: query vector, weight matrix, loss function, exact matching
    
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
            lx.data.Extraction(extraction_class="object", extraction_text="entities"),
            lx.data.Extraction(extraction_class="object", extraction_text="text"),
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
            lx.data.Extraction(extraction_class="object", extraction_text="queries"),
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
            lx.data.Extraction(extraction_class="object", extraction_text="objects"),
            lx.data.Extraction(extraction_class="object", extraction_text="images"),
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
            lx.data.Extraction(extraction_class="object", extraction_text="text"),
            lx.data.Extraction(extraction_class="object", extraction_text="languages"),
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

