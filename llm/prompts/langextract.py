import textwrap
import langextract as lx


PROMPT = textwrap.dedent(
    """\
    You are a PhD researcher extracting scientific entities from research papers. Return each entity with its type and exact text span.
    
    ## ENTITY TYPES:
    
    1. method - Named algorithms, models, or complete systems.
       Examples: BERT, ResNet, BiLSTM-CRF, AlphaFold2, YOLO, U-Net, Adam optimizer
       Do NOT use for: architectural components (encoder-decoder), paradigms (transfer learning)
    
    2. dataset - Named benchmarks, corpora, or data collections.
       Examples: ImageNet, SQuAD, GLUE, CoNLL2003, Wikipedia, training data, dev set
       Include anything with: dataset, corpus, data, examples, set, benchmark
    
    3. metric - Specific performance measures.
       Examples: F1, BLEU score, accuracy, mAP, RMSE, exact match
       Must be concrete measurement names, not vague terms
    
    4. task - Well-defined research problems or computational goals.
       Examples: image classification, question answering, named entity recognition, machine translation
       Do NOT extract single words without context: avoid "text", "classification", "predictions"
    
    5. object - Domain-specific entities under investigation.
       Examples: amino acid sequences, answer span, target proteins
       Do NOT extract generic terms: features, parameters, representations
    
    6. other - Concepts, techniques, or architectural components.
       Examples: attention mechanisms, residual connections, transfer learning, ablation studies
    
    7. generic - References to previously mentioned entities.
       Examples: "The model", "Our approach"
    
    ## CRITICAL RULES:
    
    ### DO NOT EXTRACT:
    - Single generic words: text, data, model, information, predictions
    - Vague activities: classification, training (without context)
    - Citations: Peters et al., any phrase with "et al."
    - Meta-discourse: prior work, related work
    - Descriptive phrases: pre-trained model, unlabeled text (these are too vague)
    
    ### DATASET CLASSIFICATION:
    If text contains "data", "dataset", "examples", "corpus", "set", label as dataset.
    Examples: training data, pre-training data, dev set, labeled examples
    
    ### TASK CLASSIFICATION:
    Must be a complete problem statement, not a single word.
    Good: question answering, named entity recognition, sentiment analysis
    Bad: text, classification, predictions, sentence A, sentence B
    
    ### EXTRACTION RULES:
    - Extract ALL occurrences
    - Use exact text without punctuation
    - For "Full Name (ACRONYM)", extract only full name
    - Prioritize types by order listed above
    
    ### OUTPUT FORMAT:
    Return entities in order of appearance with type and exact text.
    """
)


EXAMPLES = [
    lx.data.ExampleData(
        text="Image classification is challenging. We propose ResNet using residual connections. We evaluate on ImageNet achieving 93% top-5 accuracy.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Image classification"),
            lx.data.Extraction(extraction_class="method", extraction_text="ResNet"),
            lx.data.Extraction(extraction_class="other", extraction_text="residual connections"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="ImageNet"),
            lx.data.Extraction(extraction_class="metric", extraction_text="top-5 accuracy"),
        ],
    ),
    lx.data.ExampleData(
        text="Named entity recognition extracts entities. BiLSTM-CRF uses contextualized embeddings. It achieves 91% F1 on CoNLL2003.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Named entity recognition"),
            lx.data.Extraction(extraction_class="method", extraction_text="BiLSTM-CRF"),
            lx.data.Extraction(extraction_class="other", extraction_text="contextualized embeddings"),
            lx.data.Extraction(extraction_class="metric", extraction_text="F1"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="CoNLL2003"),
        ],
    ),
    lx.data.ExampleData(
        text="Question answering systems answer queries. BERT-QA uses transformer encoders for passage ranking on SQuAD achieving 89% exact match.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Question answering"),
            lx.data.Extraction(extraction_class="method", extraction_text="BERT-QA"),
            lx.data.Extraction(extraction_class="other", extraction_text="transformer encoders"),
            lx.data.Extraction(extraction_class="task", extraction_text="passage ranking"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="SQuAD"),
            lx.data.Extraction(extraction_class="metric", extraction_text="exact match"),
        ],
    ),
    lx.data.ExampleData(
        text="We train on labeled training examples from GLUE. The dev set contains 10000 examples. We use dropout probability of 0.1.",
        extractions=[
            lx.data.Extraction(extraction_class="dataset", extraction_text="labeled training examples"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="GLUE"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="dev set"),
        ],
    ),
    lx.data.ExampleData(
        text="Machine translation converts text. We apply Transformer architecture with self-attention. We train on WMT 2014 data obtaining 28 BLEU score.",
        extractions=[
            lx.data.Extraction(extraction_class="task", extraction_text="Machine translation"),
            lx.data.Extraction(extraction_class="other", extraction_text="Transformer architecture"),
            lx.data.Extraction(extraction_class="other", extraction_text="self-attention"),
            lx.data.Extraction(extraction_class="dataset", extraction_text="WMT 2014 data"),
            lx.data.Extraction(extraction_class="metric", extraction_text="BLEU score"),
        ],
    ),
    lx.data.ExampleData(
        text="We conduct ablation studies on masking strategies. Peters et al. proposed ELMo. NER systems extract entities from unlabeled text.",
        extractions=[
            lx.data.Extraction(extraction_class="other", extraction_text="ablation studies"),
            lx.data.Extraction(extraction_class="other", extraction_text="masking strategies"),
            lx.data.Extraction(extraction_class="method", extraction_text="ELMo"),
            lx.data.Extraction(extraction_class="task", extraction_text="NER"),
        ],
    ),
    lx.data.ExampleData(
        text="Sentence A is input to the model. Predictions are made on sentence B. The pre-trained model uses pre-training data.",
        extractions=[
            lx.data.Extraction(extraction_class="dataset", extraction_text="pre-training data"),
        ],
    ),
]
