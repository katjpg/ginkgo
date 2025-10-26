"""Prompts for relation extraction."""

PROMPT = """Extract the semantic relationship between these two entities from scientific text.

Relation types:
- used_for: method solves/performs task (e.g., "BERT for named entity recognition")
- evaluated_on: testing performance on dataset/metric (e.g., "ResNet evaluated on ImageNet")
- compared_with: methods compared for performance (e.g., "BERT outperforms LSTM")
- improves_upon: method improves over another (e.g., "RoBERTa improves upon BERT")
- based_on: method derived from another (e.g., "GPT-2 based on transformer architecture")
- applied_to: method processes object/domain (e.g., "attention mechanism applied to proteins")

Relation distinctions:
- used_for: METHOD solves/performs TASK
- applied_to: METHOD processes OBJECT/domain data
- evaluated_on: performance testing (requires DATASET or METRIC as tail)

If no clear relation exists, return "NONE".
"""

EXAMPLES = [
    {
        "head": {"text": "BERT", "type": "METHOD"},
        "tail": {"text": "named entity recognition", "type": "TASK"},
        "syntax": "via 'fine-tune'; using 'for'",
        "sentence": "We fine-tune BERT for named entity recognition on biomedical text.",
        "relation": "used_for",
    },
    {
        "head": {"text": "ResNet", "type": "METHOD"},
        "tail": {"text": "ImageNet", "type": "DATASET"},
        "syntax": "via 'evaluate'; using 'on'",
        "sentence": "We evaluate ResNet on the ImageNet dataset to measure classification accuracy.",
        "relation": "evaluated_on",
    },
    {
        "head": {"text": "BERT", "type": "METHOD"},
        "tail": {"text": "GLUE", "type": "DATASET"},
        "syntax": "via 'evaluated'; using 'on'",
        "sentence": "We evaluated BERT on the GLUE benchmark dataset.",
        "relation": "evaluated_on",
    },
    {
        "head": {"text": "BERT", "type": "METHOD"},
        "tail": {"text": "F1 score", "type": "METRIC"},
        "syntax": "via 'achieves'",
        "sentence": "BERT achieves 92.4% F1 score on the test set.",
        "relation": "evaluated_on",
    },
    {
        "head": {"text": "transformer", "type": "METHOD"},
        "tail": {"text": "LSTM", "type": "METHOD"},
        "syntax": "via 'outperforms'",
        "sentence": "The transformer model significantly outperforms LSTM on long-range dependencies.",
        "relation": "compared_with",
    },
    {
        "head": {"text": "RoBERTa", "type": "METHOD"},
        "tail": {"text": "BERT", "type": "METHOD"},
        "syntax": "via 'improves'; using 'upon'",
        "sentence": "RoBERTa improves upon BERT by training on more data with dynamic masking.",
        "relation": "improves_upon",
    },
    {
        "head": {"text": "GPT-2", "type": "METHOD"},
        "tail": {"text": "transformer", "type": "OTHER"},
        "syntax": "via 'based'; using 'on'",
        "sentence": "GPT-2 is based on the transformer architecture with modifications for generation.",
        "relation": "based_on",
    },
    {
        "head": {"text": "attention mechanism", "type": "METHOD"},
        "tail": {"text": "protein sequences", "type": "OBJECT"},
        "syntax": "via 'applied'; using 'to'",
        "sentence": "We applied attention mechanisms to protein sequences for structure prediction.",
        "relation": "applied_to",
    },
    {
        "head": {"text": "sentiment analysis", "type": "TASK"},
        "tail": {"text": "IMDb reviews", "type": "DATASET"},
        "syntax": "via 'performed'; using 'on'",
        "sentence": "Sentiment analysis was performed on the IMDb reviews dataset.",
        "relation": "evaluated_on",
    },
    {
        "head": {"text": "ImageNet", "type": "DATASET"},
        "tail": {"text": "accuracy", "type": "METRIC"},
        "syntax": "direct connection",
        "sentence": "ImageNet accuracy remains the standard benchmark metric.",
        "relation": "evaluated_on",
    },
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": [
                "used_for",
                "evaluated_on",
                "compared_with",
                "improves_upon",
                "based_on",
                "applied_to",
                "NONE",
            ],
        },
        "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of classification decision",
        },
    },
    "required": ["relation", "confidence", "reasoning"],
}


def build_prompt(pair: dict, examples: list[dict] | None = None) -> str:
    """Build LLM prompt for relation classification."""
    if examples is None:
        examples = EXAMPLES[:6]

    head = pair["head"]
    tail = pair["tail"]
    syntax = pair.get("syntax", "no syntactic pattern")
    sentence = pair.get("sentence", "")

    example_text = "\n\n".join(
        [
            f"Example {i+1}:\n"
            f"Head: {ex['head']['text']} ({ex['head']['type']})\n"
            f"Tail: {ex['tail']['text']} ({ex['tail']['type']})\n"
            f"Context: \"{ex['sentence']}\"\n"
            f"Syntax: {ex.get('syntax', 'no pattern')}\n"
            f"→ Relation: {ex['relation']}"
            for i, ex in enumerate(examples)
        ]
    )

    return f"""{PROMPT}

{example_text}

Query:
Head: {head['text']} ({head['type']})
Tail: {tail['text']} ({tail['type']})
Context: "{sentence}"
Syntax: {syntax}

Analyze the relationship and return your classification."""
