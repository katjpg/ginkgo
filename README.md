# ginkgo

> a research tool for scientific information extraction and knowledge graph construction.

## Overview

ginkgo extracts research concepts, identifies semantic relationships, and transforms academic papers into structured knowledge representations. Built for researchers to accelerate scientific exploration and discovery.

ginkgo features a comprehensive document-level processing pipeline that consists of the following core components:
- **Structural**: Parses PDFs to structured TEI XML using GROBID, extracting sections, citations, and document organization from academic papers.
- **Syntactic**: Extracts shortest dependency paths and contextual features between entities via spaCy to support relation classification.
- **Semantic**: Extracts entities and classifies semantic relationships using few-shot LLM prompting augmented with syntactic features.