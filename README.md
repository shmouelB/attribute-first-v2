# Attribute First 2.0

This private review repository contains the implementation and configuration
files for **Attribute First 2.0**, a modular pipeline for attributable
long-form generation on multi-document summarization (MDS) and long-form
question answering (LFQA).

The pipeline keeps evidence selection before generation and exposes the
following stages explicitly:

1. content selection;
2. optional ambiguity highlighting / context augmentation;
3. evidence clustering;
4. cluster reordering;
5. planned, sentence-level attributed fusion.

Structured outputs and role-separated prompts are fixed reliability controls.
The repository also contains independent-call and sequential-dialogue
implementations used by the ablation design.

## Review scope

This repository is intentionally limited to implementation material:

- pipeline and stage code;
- prompts and experiment configurations;
- architecture documentation;
- focused protocol and parser tests.

Generated outputs, numerical results, evaluation scripts, evaluator artifacts,
datasets, the paper source, and local provenance snapshots are intentionally
excluded from this review repository.

## Layout

- `few_shot_experiments/attribute_first/`: core domain, application, stage,
  runtime, prompting, and persistence modules;
- `few_shot_experiments/configs/`: pipeline and stage configurations;
- `few_shot_experiments/prompts/`: prompt demonstrations and templates;
- `few_shot_experiments/run_*.py`: command-line pipeline entry points;
- `few_shot_experiments/tests/`: focused implementation tests;
- `few_shot_experiments/ARCHITECTURE.md`: architecture and module boundaries;
- `few_shot_experiments/GLOSSARY.md`: terminology and historical aliases.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env` before running model-backed pipelines.
Dataset files are not distributed in this review repository.

## Entry points

- `run_full_pipeline.py`: standard Attribute-First pipeline;
- `run_coherence_structured.py`: clustering, reordering, and planned fusion;
- `run_dialogue_sequential.py`: sequential-dialogue ablation;
- `run_iterative_sentence_generation.py`: sentence-wise attributed
  generation.

Run the scripts from the `few_shot_experiments/` directory so legacy relative
configuration paths resolve consistently.

## Status

This is a private first-review snapshot. It is not a public benchmark release
and deliberately makes no numerical performance claim. Because datasets,
evaluation code, and provenance snapshots are excluded, it is an
implementation-review snapshot rather than a standalone reproduction bundle.
