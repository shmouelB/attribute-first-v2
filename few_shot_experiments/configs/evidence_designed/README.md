# Evidence-designed pipelines

These configurations are append-only experiments outside the frozen
16-cell controlled campaign.

## LFQA MEGA (2026-07-31)

Canonical identity:

`lfqa.planned_fs_context_augmented_dialogue`

Explicit output name:

`planned_few_shot_context_augmented_dialogue`

Per LFQA example, one stateful chat executes:

`content_selection (2 demos) -> ambiguity_highlight (4 demos) -> clustering
(0 demos) -> reorder (0 demos) -> planned fusion (0 demos)`

The first two stages make the experiment globally **few-shot**. The planning
stages intentionally use zero demonstrations because the historical direct
FiC demonstration is not compatible with the ordered sentence-plan schema.
After Content Selection, application code sends only the new stage task and
the minimum canonical derived state (the highlight registry once for
clustering and the ordered plan once for fusion); it does not resend the
source documents. Explicit context caching is disabled because the provider
chat history is the state carrier.

This is the requested LFQA MEGA hypothesis. It does not replace or rename the
historical `mega` (`planned + zero-shot + context augmentation + independent
calls`) and is not part of the frozen controlled campaign.

Run contract:

```bash
python run_full_pipeline.py \
  --config-file configs/evidence_designed/test/LFQA/pipelines/planned_few_shot_context_augmented_dialogue.json \
  --dialogue-mode \
  --planned-dialogue \
  --concurrency 1 \
  --generation-strategy planned \
  --demonstration-mode few_shot \
  --context-augmentation enabled \
  --transport-mode dialogue \
  --canonical-cell-id lfqa.planned_fs_context_augmented_dialogue \
  -o <new-output-directory>
```
