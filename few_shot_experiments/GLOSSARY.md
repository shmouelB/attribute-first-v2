# Experiment and code glossary

This glossary separates scientific factors from historical repository names.
Historical aliases remain supported for old artifacts, but new code uses the
canonical vocabulary.

## Models

| Name | Exact meaning | Status |
| --- | --- | --- |
| `g3flash` | Local shorthand for the requested provider ID `models/gemini-3-flash-preview` | Preview model; never use the shorthand as a scientific factor |
| `liteweak` | Local shorthand for `models/gemini-2.5-flash-lite` | Stable provider model ID |
| `prolatest` | Local shorthand for `models/gemini-pro-latest` | Mutable `latest` alias |

The model belongs in `ModelSpec`; it does not belong in a pipeline name. A
filename such as `fic_structured_g3flash.json` is retained only when a frozen
historical hash or archive references it.

## Datasets and stages

| Name | Meaning |
| --- | --- |
| `MDS` | Multi-document summarization |
| `LFQA` | Long-form question answering |
| `CS` | Content selection: select source spans relevant to the target |
| `AH` | Historical “ambiguity highlight” stage; canonical name: context augmentation |
| `FiC` | Fusion in context: synthesize attributed text from selected evidence |
| clustering | Group selected evidence into planned sentence units |
| reordering | Choose the generation order of planned units |

## Experimental factors

| Factor | Values | Meaning |
| --- | --- | --- |
| generation | `direct`, `planned` | Direct synthesis versus cluster/order planning before synthesis |
| demonstrations | `few_shot`, `zero_shot` | Whether configurable standard stages receive demonstrations |
| context augmentation | `enabled`, `disabled` | Whether the AH/context-augmentation stage is present |
| transport | `independent`, `dialogue` | Separate requests versus one accumulated chat session |

Canonical cell IDs combine these factors, for example:

- `direct_fs_independent`
- `direct_zs_context_augmented_independent`
- `planned_fs_independent`
- `planned_zs_context_augmented_independent`

Persisted canonical identities add the dataset prefix, for example
`mds.planned_fs_independent`. Historical identities such as `MDS.coherence`
remain beside them only for archive compatibility.

## Historical pipeline names

| Historical name | Canonical factors |
| --- | --- |
| `fullcot` / `full_cot` | direct + few-shot + no context augmentation |
| `decontex` / `decontextualization` | direct + few-shot + context augmentation |
| `coherence` | planned + few-shot + no context augmentation + independent calls |
| `mega` | planned + zero-shot + context augmentation + independent calls |

`MEGA` has no expansion documented by the repository or project records. It
must therefore be treated as an opaque historical label, not presented as an
acronym. In the report, prefer the factor name **Plan-ZS+AH** or the canonical
ID `planned_zs_context_augmented_independent`.

## Dialogue and cache

In dialogue mode, application code sends only the new task for each turn. The
chat session already contains previous user/model turns, and the pinned SDK
sends the accumulated history to the provider. This is conversational
continuity, not a token-saving cache.

Only the current stage's few-shot demonstrations are temporarily present.
After a parsed response, those demonstrations are removed while the live
exchange remains. AH and FiC therefore receive a pure continuation message and
recover prior state from chat history; the internal FiC highlight registry is
not resent.

CS, optional AH, and FiC share that one session, so their configuration files
must declare exactly the same `model_name`. A mixed-model dialogue is rejected
before any output directory is claimed or any model request is attempted.

Explicit context caching is separate. In the current role-dialogue
implementation, only the stage-neutral system instruction and shared CS
demonstrations are bound to cached content; AH and FiC demonstrations remain
dynamic, just-in-time history. Provider `cached_content_token_count` is
evidence of cache use for a response; ordinary chat history alone is not.

An explicit-cache run is complete only when its trace agrees with every call
and the provider reports a positive `cached_content_token_count`. Creating or
binding a cache object alone is not evidence that a response used it.

`provider_total` is the sum of Gemini's exposed `total_token_count` values;
`provider_total_calls` is the number of responses that exposed that field.
They are not reconstructed as `prompt + completion`, because Gemini reasoning
tokens can make those quantities differ.

## Historical compatibility

- `decontex` is a misspelling accepted only at the compatibility boundary.
- `itermediate_results` is a historical artifact-directory spelling and remains
  unchanged for archive compatibility.
- Old result-directory labels identify archived runs; they are not a reliable
  substitute for the factor metadata recorded in provenance.
