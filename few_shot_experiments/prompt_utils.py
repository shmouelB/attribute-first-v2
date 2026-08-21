import hashlib
import json
import logging
import re
from typing import List, Dict

from attribute_first.stages.structured_highlights import (
    HighlightMarkupParser,
    SourceSpanNotFound,
    UniqueSourceSpanExtender,
    UniqueSourceSpanLocator,
)
from utils import (
    SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS,
    env_flag,
    extract_highlights,
    get_highlighted_doc,
    get_highlighted_doc_two_sets,
    make_demo,
    protocol_environment,
    rmv_txt_after_last_highlight,
)


_STRUCTURED_FIC_INSTRUCTION = """
[STRUCTURED OUTPUT CONTRACT]
Return only one valid JSON object with this exact shape:
{"sentences": [{"sentence_id": 1, "sentence_text": "<sentence text>", "highlight_ids": [1, 2]}]}
The "sentences" value must be an array. Each item must contain an integer
"sentence_id", a string "sentence_text", and an array of integer
"highlight_ids". Use one item per output sentence, number sentence_id values
consecutively from 1, and assign every used highlight index to the sentence
that covers it. Use as few sentences as the highlights warrant; do not force
a fixed word count or minimum sentence count. Do not emit clustering lines, a
final-answer sentinel, Markdown, or any text outside the JSON object.
""".strip()

_CONTENT_SELECTION_UNIQUE_SOURCE_SPAN_INSTRUCTION = """
[UNIQUE SOURCE ATTRIBUTION]
Every "span_text" must identify exactly one unique occurrence in its declared
document. If the selected words occur more than once, extend the consecutive
verbatim span with adjacent source words until only one occurrence remains.
Never choose or imply an arbitrary first occurrence.
""".strip()

_AMBIGUITY_UNIQUE_SOURCE_SPAN_INSTRUCTION = """
[UNIQUE SOURCE ATTRIBUTION]
Every "span_text" must identify exactly one unique occurrence in its declared
document. If the selected context occurs more than once, extend it only with
adjacent non-highlighted source words until one unique occurrence remains.
Never include or cross any already-highlighted evidence, and never choose or
imply an arbitrary first occurrence.
""".strip()

DEMO_SELECTION_ALGORITHM = "sha256-rank-v1"


class StructuredContentSelectionOverrides:
    """Validated per-demo repairs for ambiguous gold highlight text."""

    def __init__(self, payload):
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError(
                "structured content-selection overrides must be an object"
            )
        self._values: dict[tuple[int, int], str] = {}
        self._used: set[tuple[int, int]] = set()
        for doc_id, spans in payload.items():
            if not str(doc_id).isdigit() or not isinstance(spans, dict):
                raise ValueError(
                    "structured content-selection override documents and "
                    "span indexes must be numeric objects"
                )
            for span_index, span_text in spans.items():
                if (
                    not str(span_index).isdigit()
                    or not isinstance(span_text, str)
                    or not span_text.strip()
                ):
                    raise ValueError(
                        "structured content-selection override spans must "
                        "be non-empty strings at numeric indexes"
                    )
                self._values[(int(doc_id), int(span_index))] = span_text

    def resolve(self, doc_id: int, span_index: int) -> str | None:
        key = (doc_id, span_index)
        value = self._values.get(key)
        if value is not None:
            self._used.add(key)
        return value

    def require_all_used(self) -> None:
        unused = sorted(set(self._values) - self._used)
        if unused:
            raise ValueError(
                "structured content-selection overrides reference missing "
                f"demo spans: {unused}"
            )


def _without_legacy_fic_output_format(instruction):
    """Keep FiC semantics while removing incompatible legacy constraints."""
    marker = "IMPORTANT: The clustering steps must be"
    if marker in instruction:
        instruction = instruction.split(marker, 1)[0].rstrip()

    # Structured FiC requires one non-empty, uniquely attributed highlight-ID
    # list per sentence.  A fixed minimum of 2/7 sentences is therefore
    # impossible whenever the live registry contains fewer unique highlights.
    # The historical free-text prompts keep their original length guidance;
    # only the structured contract removes it.
    incompatible_constraints = (
        r"\s*IMPORTANT:\s*make sure that the (?:final )?summary consists "
        r"of approximately 200 words and at least 7 sentences\.?",
        r",\s*and that the output summary has at least two sentences\.?",
        r"\s*Also,\s*make sure there are at least two sentences in the "
        r"output summary\.?",
    )
    for constraint in incompatible_constraints:
        instruction = re.sub(
            constraint,
            "",
            instruction,
            flags=re.IGNORECASE,
        )
    return instruction.strip()


def _roles_strip_instruction(text, instruction):
    """A rendered demo/target begins with the instruction ({INST} is the template head).
    The instruction goes to the system role, so the user turn is everything after it."""
    if instruction and text.startswith(instruction):
        return text[len(instruction):].strip()
    return text.strip()


def _roles_split_answer(full_demo, input_only):
    """`full_demo` = make_demo(test=False) = instruction + input + answer.
    `input_only` = make_demo(test=True) = (instruction + input) stripped.
    Since {A} is always the template tail, the answer is the suffix. Returns None if the
    prefix does not line up (caller then falls back to a single flat user turn)."""
    if full_demo.startswith(input_only):
        answer = full_demo[len(input_only):].strip()
        return answer or None
    return None


def _legacy_document_spans_to_structured(answer):
    """Convert the prompt files' gold ``Document [n]: ...`` answer to JSON."""
    highlights = []
    pattern = re.compile(
        r"^Document\s*\[(\d+)\]\s*:\s*(.*?)"
        r"(?=^Document\s*\[\d+\]\s*:|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    matches = pattern.findall(answer or "")
    if (answer or "").strip() and not matches:
        raise ValueError("structured demo has an invalid Document [n] gold answer")
    for doc_id, spans_blob in matches:
        for span_text in spans_blob.split("<SPAN_DELIM>"):
            span_text = span_text.strip()
            if span_text:
                highlights.append(
                    {
                        "doc_id": f"Document [{doc_id}]",
                        "span_text": span_text,
                    }
                )
    return {"highlights": highlights}


def _structured_role_demo_answer(item, subtask):
    """Build a schema-shaped model turn directly from one gold demo object."""
    if subtask == "content_selection":
        highlights = []
        markup_parser = HighlightMarkupParser()
        extender = UniqueSourceSpanExtender()
        locator = UniqueSourceSpanLocator()
        overrides = StructuredContentSelectionOverrides(
            item.get("structured_content_selection_span_overrides")
        )
        for doc_id, document in enumerate(item["docs"], start=1):
            marked_source = markup_parser.parse(document["text"])
            gold_spans = extract_highlights(
                document["text"], "{HS}", "{HE}"
            )
            if len(gold_spans) != len(
                marked_source.highlighted_intervals
            ):
                raise ValueError(
                    "structured CS demo highlight markers do not match "
                    f"the extracted spans in Document [{doc_id}]"
                )
            for span_index, (span_text, target) in enumerate(
                zip(
                    gold_spans,
                    marked_source.highlighted_intervals,
                    strict=True,
                )
            ):
                if (
                    marked_source.text[target.start : target.end]
                    != span_text
                ):
                    raise ValueError(
                        "structured CS demo marker offsets do not preserve "
                        f"the gold text in Document [{doc_id}]"
                    )
                forbidden = tuple(
                    interval
                    for index, interval in enumerate(
                        marked_source.highlighted_intervals
                    )
                    if index != span_index
                )
                override_text = overrides.resolve(doc_id, span_index)
                if override_text is None:
                    unique_span = extender.extend(
                        f"Document [{doc_id}]",
                        marked_source.text,
                        target,
                        forbidden_intervals=forbidden,
                    )
                else:
                    unique_span = locator.locate(
                        f"Document [{doc_id}]",
                        marked_source.text,
                        override_text,
                    )
                    if not (
                        unique_span.start <= target.start
                        and target.end <= unique_span.end
                    ):
                        raise ValueError(
                            "structured content-selection override does not "
                            "contain its intended gold highlight in "
                            f"Document [{doc_id}]"
                        )
                    if any(
                        unique_span.start < interval.end
                        and interval.start < unique_span.end
                        for interval in forbidden
                    ):
                        raise ValueError(
                            "structured content-selection override crosses "
                            "another gold highlight in "
                            f"Document [{doc_id}]"
                        )
                highlights.append(
                    {
                        "doc_id": str(doc_id),
                        "span_text": marked_source.text[
                            unique_span.start : unique_span.end
                        ],
                    }
                )
        overrides.require_all_used()
        return {"highlights": highlights}
    if subtask == "ambiguity_highlight":
        structured = _legacy_document_spans_to_structured(
            item.get("ambiguity_highlight_answer", "")
        )
        markup_parser = HighlightMarkupParser()
        locator = UniqueSourceSpanLocator()
        for highlight in structured["highlights"]:
            document_match = re.fullmatch(
                r"Document\s*\[(\d+)\]",
                highlight["doc_id"],
            )
            if document_match is None:
                raise ValueError("structured AH demo has an invalid doc_id")
            document_index = int(document_match.group(1)) - 1
            if (
                document_index < 0
                or document_index >= len(item["docs"])
            ):
                raise ValueError(
                    "structured AH demo references an unavailable "
                    f"{highlight['doc_id']}"
                )
            source = markup_parser.parse(
                item["docs"][document_index]["text"]
            )
            try:
                located = locator.locate(
                    highlight["doc_id"],
                    source.text,
                    highlight["span_text"],
                )
            except SourceSpanNotFound as exc:
                raise ValueError(
                    "structured AH demo span is not a verbatim substring "
                    f"of {highlight['doc_id']}: "
                    f"{highlight['span_text']!r}"
                ) from exc
            if source.overlaps_highlight(located):
                raise ValueError(
                    "structured AH demo span overlaps highlighted evidence "
                    f"in {highlight['doc_id']}: "
                    f"{highlight['span_text']!r}"
                )
        return structured
    if subtask == "FiC":
        per_document_counts = [
            len(
                extract_highlights(
                    document["text"], "{HS}", "{HE}"
                )
            )
            for document in item["docs"]
        ]
        offsets = [0]
        for count in per_document_counts:
            offsets.append(offsets[-1] + count)
        seen_highlights = set()
        sentences = []
        for sentence_id, plan in enumerate(item["planning"], start=1):
            candidate_ids = sorted(
                {
                    offsets[cluster["doc"] - 1] + relative_id
                    for cluster in plan["highlights_cluster"]
                    for relative_id in cluster["relative_highlights"]
                    if relative_id
                    <= per_document_counts[cluster["doc"] - 1]
                }
            )
            highlight_ids = [
                highlight_id
                for highlight_id in candidate_ids
                if highlight_id not in seen_highlights
            ]
            if not highlight_ids:
                raise ValueError(
                    "structured FiC demo sentence has no unique evidence"
                )
            seen_highlights.update(highlight_ids)
            sentences.append(
                {
                    "sentence_id": sentence_id,
                    "sentence_text": plan["output"],
                    "highlight_ids": highlight_ids,
                }
            )
        return {"sentences": sentences}
    return None


def get_indir_paths(args):
    if type(args)!=dict: # i.e. argparse
        args = args.__dict__
    indir_alignments = args['indir_alignments'] if args['indir_alignments'] else f"../data/{args['setting']}/{args['split']}.json"
    indir_prompt = args['indir_prompt'] if args['indir_prompt'] else f"prompts/{args['setting']}.json"
    return indir_prompt, indir_alignments

def get_data(args):
    indir_prompt, indir_alignments = get_indir_paths((args))

    alignments_dict = []
    with open(indir_alignments, 'r') as f1:
        alignments_dict = [json.loads(line) for line in f1.readlines()]

    with open(indir_prompt, 'r') as f1:
        prompt_dict = json.loads(f1.read())

    max_examples = getattr(args, "max_examples", None)
    if max_examples is not None:
        alignments_dict = alignments_dict[:max_examples]

        if isinstance(prompt_dict, list):
            prompt_dict = prompt_dict[:max_examples]

        elif isinstance(prompt_dict, dict) and len(alignments_dict) > 0 and "id" in alignments_dict[0]:
            keep_ids = {a["id"] for a in alignments_dict}
            prompt_dict = {k: v for k, v in prompt_dict.items() if k in keep_ids}

    return prompt_dict, alignments_dict

def get_subtask_prompt_structures(prompt_dict : Dict, setting: str, subtask : str, CoT : bool, always_with_question : bool, structured_output: bool = False) -> Dict:
    """returns the subtask relevant prompt structures (instruction and answer-related and demo_prompt)"""

    demo_prompt = prompt_dict["demo_prompt_content_selection"] if setting=="LFQA" and (subtask in SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS or always_with_question) else prompt_dict["demo_prompt"]

    # Proposal #1 (AF_DOCS_FIRST): put the documents (and question) BEFORE the instruction so the
    # docs block is a shared leading prefix across subtask calls -> Gemini implicit prefix-cache can
    # reuse it (the instruction, which differs per subtask, no longer breaks the prefix at token 0).
    # make_demo substitutes {INST}/{D}/{Q}/{A} independently, so reordering the template only changes
    # output order, not content. This is the non-roles flat-prompt caching experiment.
    if env_flag("AF_DOCS_FIRST"):
        if "{Q}" in demo_prompt:
            demo_prompt = "Question: {Q}\n\n{D}\n\n{INST}\n{A}"
        else:
            demo_prompt = "{D}\n\n{INST}\n{A}"

    with_question_suffix = "-with-question" if always_with_question and setting=="LFQA" else ""

    if subtask == "FiC":
        CoT_suffix = "-CoT" if CoT else ""
        answer_related_prompts = {"answer_prompt":prompt_dict[f"answer_FiC{CoT_suffix}_prompt"],
                                  "answer_FiC_planning_prompt":prompt_dict["answer_FiC_planning_prompt"],
                                  "answer_highlights_listing_prompt":prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_highlights_fusion_prompt":prompt_dict["answer_highlights_fusion_prompt"]}
        instruction_prompt = prompt_dict[f"instruction-FiC{CoT_suffix}{with_question_suffix}"]
        if structured_output:
            instruction_prompt = (
                f"{_without_legacy_fic_output_format(instruction_prompt)}"
                f"\n\n{_STRUCTURED_FIC_INSTRUCTION}"
            )
    elif subtask == "content_selection":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_content_selection_prompt"],
                                  "answer_content_selection_format":prompt_dict["answer_content_selection_format"]}
        instr_key = "instruction-content-selection-structured" if structured_output and "instruction-content-selection-structured" in prompt_dict else "instruction-content-selection"
        instruction_prompt = prompt_dict[instr_key]
    elif subtask == "clustering":
        CoT_suffix = "-CoT" if CoT else ""
        answer_related_prompts = {"answer_prompt":prompt_dict[f"answer_clustering{CoT_suffix}_prompt"],
                                  "answer_highlights_listing_prompt":prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_clustering_CoT_prompt_intermediate":prompt_dict["answer_clustering-CoT_prompt_intermediate"],
                                  "answer_clustering_format":prompt_dict["answer_clustering_format"],
                                  "answer_clustering_CoT_format":prompt_dict["answer_clustering-CoT_format"]}
        instruction_prompt = prompt_dict[f"instruction-clustering{with_question_suffix}"]
    elif subtask == "e2e_only_setting":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_e2e_only_setting_prompt"]}
        instruction_prompt = prompt_dict["instruction-e2e-only-setting"]
    elif subtask == "ALCE":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_ALCE_prompt"],
                                  "answer_ALCE_format":prompt_dict["answer_ALCE_format"]}
        instruction_prompt = prompt_dict["instruction-ALCE"]
    elif subtask == "ambiguity_highlight" :
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_ambiguity_highlight_prompt"],
                                  "answer_ambiguity_highlight_format":prompt_dict["answer_ambiguity_highlight_format"]}
        instr_key = "instruction-ambiguity-highlight-structured" if structured_output and "instruction-ambiguity-highlight-structured" in prompt_dict else "instruction-ambiguity-highlight"
        instruction_prompt = prompt_dict[instr_key]
    elif subtask in ("topic_outline_fusion", "topic_cluster_fusion"):
        answer_related_prompts = {"answer_prompt": prompt_dict["answer_FiC-CoT_prompt"],
                                  "answer_FiC_planning_prompt": prompt_dict["answer_FiC_planning_prompt"],
                                  "answer_highlights_listing_prompt": prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_highlights_fusion_prompt": prompt_dict["answer_highlights_fusion_prompt"]}
        instr_key = "topic-outline" if subtask == "topic_outline_fusion" else "topic-cluster"
        instruction_prompt = prompt_dict[f"instruction-{instr_key}-CoT{with_question_suffix}"]
    elif subtask == "FiC_v2":
        answer_related_prompts = {"answer_prompt": prompt_dict["answer_FiC-CoT_prompt"],
                                  "answer_FiC_planning_prompt": prompt_dict["answer_FiC_planning_prompt"],
                                  "answer_highlights_listing_prompt": prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_highlights_fusion_prompt": prompt_dict["answer_highlights_fusion_prompt"]}
        if always_with_question and setting == "LFQA":
            instruction_prompt = prompt_dict["instruction-FiC-CoT-with-question-v2"]
        else:
            instruction_prompt = prompt_dict["instruction-FiC-CoT"]
    else:
        raise Exception(f"{subtask} is not yet supported")

    if (
        structured_output
        and subtask in {"content_selection", "ambiguity_highlight"}
    ):
        attribution_instruction = (
            _CONTENT_SELECTION_UNIQUE_SOURCE_SPAN_INSTRUCTION
            if subtask == "content_selection"
            else _AMBIGUITY_UNIQUE_SOURCE_SPAN_INSTRUCTION
        )
        instruction_prompt = (
            f"{instruction_prompt}\n\n{attribution_instruction}"
        )

    return {"answer_related_prompts" : answer_related_prompts,
            "instruction_prompt" : instruction_prompt,
            "demo_prompt" : demo_prompt,
            "structured_output": structured_output,
            "subtask": subtask}

def construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus: bool = False, prct_surplus: float = 0.25):
        """prct_surplus: the percentage of last sentences to consider as surplus and remove in subtasks without given highlights (e.g., content_selection or end-to-end or ALCE)"""
        highlight_start_tkn = "{HS}"
        highlight_end_tkn="{HE}"
        context_start_tkn = "{CS}"
        context_end_tkn = "{CE}"
        topic_name = instance['unique_id']
        docs_map = {elem['documentFile']:elem['rawDocumentText'] for elem in instance['documents'] if elem['documentFile']}
        doc_sents_map = {elem['documentFile']:elem['documentText'] for elem in instance['documents'] if elem['documentFile']}

        # Arm A (AF_MARK_CONTEXT): render AH context spans with {CS}/{CE} so they appear inline
        # for disambiguation but are NOT listed in {HLIST} (which extracts only {HS} spans),
        # so FiC won't try to "cover" them as answer content.
        mark_context = env_flag("AF_MARK_CONTEXT") and instance.get("context_set_of_highlights_in_context")
        if mark_context:
            context_h = instance["context_set_of_highlights_in_context"]
            ctx_keys = {(c['documentFile'], tuple(map(tuple, c['docSpanOffsets']))) for c in context_h}
            evidence_h = [h for h in instance['set_of_highlights_in_context']
                          if (h['documentFile'], tuple(map(tuple, h['docSpanOffsets']))) not in ctx_keys]
            highlighted_texts = get_highlighted_doc_two_sets(
                docs=docs_map, evidence_h=evidence_h, context_h=context_h,
                hs_tkn=highlight_start_tkn, he_tkn=highlight_end_tkn,
                cs_tkn=context_start_tkn, ce_tkn=context_end_tkn,
                merge_cross_sents_highlights=merge_cross_sents_highlights, doc_sents=doc_sents_map)
        else:
            highlighted_texts = get_highlighted_doc(docs=docs_map,
                                                    highlights=instance['set_of_highlights_in_context'],
                                                    highlight_start_tkn = highlight_start_tkn,
                                                    highlight_end_tkn=highlight_end_tkn,
                                                    merge_cross_sents_highlights=merge_cross_sents_highlights,
                                                    doc_sents=doc_sents_map)

        if cut_surplus:
            if no_highlights:
                highlighted_texts = {elem['documentFile']:"".join(elem['documentText'][:max(int(len(elem['documentText'])*(1-prct_surplus)), 5)]) for elem in instance['documents'] if elem['documentFile']}
            else:
                cut_tkn = context_end_tkn if mark_context else highlight_end_tkn
                # cut after the last span-end token of EITHER kind so trailing context is kept
                def _cut_both(doc_text):
                    last = max(doc_text.rfind(highlight_end_tkn), doc_text.rfind(context_end_tkn))
                    if last == -1:
                        return ""
                    end_tkn = highlight_end_tkn if doc_text.rfind(highlight_end_tkn) == last else context_end_tkn
                    return doc_text[:last] + end_tkn
                if mark_context:
                    highlighted_texts = {doc_name:_cut_both(doc_text) for doc_name,doc_text in highlighted_texts.items()}
                else:
                    highlighted_texts = {doc_name:rmv_txt_after_last_highlight(doc_text, highlight_end_tkn) for doc_name,doc_text in highlighted_texts.items()}
                highlighted_texts = {key:value for key,value in highlighted_texts.items() if value}

        docs_order = [{"doc_name":doc_name, "doc_text":doc_text} for doc_name,doc_text in highlighted_texts.items()]
        eval_item = {"docs":[{'text':dct["doc_text"]} for dct in docs_order]}

        _q = instance.get("query") or instance.get("question") or ""
        if _q:
            eval_item["question"] = _q

        curr_prompt, curr_highlight_list = make_demo(
            item=eval_item, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"],
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights,
            test=True
        )

        return curr_prompt, curr_highlight_list, topic_name, docs_order

def select_demo_ids(available, requested, *, seed=None, debugging=False):
    """Select unique demonstration IDs with an optional reproducible seed."""
    if (
        type(available) is not int
        or type(requested) is not int
        or available < 0
        or requested < 0
        or requested > available
    ):
        raise ValueError(
            f"cannot select {requested!r} demonstrations from "
            f"{available!r} available"
        )
    if requested == 0:
        return []
    if debugging:
        return list(range(requested))
    if type(seed) is not int:
        raise ValueError(
            "a deterministic integer seed is required for demonstration "
            "selection"
        )
    ranked_indices = sorted(
        range(available),
        key=lambda index: (
            hashlib.sha256(
                (
                    f"{DEMO_SELECTION_ALGORITHM}:"
                    f"{seed}:{index}"
                ).encode("utf-8")
            ).digest(),
            index,
        ),
    )
    return ranked_indices[:requested]


def construct_prompts(prompt_dict : Dict, alignments_dict : List[Dict], n_demos : int, debugging : bool, merge_cross_sents_highlights : bool, specific_prompt_details : Dict, tkn_counter: Dict, no_highlights : bool = False, cut_surplus : bool = False, prct_surplus: float = None, seed: int = None):
    DEMO_HEADER = "### DEMO EXAMPLES (DO NOT ANSWER) ###\n"
    TARGET_HEADER = "### TARGET DOCUMENTS (ANSWER ONLY THESE) ###\n"

    train_ids = select_demo_ids(
        len(prompt_dict["demos"]),
        n_demos,
        seed=seed,
        debugging=debugging,
    )
    head_prompt = DEMO_HEADER if n_demos and n_demos > 0 else ""
    head_prompt_shorter = DEMO_HEADER if n_demos and n_demos > 0 else ""
    used_demos = []

    # Roles mode (AF_USE_ROLES): render each demo as a user turn (docs/question, instruction
    # stripped -> system) + a model turn (the gold answer). Built for both the full and the
    # token-budget-shortened demo sets, mirroring head_prompt / head_prompt_shorter.
    use_roles = env_flag("AF_USE_ROLES")
    instruction = specific_prompt_details["instruction_prompt"]
    # make_demo substitutes {HS}/{HE} inside the instruction too (removed for content_selection,
    # replaced with the highlight tokens otherwise). The system role and the strip prefix must use
    # this RENDERED instruction so it matches the head of every rendered demo/target.
    if no_highlights:
        instruction_rendered = instruction.replace("{HS}", "").replace("{HE}", "")
    else:
        instruction_rendered = (instruction.replace("{HS}", prompt_dict["highlight_start_tkn"])
                                           .replace("{HE}", prompt_dict["highlight_end_tkn"]))
    demo_turns_full, demo_turns_shorter = [], []

    def _demo_turns(item):
        full_demo, _ = make_demo(
            item=item, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"],
            instruction=instruction, answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights)
        input_only, _ = make_demo(
            item=item, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"],
            instruction=instruction, answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights, test=True)
        answer = _roles_split_answer(full_demo, input_only)
        if answer is None:
            return None
        if specific_prompt_details.get("structured_output"):
            structured_answer = _structured_role_demo_answer(
                item, specific_prompt_details.get("subtask")
            )
            if structured_answer is not None:
                answer = json.dumps(structured_answer, ensure_ascii=False)
        return [{"role": "user", "parts": [_roles_strip_instruction(input_only, instruction_rendered)]},
                {"role": "model", "parts": [answer]}]

    for train_id in train_ids:
        train_item = prompt_dict["demos"][train_id]
        used_demos.append(train_item)

        curr_prompt_demo, _ = make_demo(
            item=train_item, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"],
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights
        )
        head_prompt += curr_prompt_demo
        head_prompt += prompt_dict["demo_sep"]

        train_item_shorter = {key:[{doc_key:elem['shorter_text'] if doc_key=="text" else doc_value
                                    for doc_key,doc_value in elem.items()} for elem in value] if key=="docs" else value for key,value in train_item.items()}

        curr_prompt_demo_shorter, _ = make_demo(
            item=train_item_shorter, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"],
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights
        )
        head_prompt_shorter += curr_prompt_demo_shorter
        head_prompt_shorter += prompt_dict["demo_sep"]

        if use_roles and demo_turns_full is not None:
            t_full, t_short = _demo_turns(train_item), _demo_turns(train_item_shorter)
            if t_full is None or t_short is None:
                demo_turns_full = None  # signal fallback: disable roles entirely
            else:
                demo_turns_full.extend(t_full)
                demo_turns_shorter.extend(t_short)

    if use_roles and demo_turns_full is None:
        logging.warning("[roles] could not split a demo into input/answer; disabling roles")
        use_roles = False

    if debugging:
        alignments_dict = alignments_dict[:3]

    final_prompts, additional_data, role_messages = {}, {}, {}
    for instance in alignments_dict:
        curr_prompt, curr_highlight_list, topic_name, docs_order = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights)
        prompt_budget = tkn_counter["tkn_max_limit"]
        initial_prompt = head_prompt + TARGET_HEADER + curr_prompt
        initial_tokens = tkn_counter["tkn_counter"].token_count(
            initial_prompt
        )
        selected_surplus = None
        shortening_strategy = "none"

        if cut_surplus:
            curr_prompt, curr_highlight_list_shorter, topic_name, docs_order_shorter = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus=True, prct_surplus=prct_surplus)
            final_prompts[topic_name] = head_prompt_shorter + TARGET_HEADER + curr_prompt
            used_shorter = True
            selected_surplus = prct_surplus
            shortening_strategy = "configured_surplus"
        elif initial_tokens >= prompt_budget:
            prct_surplus_lst = [0.5, 0.6, 0.7]
            for curr_prct_surplus in prct_surplus_lst:
                curr_prompt, curr_highlight_list_shorter, topic_name, docs_order_shorter = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus=True, prct_surplus=curr_prct_surplus)
                candidate_prompt = (
                    head_prompt_shorter + TARGET_HEADER + curr_prompt
                )
                candidate_tokens = tkn_counter[
                    "tkn_counter"
                ].token_count(candidate_prompt)
                selected_surplus = curr_prct_surplus
                if candidate_tokens < prompt_budget:
                    break
            final_prompts[topic_name] = candidate_prompt
            used_shorter = True
            shortening_strategy = "adaptive_surplus"
        else:
            curr_highlight_list_shorter, docs_order_shorter = [], []
            final_prompts[topic_name] = initial_prompt
            used_shorter = False

        final_tokens = tkn_counter["tkn_counter"].token_count(
            final_prompts[topic_name]
        )
        if final_tokens >= prompt_budget:
            raise ValueError(
                f"{topic_name}: constructed prompt uses {final_tokens} "
                f"tokens, which does not fit the explicit "
                f"prompt_token_budget={prompt_budget} after "
                f"{shortening_strategy}"
            )

        if use_roles:
            turns = demo_turns_shorter if used_shorter else demo_turns_full
            target_user = _roles_strip_instruction(curr_prompt, instruction_rendered)
            role_messages[topic_name] = {
                "system": instruction_rendered,
                "contents": list(turns) + [{"role": "user", "parts": [target_user]}],
            }
        highlighted_docs = [{"doc_name":elem["doc_name"],
                             "doc_text":elem["doc_text"].replace("{HS}", prompt_dict["highlight_start_tkn"]).replace("{HE}", prompt_dict["highlight_end_tkn"])} for elem in docs_order]
        non_highlighted_docs = [{"doc_name":elem["doc_name"],
                                 "doc_text":elem["doc_text"].replace("{HS}", "").replace("{HE}", "")} for elem in docs_order]
        highlighted_docs_shorter = [{"doc_name":elem["doc_name"],
                                     "doc_text":elem["doc_text"].replace("{HS}", prompt_dict["highlight_start_tkn"]).replace("{HE}", prompt_dict["highlight_end_tkn"])} for elem in docs_order_shorter]
        non_highlighted_docs_shorter = [{"doc_name":elem["doc_name"],
                                         "doc_text":elem["doc_text"].replace("{HS}", "").replace("{HE}", "")} for elem in docs_order_shorter]

        no_highlights_prfx = "gold_" if no_highlights else ""

        additional_data[topic_name] = {"non_highlighted_docs":non_highlighted_docs,
                                       f"{no_highlights_prfx}highlighted_docs":highlighted_docs,
                                       f"{no_highlights_prfx}highlights":curr_highlight_list,
                                       f"non_highlighted_docs_shorter":non_highlighted_docs_shorter,
                                       f"{no_highlights_prfx}highlighted_docs_shorter":highlighted_docs_shorter,
                                       f"{no_highlights_prfx}highlights_shorter":curr_highlight_list_shorter,
                                       "prompt_budget_trace": {
                                           "prompt_token_budget": prompt_budget,
                                           "initial_prompt_tokens": initial_tokens,
                                           "final_prompt_tokens": final_tokens,
                                           "shortening_strategy": shortening_strategy,
                                           "surplus_fraction": selected_surplus,
                                           "transport_scope": "constructed_stage_prompt",
                                       }}

        _q = instance.get("query") or instance.get("question") or ""
        if _q:
            additional_data[topic_name].update({"question": _q})
    return used_demos, final_prompts, additional_data, role_messages
