"""Pure prompt-template rendering helpers."""

from .highlights import extract_highlights


def one_doc_fusion_prompt(doc_cluster, single_doc_prompt):
    highlights_text = ",".join(
        str(highlight) for highlight in doc_cluster["highlights"]
    )
    return single_doc_prompt.replace(
        "{ID}",
        str(doc_cluster["doc"]),
    ).replace(
        "{HLIST_DOC_FUSION}",
        highlights_text,
    )


def make_highlights_fusion_prompt(
    highlights_fuse_dict,
    sent_id,
    curr_prompt,
):
    sent_text = highlights_fuse_dict["output"]
    fusion_dict = sorted(
        highlights_fuse_dict["highlights_cluster"],
        key=lambda x: x["doc"],
    )
    fusion_dict = [
        {
            "doc": element["doc"],
            "highlights": sorted(element["highlights"]),
        }
        for element in fusion_dict
    ]
    highlight_indices = sorted(
        highlight_index
        for doc_cluster in fusion_dict
        for highlight_index in doc_cluster["highlights"]
    )
    fusion_text = ",".join(
        str(highlight_index) for highlight_index in highlight_indices
    )
    return (
        curr_prompt
        .replace("{HFUSE_SENT}", fusion_text)
        .replace("{SENT_ID}", str(sent_id + 1))
        .replace("{SENT}", sent_text)
    )


def make_clustering_prompt(
    highlights_fuse_dict,
    answer_clustering_format,
):
    current_highlights = sorted(
        highlight_index
        for element in highlights_fuse_dict["highlights_cluster"]
        for highlight_index in element["highlights"]
    )
    highlights_text = ",".join(
        str(highlight_index) for highlight_index in current_highlights
    )
    return (
        answer_clustering_format
        .replace("{HCLUSTER}", highlights_text)
        .replace("{COT_HCLUSTER}", highlights_text)
        .replace(
            "{COT_HCLUSTER_TOPIC}",
            highlights_fuse_dict["cluster_CoT_topic"],
        )
    )


def make_content_selection_prompt(
    highlights_list,
    doc_id,
    answer_content_selection_format,
):
    text = " <SPAN_DELIM> ".join(highlights_list)
    return answer_content_selection_format.replace(
        "{ID}",
        str(doc_id + 1),
    ).replace(
        "{CONTENT_LIST}",
        text,
    )


def make_highlights_listing_prompt(
    highlights_list,
    doc_id,
    highlights_cnt,
    curr_prompt,
):
    text = "\n".join(
        f"{index + highlights_cnt + 1}. {highlight}"
        for index, highlight in enumerate(highlights_list)
    )
    return curr_prompt.replace(
        "{HLIST_DOC}",
        text,
    ).replace(
        "{ID}",
        str(doc_id + 1),
    )


def make_doc_prompt(doc, doc_id, doc_prompt):
    return doc_prompt.replace(
        "{P}",
        doc["text"],
    ).replace(
        "{ID}",
        str(doc_id + 1),
    )


def make_ALCE_prompt(sent_plan, prompt):
    cited_docs_ids = sorted(
        element["doc"] for element in sent_plan["highlights_cluster"]
    )
    cited_docs_str = "".join(f"[{doc_id}]" for doc_id in cited_docs_ids)
    return prompt.replace(
        "{ALCE_SENT}",
        sent_plan["output"],
    ).replace(
        "{ALCE_CITATIONS}",
        cited_docs_str,
    )


def make_demo(
    item,
    prompt,
    doc_prompt=None,
    instruction=None,
    answer_related_prompts=None,
    highlight_start_tkn=None,
    highlight_end_tkn=None,
    test=False,
    content_selection=False,
):
    prompt = prompt.replace("{INST}", instruction)
    doc_list = item["docs"]
    if "{D}" in prompt:
        text = "".join(
            make_doc_prompt(doc, doc_id, doc_prompt)
            for doc_id, doc in enumerate(doc_list)
        )
        prompt = prompt.replace("{D}", text)

    if content_selection:
        prompt = prompt.replace("{HS}", "").replace("{HE}", "")

    if "{A}" in prompt:
        prompt = prompt.replace(
            "{A}",
            answer_related_prompts["answer_prompt"],
        )

    if "{Q}" in prompt:
        prompt = prompt.replace("{Q}", item["question"])

    if ("{HS}" in prompt) != ("{HE}" in prompt):
        raise ValueError(
            "prompt template must contain both {HS} and {HE}, or neither"
        )

    if "{HS}" in prompt:
        prompt = (
            prompt
            .replace("{HS}", highlight_start_tkn)
            .replace("{HE}", highlight_end_tkn)
        )

    highlight_lists = [
        extract_highlights(doc["text"], "{HS}", "{HE}")
        for doc in doc_list
    ]

    if "{HLIST}" in prompt:
        highlights_list_text = ""
        highlights_cnt = 0
        for doc_id, _ in enumerate(doc_list):
            highlights_list_text += make_highlights_listing_prompt(
                highlights_list=highlight_lists[doc_id],
                doc_id=doc_id,
                highlights_cnt=highlights_cnt,
                curr_prompt=answer_related_prompts[
                    "answer_highlights_listing_prompt"
                ],
            )
            highlights_cnt += len(highlight_lists[doc_id])
        prompt = prompt.replace("{HLIST}", highlights_list_text)

    if "{PRFX}" in prompt:
        prompt = prompt.replace("{PRFX}", item["prefix"])

    if not test:
        if "{HDOCS}" in prompt:
            doc_list = item["docs"]
            current_answer_format = answer_related_prompts[
                "answer_content_selection_format"
            ]
            text = "\n".join(
                make_content_selection_prompt(
                    highlights_list=highlight_lists[doc_id],
                    doc_id=doc_id,
                    answer_content_selection_format=current_answer_format,
                )
                for doc_id, _ in enumerate(doc_list)
            )
            prompt = prompt.replace("{HDOCS}", text)
            prompt = (
                prompt
                .replace("{HS}", highlight_start_tkn)
                .replace("{HE}", highlight_end_tkn)
                .strip()
            )

        if "{PLANNING}" in prompt:
            prompt = prompt.replace(
                "{PLANNING}",
                answer_related_prompts["answer_FiC_planning_prompt"],
            )
            fusion_text = "".join(
                make_highlights_fusion_prompt(
                    highlights_fuse_dict=fusion_dict,
                    sent_id=sent_id,
                    curr_prompt=answer_related_prompts[
                        "answer_highlights_fusion_prompt"
                    ],
                )
                for sent_id, fusion_dict in enumerate(item["planning"])
            )
            prompt = (
                prompt
                .replace("{HFUSE}", fusion_text)
                .replace("{SUMM}", item["answer"])
            )

        if "{CoT_RESP}" in prompt:
            prompt = prompt.replace("{CoT_RESP}", item["answer"])

        if "{RESP}" in prompt:
            prompt = prompt.replace("{RESP}", item["answer"])

        if "{ALCE_RESP}" in prompt:
            alce_response = " ".join(
                make_ALCE_prompt(
                    sent_plan=element,
                    prompt=answer_related_prompts["answer_ALCE_format"],
                )
                for element in item["planning"]
            )
            prompt = prompt.replace("{ALCE_RESP}", alce_response)

        if "{CoT_CLUSTERING}" in prompt:
            prompt = prompt.replace(
                "{CoT_CLUSTERING}",
                answer_related_prompts[
                    "answer_clustering_CoT_prompt_intermediate"
                ],
            )
            cot_clustering_text = "".join(
                make_clustering_prompt(
                    highlights_fuse_dict=fusion_dict,
                    answer_clustering_format=answer_related_prompts[
                        "answer_clustering_CoT_format"
                    ],
                )
                for _, fusion_dict in enumerate(item["planning"])
            )
            prompt = prompt.replace(
                "{CHAINS_CLUSTERING}",
                cot_clustering_text,
            )

        if "{CLUSTERS}" in prompt:
            clustering_text = ",".join(
                make_clustering_prompt(
                    highlights_fuse_dict=fusion_dict,
                    answer_clustering_format=answer_related_prompts[
                        "answer_clustering_format"
                    ],
                )
                for _, fusion_dict in enumerate(item["planning"])
            )
            prompt = prompt.replace(
                "{CLUSTERS}",
                f"[{clustering_text}]",
            )

        if "{NEXT_SENT}" in prompt:
            prompt = prompt.replace("{NEXT_SENT}", item["answer"])

        if "{AH_RESP}" in prompt:
            prompt = prompt.replace(
                "{AH_RESP}",
                item.get("ambiguity_highlight_answer", ""),
            )
    else:
        if not answer_related_prompts["answer_prompt"].startswith("Answer:"):
            prompt = prompt.replace(
                answer_related_prompts["answer_prompt"],
                "",
            )
        prompt = (
            prompt
            .replace("{PLANNING}", "")
            .replace("{CLUSTERS}", "")
            .replace("{CoT_CLUSTERING}", "")
            .replace("{HDOCS}", "")
            .replace("{NEXT_SENT}", "")
            .replace("{CoT_RESP}", "")
            .replace("{RESP}", "")
            .replace("{ALCE_RESP}", "")
            .replace("{AH_RESP}", "")
            .strip()
        )
    return prompt, highlight_lists


__all__ = [
    "make_ALCE_prompt",
    "make_clustering_prompt",
    "make_content_selection_prompt",
    "make_demo",
    "make_doc_prompt",
    "make_highlights_fusion_prompt",
    "make_highlights_listing_prompt",
    "one_doc_fusion_prompt",
]
