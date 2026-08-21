"""Pure text-normalization and highlight-rendering helpers.

These functions intentionally remain stateless functions: they transform
their inputs without owning runtime configuration, provider state, or
experiment invariants.
"""

import re
import string
from typing import List


SPAN_SEP = "<HIGHLIGHT_SEP>"
SENT_SEP = "<SENT_SEP>"


def highlight_sep_strip(txt):
    """Remove leading and trailing ``SPAN_SEP`` markers."""
    txt = txt.strip()
    if txt.startswith(SPAN_SEP):
        txt = txt[len(SPAN_SEP):]
    if txt.endswith(SPAN_SEP):
        txt = txt[:-len(SPAN_SEP)]
    return txt


def find_substring_indices(s, sub):
    indices = []
    i = s.find(sub)
    while i >= 0:
        indices.append(i)
        i = s.find(sub, i + 1)
    return indices


def longest_common_subsequence(list1, list2):
    m = len(list1)
    n = len(list2)
    dp = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i = m
    j = n
    lcs = []
    indices1 = []
    indices2 = []
    while i > 0 and j > 0:
        if list1[i - 1] == list2[j - 1]:
            lcs.insert(0, list1[i - 1])
            indices1.insert(0, i - 1)
            indices2.insert(0, j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return lcs, indices1, indices2


def rmv_txt_after_last_highlight(text, highlight_end_tkn):
    shortened_text = (
        highlight_end_tkn.join(text.split(highlight_end_tkn)[:-1])
        + highlight_end_tkn
    )
    if shortened_text.strip() == highlight_end_tkn:
        return ""
    return shortened_text


def rmv_spaces_and_punct(txt):
    return txt.lower().translate(
        str.maketrans("", "", string.whitespace + string.punctuation)
    )


def remove_spaces_and_punctuation(text):
    return "".join(char for char in text if char.isalnum())


def find_substring(s, sb):
    if sb.lower().strip() in s.lower():
        start_index = s.lower().index(sb.lower().strip())
        return start_index, start_index + len(sb.strip())

    modified_s = remove_spaces_and_punctuation(s).lower()
    modified_sb = remove_spaces_and_punctuation(sb).lower()
    index = modified_s.find(modified_sb)
    if index == -1:
        return -1, -1

    actual_start_index = 0
    count = 0
    for char in s:
        is_char_removed = remove_spaces_and_punctuation(char) == ""
        if count == index and not is_char_removed:
            break
        if not is_char_removed:
            count += 1
        actual_start_index += 1

    actual_end_index = actual_start_index
    modified_sb_length = len(modified_sb)
    while modified_sb_length > 0:
        if s[actual_end_index].isalnum():
            modified_sb_length -= 1
        actual_end_index += 1

    if remove_spaces_and_punctuation(
        sb.lower()
    ) != remove_spaces_and_punctuation(
        s[actual_start_index:actual_end_index].lower()
    ):
        raise ValueError("found substring doesn't match indices")
    return actual_start_index, actual_end_index


def get_consecutive_subspans(idx_lst):
    if not idx_lst:
        return []
    idx_subspans = []
    low_lim, up_lim = -1, -1
    for i in range(len(idx_lst) - 1):
        if low_lim == -1:
            low_lim = idx_lst[i]
            up_lim = -1
        if idx_lst[i + 1] > idx_lst[i] + 1:
            up_lim = idx_lst[i]
            idx_subspans.append([low_lim, up_lim])
            low_lim = -1
    if low_lim == -1:
        idx_subspans.append([idx_lst[-1], idx_lst[-1]])
    else:
        idx_subspans.append([low_lim, idx_lst[-1]])
    return idx_subspans


def merge_spans(spans: List[List[List[int]]]) -> List[List[int]]:
    flattened_spans = [span for elem in spans for span in elem]
    idxs = sorted(
        idx
        for span in flattened_spans
        for idx in range(span[0], span[1] + 1)
    )
    return get_consecutive_subspans(idxs)


def add_highlights(
    doc,
    highlights,
    highlight_start_tkn,
    highlight_end_tkn,
):
    if not highlights:
        return doc
    highlights = sorted(highlights, key=lambda x: x[0])
    highlighted_doc = doc[:highlights[0][0]]

    for i, span in enumerate(highlights):
        end_idx_non_highlighted = (
            highlights[i + 1][0]
            if i < len(highlights) - 1
            else len(doc)
        )
        addition_txt = (
            highlight_start_tkn
            + doc[span[0]:span[1]]
            + highlight_end_tkn
            + doc[span[1]:end_idx_non_highlighted]
        )
        highlighted_doc += addition_txt

    if (
        highlighted_doc
        .replace(highlight_start_tkn, "")
        .replace(highlight_end_tkn, "")
        != doc
    ):
        raise ValueError("highlight insertion changed the source document")
    return highlighted_doc


def extract_highlights(
    highlighted_doc,
    highlight_start_tkn,
    highlight_end_tkn,
):
    pattern = fr"{highlight_start_tkn}(.*?){highlight_end_tkn}"
    return re.findall(pattern, highlighted_doc, re.DOTALL)


def get_highlighted_doc(
    docs,
    highlights,
    highlight_start_tkn,
    highlight_end_tkn,
    merge_cross_sents_highlights: bool = False,
    doc_sents: List = None,
):
    highlighted_docs = {}
    for doc_name, doc_text in docs.items():
        curr_highlights = [
            elem
            for elem in highlights
            if elem["documentFile"] == doc_name
        ]

        if not merge_cross_sents_highlights:
            curr_merged_spans = []
            sentence_starts = {
                elem["docSentCharIdx"] for elem in curr_highlights
            }
            for doc_sent_char_idx in sentence_starts:
                sentence_highlights = [
                    elem
                    for elem in curr_highlights
                    if elem["docSentCharIdx"] == doc_sent_char_idx
                ]
                sentence_spans = [
                    elem["docSpanOffsets"]
                    for elem in sentence_highlights
                ]
                if not all(
                    highlight_sep_strip(
                        sentence_highlights[i]["docSpanText"].replace(
                            SENT_SEP,
                            "",
                        )
                    )
                    .replace(" ", "")
                    .replace("\n", "")
                    .lower()
                    == highlight_sep_strip(
                        SPAN_SEP.join(
                            doc_text[span[0]:span[1]]
                            for span in instance
                        )
                    )
                    .replace(" ", "")
                    .replace("\n", "")
                    .lower()
                    for i, instance in enumerate(sentence_spans)
                ):
                    raise ValueError(
                        "not all docSpanOffsets align with the docSpanText "
                        f"for docSentCharIdx {doc_sent_char_idx}"
                    )
                curr_merged_spans.extend(merge_spans(sentence_spans))
        else:
            curr_highlights_spans = [
                elem["docSpanOffsets"] for elem in curr_highlights
            ]
            if not all(
                highlight_sep_strip(
                    curr_highlights[i]["docSpanText"].replace(SENT_SEP, "")
                )
                .replace(" ", "")
                .replace("\n", "")
                .lower()
                == highlight_sep_strip(
                    SPAN_SEP.join(
                        doc_text[span[0]:span[1]] for span in instance
                    )
                )
                .replace(" ", "")
                .replace("\n", "")
                .lower()
                for i, instance in enumerate(curr_highlights_spans)
            ):
                raise ValueError(
                    "not all docSpanOffsets align with the docSpanText"
                )
            curr_merged_spans = merge_spans(curr_highlights_spans)

        curr_merged_spans = [
            list(span) for span in {tuple(elem) for elem in curr_merged_spans}
        ]
        curr_merged_spans = [
            span
            for span in curr_merged_spans
            if rmv_spaces_and_punct(doc_text[span[0]:span[1]])
        ]
        curr_merged_spans = sorted(
            curr_merged_spans,
            key=lambda x: x[0],
        )
        curr_merged_spans = [
            elem for elem in curr_merged_spans if elem[0] < elem[1]
        ]

        cleaned = []
        for elem in curr_merged_spans:
            if not cleaned or elem[0] >= cleaned[-1][1]:
                cleaned.append(elem)
        curr_merged_spans = cleaned

        highlighted_docs[doc_name] = add_highlights(
            doc_text,
            curr_merged_spans,
            highlight_start_tkn,
            highlight_end_tkn,
        )
        if extract_highlights(
            highlighted_docs[doc_name],
            highlight_start_tkn,
            highlight_end_tkn,
        ) != [
            doc_text[span[0]:span[1]] for span in curr_merged_spans
        ]:
            raise ValueError("the correct spans weren't highlighted")

    return highlighted_docs


def _merge_doc_spans(
    doc_text,
    curr_highlights,
    merge_cross_sents_highlights,
):
    """Merge one document's offset spans for two-set rendering."""
    if not curr_highlights:
        return []
    if not merge_cross_sents_highlights:
        curr_merged_spans = []
        sentence_starts = {
            element["docSentCharIdx"] for element in curr_highlights
        }
        for doc_sent_char_idx in sentence_starts:
            spans = [
                element["docSpanOffsets"]
                for element in curr_highlights
                if element["docSentCharIdx"] == doc_sent_char_idx
            ]
            curr_merged_spans.extend(merge_spans(spans))
    else:
        curr_merged_spans = merge_spans(
            [element["docSpanOffsets"] for element in curr_highlights]
        )
    curr_merged_spans = [
        list(span) for span in {tuple(elem) for elem in curr_merged_spans}
    ]
    curr_merged_spans = [
        span
        for span in curr_merged_spans
        if rmv_spaces_and_punct(doc_text[span[0]:span[1]])
    ]
    curr_merged_spans = sorted(curr_merged_spans, key=lambda x: x[0])
    curr_merged_spans = [
        span
        for index, span in enumerate(curr_merged_spans)
        if index == 0
        or span[0] >= curr_merged_spans[index - 1][1]
    ]
    return curr_merged_spans


def add_highlights_typed(doc, typed_spans):
    """Insert the token pair carried by each non-overlapping span."""
    if not typed_spans:
        return doc
    typed_spans = sorted(typed_spans, key=lambda x: x[0])
    result = doc[:typed_spans[0][0]]
    for index, (start, end, start_token, end_token) in enumerate(
        typed_spans
    ):
        next_start = (
            typed_spans[index + 1][0]
            if index < len(typed_spans) - 1
            else len(doc)
        )
        result += (
            start_token
            + doc[start:end]
            + end_token
            + doc[end:next_start]
        )
    return result


def get_highlighted_doc_two_sets(
    docs,
    evidence_h,
    context_h,
    hs_tkn,
    he_tkn,
    cs_tkn,
    ce_tkn,
    merge_cross_sents_highlights=False,
    doc_sents=None,
):
    """Render evidence and context spans, giving evidence precedence."""
    output = {}
    for doc_name, doc_text in docs.items():
        evidence = _merge_doc_spans(
            doc_text,
            [
                element
                for element in evidence_h
                if element["documentFile"] == doc_name
            ],
            merge_cross_sents_highlights,
        )
        context = _merge_doc_spans(
            doc_text,
            [
                element
                for element in context_h
                if element["documentFile"] == doc_name
            ],
            merge_cross_sents_highlights,
        )
        typed = [
            (span[0], span[1], hs_tkn, he_tkn) for span in evidence
        ] + [
            (span[0], span[1], cs_tkn, ce_tkn) for span in context
        ]
        typed = sorted(typed, key=lambda x: x[0])
        cleaned = []
        for typed_span in typed:
            if cleaned and typed_span[0] < cleaned[-1][1]:
                continue
            cleaned.append(typed_span)
        output[doc_name] = add_highlights_typed(doc_text, cleaned)
    return output


__all__ = [
    "SPAN_SEP",
    "SENT_SEP",
    "add_highlights",
    "add_highlights_typed",
    "extract_highlights",
    "find_substring",
    "find_substring_indices",
    "get_consecutive_subspans",
    "get_highlighted_doc",
    "get_highlighted_doc_two_sets",
    "highlight_sep_strip",
    "longest_common_subsequence",
    "merge_spans",
    "remove_spaces_and_punctuation",
    "rmv_spaces_and_punct",
    "rmv_txt_after_last_highlight",
]
