"""Fixed-population loading for controlled derived variants."""

import hashlib
import json
from pathlib import Path

from ..domain import CONTROLLED_TEST_POPULATIONS


EXPECTED_TEST_POPULATIONS = CONTROLLED_TEST_POPULATIONS


def read_jsonl_snapshot(path, label):
    """Read one byte-hashed JSONL snapshot without normalizing its payload."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    payload = resolved.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8: {resolved}") from exc

    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} {resolved}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"{label} {resolved}:{line_number}: row must be an object"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {resolved}")
    return {
        "path": resolved,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "payload": payload,
        "rows": rows,
    }


def validated_unique_ids(rows, label):
    """Return ordered IDs while rejecting missing or duplicate identifiers."""
    unique_ids = []
    seen = set()
    for index, row in enumerate(rows, start=1):
        unique_id = row.get("unique_id")
        if not isinstance(unique_id, str) or not unique_id:
            raise ValueError(
                f"{label} row {index} has no non-empty string unique_id"
            )
        if unique_id in seen:
            raise ValueError(f"{label} has duplicate unique_id {unique_id!r}")
        seen.add(unique_id)
        unique_ids.append(unique_id)
    return unique_ids


class PopulationLoader:
    """Load and prove equality with the selected dataset population."""

    def __init__(
        self,
        experiment_root,
        stable_value_sha256,
        expected_populations=None,
    ):
        self.experiment_root = Path(experiment_root).resolve()
        self.stable_value_sha256 = stable_value_sha256
        self.expected_populations = (
            expected_populations or EXPECTED_TEST_POPULATIONS
        )

    def _dataset_snapshot(self, args):
        configured_dataset = getattr(args, "dataset", None)
        dataset_path = (
            Path(configured_dataset)
            if configured_dataset
            else self.experiment_root.parent
            / "data"
            / args.setting
            / f"{args.split}.json"
        )
        return (
            read_jsonl_snapshot(dataset_path, "population reference"),
            configured_dataset,
        )

    def _validate_canonical_fingerprint(
        self,
        args,
        dataset_snapshot,
        dataset_ids,
        configured_dataset,
    ):
        if configured_dataset is not None:
            return
        expected = self.expected_populations[args.setting]
        observed = {
            "count": len(dataset_ids),
            "dataset_sha256": dataset_snapshot["sha256"],
            "unique_ids_sha256": self.stable_value_sha256(
                sorted(dataset_ids)
            ),
        }
        if observed != expected:
            raise ValueError(
                f"canonical {args.setting} test population fingerprint "
                f"mismatch: expected {expected}, observed {observed}"
            )

    @staticmethod
    def _canonicalize_order(input_rows, input_ids, dataset_ids):
        if set(input_ids) != set(dataset_ids):
            missing = sorted(set(dataset_ids) - set(input_ids))
            extra = sorted(set(input_ids) - set(dataset_ids))
            raise ValueError(
                "derived input population does not equal the population "
                f"reference: input={len(input_ids)} "
                f"reference={len(dataset_ids)} missing={missing} extra={extra}"
            )
        normalization = {
            "applied": input_ids != dataset_ids,
            "method": "unique_id_reindex_exact_set",
            "original_unique_ids": list(input_ids),
            "canonical_unique_ids": list(dataset_ids),
        }
        if not normalization["applied"]:
            return list(input_rows), normalization
        rows_by_id = {
            row["unique_id"]: row for row in input_rows
        }
        return (
            [rows_by_id[unique_id] for unique_id in dataset_ids],
            normalization,
        )

    @staticmethod
    def _validate_reference_rows(args, input_rows, dataset_by_id):
        for unique_id, reference_row in dataset_by_id.items():
            response = reference_row.get("response")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"population reference {unique_id!r} has no non-empty "
                    "gold response"
                )
        if args.setting != "LFQA":
            return
        for row in input_rows:
            unique_id = row["unique_id"]
            source_query = row.get("query")
            reference_query = dataset_by_id[unique_id].get("query")
            if (
                not isinstance(reference_query, str)
                or not reference_query.strip()
            ):
                raise ValueError(
                    f"population reference {unique_id!r} has no LFQA query"
                )
            if source_query != reference_query:
                raise ValueError(
                    "derived input query differs from the population "
                    f"reference for {unique_id!r}"
                )

    def load(self, args):
        input_snapshot = read_jsonl_snapshot(args.cs, "derived input")
        input_ids = validated_unique_ids(
            input_snapshot["rows"],
            "derived input",
        )
        dataset_snapshot, configured_dataset = self._dataset_snapshot(args)
        dataset_ids = validated_unique_ids(
            dataset_snapshot["rows"],
            "population reference",
        )
        self._validate_canonical_fingerprint(
            args,
            dataset_snapshot,
            dataset_ids,
            configured_dataset,
        )
        selected_rows, normalization = self._canonicalize_order(
            input_snapshot["rows"],
            input_ids,
            dataset_ids,
        )
        max_examples = getattr(args, "max_examples", None)
        if max_examples is not None:
            raise ValueError(
                "controlled derived variants require the complete canonical "
                "population; max_examples is not allowed"
            )
        dataset_by_id = {
            row["unique_id"]: row for row in dataset_snapshot["rows"]
        }
        self._validate_reference_rows(
            args,
            selected_rows,
            dataset_by_id,
        )
        return {
            "input": input_snapshot,
            "reference": dataset_snapshot,
            "full_input_ids": dataset_ids,
            "reference_ids": dataset_ids,
            "selected_rows": selected_rows,
            "selected_ids": dataset_ids,
            "dataset_by_id": dataset_by_id,
            "max_examples": max_examples,
            "input_order_normalization": normalization,
        }
