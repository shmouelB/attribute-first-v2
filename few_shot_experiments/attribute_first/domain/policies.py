"""Immutable scientific and execution policies for controlled experiments."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass

from .enums import Dataset


def _is_positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


class _ImmutableDictionary(dict):
    """JSON-compatible dictionary that rejects in-place mutation."""

    @staticmethod
    def _reject_mutation(*_args, **_kwargs):
        raise TypeError("controlled population contracts are immutable")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation
    __ior__ = _reject_mutation

    def __deepcopy__(self, memo):
        """Return a mutable archival snapshot without exposing canonical state."""

        return {
            copy.deepcopy(key, memo): copy.deepcopy(value, memo)
            for key, value in self.items()
        }


@dataclass(frozen=True, slots=True)
class PopulationFingerprint:
    """Exact size and byte/identity hashes of one controlled population."""

    count: int
    dataset_sha256: str
    unique_ids_sha256: str

    def __post_init__(self) -> None:
        if not _is_positive_integer(self.count):
            raise ValueError("population count must be a positive integer")
        for name, value in (
            ("dataset_sha256", self.dataset_sha256),
            ("unique_ids_sha256", self.unique_ids_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    def as_mapping(self) -> Mapping[str, object]:
        """Return the immutable legacy JSON shape used by artifact schemas."""

        return _ImmutableDictionary(
            {
                "count": self.count,
                "dataset_sha256": self.dataset_sha256,
                "unique_ids_sha256": self.unique_ids_sha256,
            }
        )


class PopulationContract(_ImmutableDictionary):
    """One immutable mapping shared by every controlled-run boundary."""

    def __init__(
        self,
        fingerprints: Mapping[Dataset, PopulationFingerprint],
    ) -> None:
        entries = dict(fingerprints)
        if set(entries) != set(Dataset):
            raise ValueError(
                "population contract must define every supported dataset"
            )
        if any(
            not isinstance(dataset, Dataset)
            or not isinstance(fingerprint, PopulationFingerprint)
            for dataset, fingerprint in entries.items()
        ):
            raise ValueError(
                "population contract entries must map Dataset values to "
                "PopulationFingerprint values"
            )
        dict.__init__(
            self,
            {
                dataset.value: entries[dataset].as_mapping()
                for dataset in Dataset
            },
        )


CONTROLLED_TEST_POPULATIONS = PopulationContract(
    {
        Dataset.MDS: PopulationFingerprint(
            count=65,
            dataset_sha256=(
                "5f53a60d96c92c9d32b3a0e76916500d"
                "b21aef60d5514ae6ee81b3e638ec0a1f"
            ),
            unique_ids_sha256=(
                "d07270e4ffeb5e78cc52c18d33e8dd20"
                "b2e6b141874f5230fbe4982dd030b8d8"
            ),
        ),
        Dataset.LFQA: PopulationFingerprint(
            count=45,
            dataset_sha256=(
                "84d66686176aaf0144744216d44d346a3"
                "63d63e3d775889b47f083fc58c38780"
            ),
            unique_ids_sha256=(
                "2dc40e707c15a656f818b3ac31467e9aa"
                "03b80d6ae71a7652aff5a17c03dc1f0"
            ),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bound the complete provider attempts made for one stage."""

    max_attempts: int
    parseable_finish_reasons: frozenset[str] = frozenset(
        {"FINISH_REASON_UNSPECIFIED", "STOP"}
    )

    def __post_init__(self) -> None:
        if not _is_positive_integer(self.max_attempts):
            raise ValueError("max_attempts must be a positive integer")

        reasons = frozenset(self.parseable_finish_reasons)
        if not reasons or any(
            not isinstance(reason, str) or not reason
            for reason in reasons
        ):
            raise ValueError(
                "parseable_finish_reasons must contain non-empty strings"
            )
        object.__setattr__(self, "parseable_finish_reasons", reasons)


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Prompt-token ceilings for isolated stages and accumulated dialogue."""

    stage_prompt_tokens: int
    dialogue_history_prompt_tokens: int

    def __post_init__(self) -> None:
        if not _is_positive_integer(self.stage_prompt_tokens):
            raise ValueError(
                "stage_prompt_tokens must be a positive integer"
            )
        if not _is_positive_integer(
            self.dialogue_history_prompt_tokens
        ):
            raise ValueError(
                "dialogue_history_prompt_tokens must be a positive integer"
            )
        if (
            self.dialogue_history_prompt_tokens
            < self.stage_prompt_tokens
        ):
            raise ValueError(
                "dialogue history budget cannot be smaller than stage budget"
            )


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Separate application-managed caching from provider-side reuse."""

    explicit_context_cache: bool
    provider_implicit_reuse_allowed: bool

    def __post_init__(self) -> None:
        if type(self.explicit_context_cache) is not bool:
            raise ValueError("explicit_context_cache must be boolean")
        if type(self.provider_implicit_reuse_allowed) is not bool:
            raise ValueError(
                "provider_implicit_reuse_allowed must be boolean"
            )
