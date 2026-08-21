"""Shared environment protocol for stateful dialogue runs."""

import json
from contextlib import contextmanager


@contextmanager
def dialogue_protocol_environment(
    full_configs,
    *,
    validate_protocol_environment_flags,
    protocol_environment,
):
    """Apply one reproducible AF_* protocol across a dialogue."""
    declared_by_stage = []
    models_by_stage = []
    for stage in full_configs:
        with open(
            stage["config_file"],
            "r",
            encoding="utf-8",
        ) as config_file:
            config = json.load(config_file)
        declared_by_stage.append(
            (
                stage["subtask"],
                validate_protocol_environment_flags(
                    config.get("protocol")
                ),
            )
        )
        models_by_stage.append(
            (stage["subtask"], config.get("model_name"))
        )

    reference = declared_by_stage[0][1] if declared_by_stage else {}
    if any(flags != reference for _, flags in declared_by_stage[1:]):
        details = ", ".join(
            f"{subtask}={flags}"
            for subtask, flags in declared_by_stage
        )
        raise ValueError(
            "all dialogue stages must declare the same "
            f"protocol.environment_flags; got {details}"
        )
    invalid_models = [
        (subtask, model)
        for subtask, model in models_by_stage
        if not isinstance(model, str) or not model.strip()
    ]
    if invalid_models:
        details = ", ".join(
            f"{subtask}={model!r}"
            for subtask, model in invalid_models
        )
        raise ValueError(
            "every dialogue stage must declare a non-empty string "
            f"model_name; got {details}"
        )
    reference_model = models_by_stage[0][1] if models_by_stage else None
    if any(
        model != reference_model
        for _, model in models_by_stage[1:]
    ):
        details = ", ".join(
            f"{subtask}={model!r}"
            for subtask, model in models_by_stage
        )
        raise ValueError(
            "all dialogue stages must declare the same model_name; "
            f"got {details}"
        )

    with protocol_environment(
        {"environment_flags": reference}
    ) as effective:
        yield effective


__all__ = ["dialogue_protocol_environment"]
