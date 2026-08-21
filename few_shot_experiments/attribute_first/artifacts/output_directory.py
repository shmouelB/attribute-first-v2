"""Atomic ownership policy for append-safe generation directories."""

import os
from pathlib import Path


class OutputDirectoryClaim:
    """Exclusively claim a new or empty generation output directory."""

    CLAIM_NAME = ".generation_run_claim"

    @classmethod
    def claim(
        cls,
        path: str | Path,
        *,
        owner: str,
    ) -> Path:
        """Create and atomically claim ``path`` before any artifact write."""

        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("output directory owner must be non-empty")
        outdir = Path(path).expanduser().resolve()
        if not outdir.exists():
            try:
                outdir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                pass
        if outdir.exists():
            if not outdir.is_dir():
                raise ValueError(
                    f"outdir is not a directory: {outdir}"
                )
            if any(outdir.iterdir()):
                raise ValueError(
                    "outdir must be new or empty to remain append-safe; "
                    f"refusing non-empty directory: {outdir}"
                )

        claim_path = outdir / cls.CLAIM_NAME
        try:
            claim_fd = os.open(
                claim_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise ValueError(
                f"outdir is already claimed by another run: {outdir}"
            ) from exc
        with os.fdopen(claim_fd, "w", encoding="utf-8") as claim_file:
            claim_file.write(f"{owner}\n")
            claim_file.flush()
            os.fsync(claim_file.fileno())
        return outdir

    @classmethod
    def prepare_child(
        cls,
        path: str | Path,
        *,
        owner_root: str | Path,
    ) -> Path:
        """Prepare a child owned by an already claimed pipeline root."""

        outdir = Path(path).expanduser().resolve()
        root = Path(owner_root).expanduser().resolve()
        if outdir != root and root not in outdir.parents:
            raise ValueError(
                f"stage outdir escapes claimed pipeline root: {outdir}"
            )
        if not (root / cls.CLAIM_NAME).is_file():
            raise ValueError(
                f"pipeline root has no output claim: {root}"
            )
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir


__all__ = ["OutputDirectoryClaim"]
