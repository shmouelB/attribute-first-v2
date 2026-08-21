"""Process-wide network guard used by the offline unittest suite."""

from __future__ import annotations

import os
import socket
import threading


_INSTALL_LOCK = threading.Lock()
_ORIGINAL_SOCKET = socket.socket


class NetworkAccessBlocked(RuntimeError):
    pass


class _OfflineSocket(_ORIGINAL_SOCKET):
    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkAccessBlocked(
                f"network access blocked by offline test guard: {address!r}"
            )
        return super().connect(address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkAccessBlocked(
                f"network access blocked by offline test guard: {address!r}"
            )
        return super().connect_ex(address)


def _blocked_create_connection(address, *args, **kwargs):
    raise NetworkAccessBlocked(
        f"network access blocked by offline test guard: {address!r}"
    )


def install():
    """Install an idempotent guard before importing application modules."""
    with _INSTALL_LOCK:
        if getattr(socket, "_attribute_first_offline_guard", False):
            return
        socket.socket = _OfflineSocket
        socket.create_connection = _blocked_create_connection
        socket._attribute_first_offline_guard = True

        # Keep the suite offline even though utils.py loads .env via setdefault.
        os.environ["GOOGLE_API_KEY"] = ""
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
