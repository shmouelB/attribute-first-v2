"""Runtime resource inspection for local model loading."""

import logging
import os

import GPUtil
import psutil


class SystemResourceInspector:
    """Compute safe model-loading memory limits for visible GPUs and CPU RAM.

    Providers and the environment mapping are injectable so the policy can be
    checked without depending on the machine running the test.
    """

    FIRST_GPU_MINIMUM_GIB = 60
    FIRST_GPU_RESERVE_GIB = 30
    OTHER_GPU_MINIMUM_GIB = 40
    OTHER_GPU_RESERVE_GIB = 10
    CPU_MINIMUM_GIB = 100
    CPU_MAXIMUM_ALLOCATION_GIB = 100

    def __init__(
        self,
        *,
        environment=None,
        gpu_provider=None,
        memory_provider=None,
        logger=None,
    ):
        self._environment = os.environ if environment is None else environment
        self._gpu_provider = GPUtil if gpu_provider is None else gpu_provider
        self._memory_provider = (
            psutil if memory_provider is None else memory_provider
        )
        self._logger = logging if logger is None else logger

    def get_max_memory(self):
        """Return model-loading limits using the legacy GPU/CPU safety policy."""
        visible_devices = self._visible_gpu_indices()
        gpus = self._gpu_provider.getGPUs()

        max_memory = {}
        visible_gpu_count = 0
        for physical_index, gpu in enumerate(gpus):
            if physical_index not in visible_devices:
                continue
            free_gib = int(gpu.memoryFree / 1024)
            max_memory[visible_gpu_count] = self._gpu_memory_limit(
                free_gib,
                is_first=visible_gpu_count == 0,
            )
            visible_gpu_count += 1

        available_memory_gib = (
            self._memory_provider.virtual_memory().available / (1024 ** 3)
        )
        if available_memory_gib < self.CPU_MINIMUM_GIB:
            raise Exception(
                "Make sure there are at least 100GiB available in the CPU "
                "memory."
            )
        max_memory["cpu"] = (
            f"{min(int(available_memory_gib / 2), self.CPU_MAXIMUM_ALLOCATION_GIB)}GiB"
        )

        self._log_limits(
            max_memory=max_memory,
            visible_devices=visible_devices,
        )
        return max_memory

    def _visible_gpu_indices(self):
        configured_devices = self._environment.get("CUDA_VISIBLE_DEVICES")
        if configured_devices:
            return [
                int(device)
                for device in configured_devices.split(",")
            ]
        return list(range(len(self._gpu_provider.getGPUs())))

    def _gpu_memory_limit(self, free_gib, *, is_first):
        if is_first:
            if free_gib < self.FIRST_GPU_MINIMUM_GIB:
                raise Exception(
                    "Make sure you first visible GPU card has at least 60GiB "
                    "available in memory."
                )
            return f"{free_gib - self.FIRST_GPU_RESERVE_GIB}GiB"

        if free_gib < self.OTHER_GPU_MINIMUM_GIB:
            raise Exception(
                "Make sure all you visible GPU cards have at least 40GiB "
                "available in memory."
            )
        return f"{free_gib - self.OTHER_GPU_RESERVE_GIB}GiB"

    def _log_limits(self, *, max_memory, visible_devices):
        gpu_limits = "\n".join(
            f"card {visible_devices[gpu_index]}: {max_memory[gpu_index]}"
            for gpu_index in range(len(max_memory) - 1)
        )
        limits = (
            f"GPU:\n{gpu_limits}\n"
            f"CPU:\n{max_memory['cpu']}"
        )
        self._logger.info(f"max memory used:\n{limits}")


def get_max_memory():
    """Return safe model-loading limits for resources visible to the process."""
    return SystemResourceInspector().get_max_memory()


__all__ = [
    "SystemResourceInspector",
    "get_max_memory",
]
