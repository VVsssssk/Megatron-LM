# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-independent consumer fences for shared replica storage."""

import weakref

import torch


class ReplicaConsumerFence:
    """Serialize consumers of shared replica storage.

    A transfer event only protects a reader from its producer. This fence
    protects the next producer from the previous GEMM/gradient consumer.
    """

    def __init__(self):
        self.owner = None
        self.done = None

    def acquire(self, owner, device):
        """Wait for the previous consumer before assigning the workspace owner."""
        if self.owner is not None:
            raise RuntimeError("Replica workspace still has an active compute consumer.")
        if self.done is not None:
            torch.cuda.current_stream(device).wait_event(self.done)
        self.owner = owner

    def release(self, owner, device):
        """Fence this consumer before allowing the next producer to overwrite."""
        if self.owner is not owner:
            raise RuntimeError("Replica workspace released by a different runtime.")
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(device))
        self.owner = None


_fences = weakref.WeakValueDictionary()


def get_replica_consumer_fence(group, device):
    """Protect replica weights and gradients until the previous consumer completes."""
    key = (group, device)
    fence = _fences.get(key)
    if fence is None:
        fence = ReplicaConsumerFence()
        _fences[key] = fence
    return fence
