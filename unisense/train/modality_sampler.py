"""Dependency-light sampler for globally homogeneous target-modality batches."""

import torch
from torch.utils.data import Sampler


class TargetModalityBatchSampler(Sampler):
    """Interleave single-modality global batches at natural data volumes."""

    def __init__(self, modalities, per_device_batch_size, world_size, seed=0):
        self.modalities = list(modalities)
        self.global_batch_size = int(per_device_batch_size) * int(world_size)
        if self.global_batch_size <= 0:
            raise ValueError("global batch size must be positive")
        self.seed = int(seed)
        self.epoch = 0
        counts = {}
        for modality in self.modalities:
            counts[modality] = counts.get(modality, 0) + 1
        self.num_samples = sum(
            count + (-count % self.global_batch_size)
            for count in counts.values()
        )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        pools = {}
        for index, modality in enumerate(self.modalities):
            pools.setdefault(modality, []).append(index)

        global_batches = []
        for modality in sorted(pools):
            pool = pools[modality]
            permutation = torch.randperm(len(pool), generator=generator).tolist()
            shuffled = [pool[position] for position in permutation]
            padding = -len(shuffled) % self.global_batch_size
            if padding:
                shuffled.extend(
                    shuffled[position % len(shuffled)]
                    for position in range(padding)
                )
            global_batches.extend(
                shuffled[start:start + self.global_batch_size]
                for start in range(0, len(shuffled), self.global_batch_size)
            )

        batch_order = torch.randperm(
            len(global_batches), generator=generator).tolist()
        return iter([
            index
            for batch_position in batch_order
            for index in global_batches[batch_position]
        ])
