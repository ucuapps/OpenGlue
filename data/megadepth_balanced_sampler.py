import torch
import torch.distributed as dist


class MegaDepthBalancedSampler(torch.utils.data.distributed.DistributedSampler):
    """Derive from DistributedSampler for compatability with PytorchLightning.
    Can sample the same indices at each process"""
    def __init__(self, dataset, seed: int = 0):
        self.num_samples = len(dataset)
        self._generator = None
        if not dist.is_available():
            self.rank = 0
        else:
            self.rank = dist.get_rank()
        # generate seed depending on rank
        self.seed = torch.randint(
            torch.iinfo(torch.int64).min,
            torch.iinfo(torch.int64).max,
            (),
            generator=torch.Generator().manual_seed(seed + self.rank)).item()

        self.pairs_per_scene = torch.tensor(list(dataset.scene_pairs_numbers.values()))
        self.scene_weights = 1.0 / self.pairs_per_scene

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self._generator = g
        return self

    def __next__(self):
        scene_idx = torch.multinomial(self.scene_weights, num_samples=1, generator=self._generator).item()
        image_idx = torch.randint(self.pairs_per_scene[scene_idx], size=())
        idx = torch.sum(self.pairs_per_scene[:scene_idx]) + image_idx
        return idx

    def __len__(self):
        return self.num_samples
