
import torch
from torchvision.datasets import ImageFolder


class TmageFolder(ImageFolder):

    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
    ):
        super(TmageFolder, self).__init__(root, transform=transform, target_transform=target_transform)

    def _load_one_item(self, idx: int):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __getitem__(self, index):
        if isinstance(index, int):
            sample, target = self._load_one_item(index)
        elif isinstance(index, list):
            assert len(index) > 0
            samples = [None for _ in range(len(index))]
            targets = [None for _ in range(len(index))]
            for i, idx in enumerate(index):
                sample, target = self._load_one_item(idx)
                samples[i] = sample
                targets[i] = target
            sample = torch.stack(samples)
            target = torch.LongTensor(targets)
        else:
            raise ValueError('Input index should be of type int or list(int)!')

        return sample, target
