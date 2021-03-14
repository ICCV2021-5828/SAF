
import torch
from torch.utils.data import Dataset


class TensorLoader(Dataset):

    def __init__(
            self,
            path: str,
            transform=None,
            target_transform=None,
    ):
        self.image_list, self.label_list = torch.load(path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.image_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

    def _load_one_item(self, idx: int):
        image = self.image_list[idx]
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

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
            raise ValueError

        return sample, target
