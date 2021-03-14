
from collections import OrderedDict
import random

import torch
from torch.utils.data import Sampler
from tqdm import tqdm

from .. import cfg
from ..utils import ConditionalEntropy


class TaskSampler:
    __instance = None

    def __new__(cls, *args):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args)
        return cls.__instance

    def __init__(self):
        self.unique_classes = [i for i in range(cfg.args.class_num)]
        self.sampled_classes = None
        self.last_seen_iter = -1

    def sample_N_classes_as_a_task(self):
        # need to make sure that source and target sample the same set of classes
        assert cfg.args.sampler_n_way <= cfg.args.class_num
        if self.sampled_classes is None or self.last_seen_iter != cfg.iter_count:
            self.sampled_classes = random.sample(self.unique_classes, cfg.args.sampler_n_way)  # sample w/o replacement
            self.last_seen_iter = cfg.iter_count
        return self.sampled_classes


class N_Way_K_Shot_BatchSampler(Sampler):
    def __init__(self, y):
        # self.y = y
        self.length = cfg.SAMPLING_BATCH_COUNT
        self.label_dict = self._build_label_dict(y)
        # self.unique_classes_from_y = sorted(set(self.y))

    @staticmethod
    def _build_label_dict(y):
        label_dict = OrderedDict()
        for i, label in enumerate(y):
            if label not in label_dict:
                label_dict[label] = [i]
            else:
                label_dict[label].append(i)
        return label_dict

    def _sample_examples_by_class(self, cls):
        if cls not in self.label_dict:
            return []

        if cfg.args.sampler_k_shot <= len(self.label_dict[cls]):
            sampled_examples = random.sample(
                self.label_dict[cls], k=cfg.args.sampler_k_shot)  # sample without replacement
        else:
            sampled_examples = random.choices(
                self.label_dict[cls], k=cfg.args.sampler_k_shot)  # sample with replacement
        return sampled_examples

    def __iter__(self):
        for _ in range(self.length):
            batch = []
            classes = TaskSampler().sample_N_classes_as_a_task()
            for cls in classes:
                samples_for_this_class = self._sample_examples_by_class(cls)
                batch.extend(samples_for_this_class)
            # assert len(batch) <= cfg.args.sampler_n_way * cfg.args.sampler_k_shot
            assert len(batch) > 0
            yield batch

    def __len__(self):
        return self.length


class SelfTrainingBaseSampler(Sampler[int]):
    def __init__(self, dataset):
        self.length = cfg.SAMPLING_BATCH_COUNT
        self.dataset = dataset
        self.probs, self.y_hat, self.y_prob = None, None, None
        self.pseudo_label_dict = None
        self.abbrev = None

    def get_abbrev(self):
        return self.abbrev

    def _build_pseudo_label_dict(self):
        label_dict = OrderedDict()
        for i, label in enumerate(self.y_hat):
            label = label.item()
            if label not in label_dict:
                label_dict[label] = [i]
            else:
                label_dict[label].append(i)

        # make sure there is no missing label
        for i in range(cfg.args.class_num):
            if i not in label_dict:
                label_dict[i] = []
        # for key in label_dict:
        #     cfg.logger.debug(f'{key}: {label_dict[key]}')
        # cfg.logger.debug(len(label_dict))
        return label_dict

    def update_predicted_probs(self, probs):
        self.probs = probs
        self.y_prob, self.y_hat = self.probs.max(1)
        if cfg.sampler_confidence_threshold is not None and cfg.sampler_confidence_threshold > 0:
            self.y_hat[self.y_prob < cfg.sampler_confidence_threshold] = -1
        self.pseudo_label_dict = self._build_pseudo_label_dict()

    def _remove_one_example_from_pseudo_label_dict(self, cls, data_index):
        if cls in self.pseudo_label_dict:
            self.pseudo_label_dict[cls].remove(data_index)
        else:
            raise ValueError(f'class {cls} not present in pseudo label dictionary')

    def remove_list_of_examples_from_pseudo_label_dict(self, cls, data_indices):
        for d in data_indices:
            self._remove_one_example_from_pseudo_label_dict(cls, d)

    def _sample_examples_by_class(self, cls):
        raise NotImplementedError

    def __iter__(self):
        assert self.pseudo_label_dict is not None, 'The sampler is not initialized!'
        for _ in range(self.length):
            batch = []
            classes = TaskSampler().sample_N_classes_as_a_task()
            for cls in classes:
                if cls not in self.pseudo_label_dict or not self.pseudo_label_dict[cls]:
                    continue
                samples_for_this_class = self._sample_examples_by_class(cls)
                batch.extend(samples_for_this_class)

            if len(batch) < (cfg.args.sampler_n_way * cfg.args.sampler_k_shot):
                random_samples = random.choices(
                    range(len(self.y_hat)),
                    k=((cfg.args.sampler_n_way * cfg.args.sampler_k_shot) - len(batch)),
                )
                batch.extend(random_samples)
            assert len(batch) % 2 == 0, f'Batch size should be even, but got {len(batch)}'
            assert len(batch) > 0
            yield batch

    def __len__(self):
        return self.length


class SelfTrainingVannilaSampler(SelfTrainingBaseSampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.abbrev = 'V'

    def _sample_examples_by_class(self, cls):
        if cfg.args.sampler_k_shot <= len(self.pseudo_label_dict[cls]):
            sampled_examples = random.sample(
                self.pseudo_label_dict[cls], k=cfg.args.sampler_k_shot)  # sample without replacement
        else:
            sampled_examples = random.choices(
                self.pseudo_label_dict[cls], k=cfg.args.sampler_k_shot)  # sample with replacement
        return sampled_examples


class SelfTrainingConfidentSampler(SelfTrainingBaseSampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.abbrev = 'C'

    def _sample_examples_by_class(self, cls):
        prob = self.probs[self.pseudo_label_dict[cls], cls]
        top_k_indices_for_this_cls = torch.topk(prob, cfg.args.sampler_k_shot)[1].cpu().numpy().tolist()
        top_k_indices = [self.pseudo_label_dict[cls][i] for i in top_k_indices_for_this_cls]
        return top_k_indices


class SelfTrainingUnConfidentSampler(SelfTrainingBaseSampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.abbrev = 'U'

    def _sample_examples_by_class(self, cls):
        prob = self.probs[self.pseudo_label_dict[cls], cls]
        top_k_indices_for_this_cls = torch.topk(1 - prob, cfg.args.sampler_k_shot)[1].cpu().numpy().tolist()
        top_k_indices = [self.pseudo_label_dict[cls][i] for i in top_k_indices_for_this_cls]
        return top_k_indices


class SelfTrainingMedianConfidentSampler(SelfTrainingBaseSampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.abbrev = 'M'

    def _sample_examples_by_class(self, cls):
        prob = self.probs[self.pseudo_label_dict[cls], cls]
        median_index_for_this_cls = [torch.median(prob, dim=0)[1].item()]
        median_index = [self.pseudo_label_dict[cls][i] for i in median_index_for_this_cls]
        return median_index


class SelfTrainingEpistemicSampler(SelfTrainingBaseSampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.abbrev = 'E'

    def _sample_examples_by_class(self, cls):
        prob = self.probs[self.pseudo_label_dict[cls]]
        entropies = ConditionalEntropy(prob)
        top_k_indices_for_this_cls = torch.topk(entropies, cfg.args.sampler_k_shot)[1].cpu().numpy().tolist()
        top_k_indices = [self.pseudo_label_dict[cls][i] for i in top_k_indices_for_this_cls]
        return top_k_indices


def get_sampler(dataset, is_source=True):
    if cfg.args.sampler_type is None:
        return None

    if is_source:
        cfg.logger.info('    Source sampler: loading labels...')
        if cfg.args.dataset == 'visda-2017':
            labels = torch.load('/data/VisDA-2017/train_labels.pt')
        else:
            labels = [0 for _ in tqdm(range(len(dataset)), ncols=80, leave=False)]
            for i, (_, y) in enumerate(tqdm(dataset, ncols=80, leave=False)):
                labels[i] = y
        cfg.source_sampler = N_Way_K_Shot_BatchSampler(labels)
        cfg.source_batch_count = len(cfg.source_sampler)
        return cfg.source_sampler

    if cfg.args.sampler_type in ['SelfTrainingVannilaSampler', 'V']:
        cfg.logger.info('    Target sampler: creating Vanilla smapler...')
        cfg.target_sampler = SelfTrainingVannilaSampler(dataset)
    else:
        raise NotImplementedError

    cfg.target_batch_count = len(cfg.target_sampler)

    return cfg.target_sampler
