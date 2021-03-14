
import argparse
from datetime import datetime
import logging
import os
# import sys
from time import time
# from types import SimpleNamespace

import torch
from torch import nn

# argparser
parser = argparse.ArgumentParser()
parser.add_argument('--makefile_arg', default=None, type=str, help='input Makefile argument')

parser.add_argument('--network', default='MDD', type=str, help='model architecture')
parser.add_argument('--backbone', default='ResNet50', type=str, help='model backbone')
parser.add_argument('--device', default='cuda', type=str, help='device for training')
parser.add_argument('--cuda_visible_devices', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--random_seed', default=None, type=int, help='DO NOT CHANGE')

parser.add_argument('--total_epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--total_iters', default=50000, type=int, help='number of total iterations to run')

parser.add_argument('--lr', default=0.004, type=float, help='initial learning rate')
parser.add_argument('--SGD_momentum', default=0.9, type=float, help='SGD momentum')
parser.add_argument('--SGD_weight_decay', default=5e-4, type=float, help='SGD weight decay')
parser.add_argument('--SGD_nesterov', default=True, type=bool, help='SGD Nesterov weight')

parser.add_argument('--INV_gamma', default=0.001, type=float, help='INV scheduler gamma')
parser.add_argument('--INV_decay_rate', default=0.75, type=float, help='INV scheduler decay rate')

parser.add_argument('--conditional_entropy', action='store_true', help='whether use conditional entropy')
parser.add_argument('--ce_temperature', default=1.0, type=float, help='temperature for conditional entropy')
parser.add_argument('--naive_entropy', action='store_true', help='whether use conditional entropy')

parser.add_argument('--GRL_alpha', default=2.0, type=float, help='GRL alpha')
parser.add_argument('--GRL_beta', default=0, type=int, help='GRL beta')
parser.add_argument('--GRL_low', default=0.0, type=float, help='GRL low_value')
parser.add_argument('--GRL_high', default=0.1, type=float, help='GRL high_value')
parser.add_argument('--GRL_max', default=1000, type=int, help='GRL max_iter')

parser.add_argument('--MIXUP_low', default=0.0, type=float, help='Mixup naive low_value')
parser.add_argument('--MIXUP_high', default=0.1, type=float, help='Mixup naive high_value')
parser.add_argument('--MIXUP_max', default=1000, type=int, help='Mixup naive max_iter')
parser.add_argument('--MIXUP_NAIVE_low', default=0.3, type=float, help='Mixup naive low_value')
parser.add_argument('--MIXUP_NAIVE_high', default=7.0, type=float, help='Mixup naive high_value')
parser.add_argument('--MIXUP_NAIVE_max', default=10000, type=int, help='Mixup naive max_iter')

parser.add_argument('--dataset', default='office-31', type=str, help='which dataset')
parser.add_argument('--source', default='amazon', type=str, help='source domain')
parser.add_argument('--target', default='dslr', type=str, help='target domain')
parser.add_argument('--batch_size', default=32, type=int, help='batch size for both source and target')
parser.add_argument('--eval_batch_size', default=128, type=int, help='batch size for target in evaluation')
parser.add_argument('--resize_size', default=256, type=int, help='resize size for raw input image')
parser.add_argument('--crop_size', default=224, type=int, help='crop size for input tensor')
parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--crop_type', default='RandomResizedCrop', type=str, choices=['RandomResizedCrop', 'RandomCrop'], help='crop type for training dataloader')

parser.add_argument('--class_num', default=0, type=int, help='DO NOT CHANGE')
parser.add_argument('--width', default=0, type=int, help='DO NOT CHANGE')
parser.add_argument('--srcweight', default=0, type=int, help='DO NOT CHANGE')
parser.add_argument('--is_cen', default=False, type=bool, help='DO NOT CHANGE')

parser.add_argument('--source_batch_count', default=0, type=int, help='DO NOT CHANGE')
parser.add_argument('--target_batch_count', default=0, type=int, help='DO NOT CHANGE')
parser.add_argument('--iter_per_epoch', default=0, type=int, help='iteration per epoch. default=min(len(src.sz), len(tgt.sz))')
parser.add_argument('--test_batch_count', default=0, type=int, help='DO NOT CHANGE')
parser.add_argument('--source_dataset_size', default=0, type=int, help='DO NOT CHANGE')
parser.add_argument('--target_dataset_size', default=0, type=int, help='DO NOT CHANGE')

parser.add_argument('--sampler_type', default=None, type=str, help='sampler type')
parser.add_argument('--sampler_n_way', '--n_way', default=None, type=int, help='sampler: label selection number')
parser.add_argument('--sampler_k_shot', '--k_shot', default=None, type=int, help='sampler: samples for each selected label')
parser.add_argument('--sampler_update_frequency', default=None, type=int, help='sampler: pseudo-label update frequency')
parser.add_argument('--sampler_confidence_threshold', default=None, type=int, help='sampler: confidence lower-bound for psudo-label activation')
parser.add_argument('--MDD_mask_all', action='store_true', help='MDD: zero-mask logits of non-selected labels in both output and output_adv while training')
parser.add_argument('--MDD_mask_classifier', action='store_true', help='MDD: zero-mask logits of non-selected labels in output while training')
parser.add_argument('--MDD_mask_divergence', action='store_true', help='MDD: zero-mask logits of non-selected labels in output_adv while training')
parser.add_argument('--SAF_entropy_lower_bound', default=None, type=float, help='conditional entropy lower bound for selection of SAF mixup samples')
parser.add_argument('--SAF_entropy_upper_bound', default=None, type=float, help='conditional entropy upper bound for selection of SAF mixup samples')

parser.add_argument('--checkpoint_root', default='./checkpoint/', type=str, help='path to save checkpoint')
parser.add_argument('--log_root', default='./log/', type=str, help='path to log files')
parser.add_argument('--chart_path', default='./log/accuracy_chart.csv', type=str, help='path to chart')

parser.add_argument('--network_alias', default='', type=str, help='DO NOT CHANGE')
parser.add_argument('--dataset_alias', default='', type=str, help='DO NOT CHANGE')
parser.add_argument('--timestamp', default=0, type=int, help='DO NOT CHANGE')
parser.add_argument('--train_session_name', default='', type=str, help='DO NOT CHANGE')
args = parser.parse_args()


# global statistics
epoch_count = 0
iter_count = 0
eval_accuracies = []
max_accuracy = 0.
max_accuracy_epoch = -1


# constants
DEVICE = args.device
SAF_ENTROPY_MIN = args.SAF_entropy_lower_bound
SAF_ENTROPY_MAX = args.SAF_entropy_upper_bound
DASHLINE_LENGTH = 60
COOL_DOWN_INTERVAL = 1000000
SRC_DOM_LABEL = 0
TGT_DOM_LABEL = 1


# environmental settings
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
if args.random_seed is None:
    args.random_seed = torch.random.initial_seed()
else:
    torch.manual_seed(args.random_seed)


# dataset parameters
args.dataset = args.dataset.lower()
if args.dataset in ['visda-2017', 'v17']:
    args.dataset = 'visda-2017'
    args.source = 'train'
    args.target = 'validation'
    args.dataset_alias = 'VDA17'
    args.class_num = 12
    args.width = 2048
    args.srcweight = 2
    args.is_cen = False
    args.sampler_n_way = 12
    args.sampler_k_shot = 4
    args.sampler_update_frequency = 500
else:
    raise ValueError

args.crop_type = 'CenterCrop' if args.is_cen else args.crop_type


# sampler parameters
SAMPLING_BATCH_COUNT = args.sampler_update_frequency
SAMPLING_BATCH_SIZE = args.sampler_n_way * args.sampler_k_shot
SAMPLER_DICT = {
    'SelfTrainingVannilaSampler'         : 'V', 'V': 'V',
    'SelfTrainingConfidentSampler'       : 'C', 'C': 'C',
    'SelfTrainingUnConfidentSampler'     : 'U', 'U': 'U',
    'SelfTrainingMedianConfidentSampler' : 'M', 'M': 'M',
    'SelfTrainingEpistemicSampler'       : 'E', 'E': 'E',
}

if args.sampler_type is None:
    sampler_alias = None
elif args.sampler_type in SAMPLER_DICT:
    sampler_alias = SAMPLER_DICT[args.sampler_type]
else:
    raise ValueError


# operations in training
args.operation_flags = set()
class_criterion = nn.CrossEntropyLoss().to(DEVICE)


# session name
args.timestamp = time()
args.network_alias = f'{args.network}'
if sampler_alias:
    args.network_alias += f'_{sampler_alias}{args.sampler_n_way}w{args.sampler_k_shot}s'
args.train_session_name = f'{args.network_alias}-{args.dataset_alias}-{int(args.timestamp)}'


# logger
os.system(f'mkdir -p {args.log_root}')
log_path = os.path.join(
    args.log_root,
    f'{args.train_session_name}.log'
)
print(log_path)
logger = logging.getLogger('TrainLogger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)3d] %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler(log_path, encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info(f'Timestamp: {datetime.fromtimestamp(args.timestamp).strftime("%Y-%m-%d  %H:%M:%S:%f")}')
logger.info(f'Timestamp: {datetime.fromtimestamp(args.timestamp).strftime("%A, %B %-d, %Y   %-I:%M:%S %p")}')


# sampler
source_sampler = None
target_sampler = None
sampler_confidence_threshold = None
