
# Standard Library Modules
import os
from time import time

# Third-Party Library Modules
import torch
import torch.optim as optim
from tqdm import tqdm

# Local Modules
from . import cfg
from .data_utils import loadImageTensor
from .model import get_model, GradientReversalLayer
from .utils import *


# register args
#  the reference value of args is passed to cfg.args.
#  any update to args will also affect cfg.args
args = cfg.args


# dataset
cfg.logger.info('Loading datasets...')
# create data loader
if args.dataset == 'visda-2017':
    source_image_list = f'./image_list/visda-2017-train.txt'
    target_image_list = f'./image_list/visda-2017-validation.txt'
    train_source_loader = loadImageTensor(source_image_list, args.batch_size, is_source=True, image_list=True, drop_last=True, num_workers=args.num_workers)
    train_target_loader = loadImageTensor(target_image_list, args.batch_size, is_source=False, image_list=True, drop_last=True, num_workers=args.num_workers)
    test_target_loader = loadImageTensor(target_image_list, args.eval_batch_size, image_list=True, is_train=False, num_workers=args.num_workers*2)
else:
    raise ValueError

assert len(test_target_loader.dataset) == len(train_target_loader.dataset)
args.source_batch_count = len(train_source_loader)
args.target_batch_count = len(train_target_loader)
if args.sampler_type:
    args.iter_per_epoch = args.sampler_update_frequency
elif args.iter_per_epoch == 0:
    args.iter_per_epoch = min(len(train_source_loader), len(train_target_loader))

args.test_batch_count = len(test_target_loader)
args.source_dataset_size = len(train_source_loader.dataset)
args.target_dataset_size = len(train_target_loader.dataset)


# model
cfg.logger.info('Creating model...')
model = get_model(args)

model = model.to(cfg.DEVICE)
group_ratios = model.get_parameter_ratio_dict()

# optimizer
cfg.logger.info('Creating optimizer...')
optimizer = optim.SGD(
    model.parameter_list,
    lr=args.lr,
    momentum=args.SGD_momentum,
    weight_decay=args.SGD_weight_decay,
    nesterov=args.SGD_nesterov,
)

# scheduler
cfg.logger.info('Creating scheduler...')
lr_scheduler = InverseDecayScheduler(
    group_ratios,
    gamma=args.INV_gamma,
    decay_rate=args.INV_decay_rate,
    init_lr=args.lr
)


cfg.logger.info('Finishing warmup...')


def train_iteration(inputs_source, labels_source, inputs_target):
    # stats
    cfg.weight_list = []
    cfg.weight_list.append(f'LR={lr_scheduler.get_lr(cfg.iter_count):.6f}') 
    cfg.weight_list.append(f'GRL={GradientReversalLayer.get_coeff():.6f}') 
    cfg.loss_list = []

    model.train()
    if 'MDD' in args.operation_flags:
        train_iteration_MDD(model, optimizer, inputs_source, inputs_target, labels_source)
    else:
        _, source_logits, _, _ = model(inputs_source)
        classifier_loss = cfg.class_criterion(source_logits, labels_source)
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

    # mixup
    if 'MIX' in args.operation_flags:
        train_iteration_MIX(model, optimizer, inputs_target)

    cfg.logger.debug(
        f'  Train Iteration [{cfg.iter_count}/{args.total_iters}] '
        f'{", ".join(cfg.weight_list)};  {", ".join(cfg.loss_list)}'
    )
    return


def train_epoch():
    cfg.epoch_count += 1
 
    cfg.logger.info('-' * cfg.DASHLINE_LENGTH)

    start_time = time()

    train_source_iterator = iter(train_source_loader)
    train_target_iterator = iter(train_target_loader)
    for _ in tqdm(
        range(args.iter_per_epoch),
        # total=args.iter_per_epoch,
        desc=f'Train epoch ({cfg.epoch_count})',
        ncols=80, leave=False
    ):
        cfg.iter_count += 1

        try:
            inputs_source, labels_source = next(train_source_iterator)
            inputs_target, _ = next(train_target_iterator)
        except StopIteration:
            train_source_iterator = iter(train_source_loader)
            train_target_iterator = iter(train_target_loader)
            inputs_source, labels_source = next(train_source_iterator)
            inputs_target, _ = next(train_target_iterator)

        lr_scheduler.update_optimizer(
            optimizer,
            cfg.iter_count,
        )

        # if use sampler, the input shape would be [1, bs, ...],
        #  so we squeeze the extra dimension here
        # if not, the squeeze command does not change the input
        inputs_source = inputs_source.squeeze().to(cfg.DEVICE)
        inputs_target = inputs_target.squeeze().to(cfg.DEVICE)
        labels_source = labels_source.squeeze().to(cfg.DEVICE)

        train_iteration(inputs_source, labels_source, inputs_target)

    train_time = time() - start_time
    cfg.logger.info(
        f'Train Epoch ({cfg.epoch_count}) '
        f'Iteration [{cfg.iter_count}/{args.total_iters}] ends. '
        f'Time elapsed: {train_time:.2f}s'
    )

    evaluate()
    return


def evaluate():
    model.eval()
    all_probs = None
    all_labels = None
    start_time = time()

    for (inputs, labels) in tqdm(
        test_target_loader, desc=f'Evaluation epoch ({cfg.epoch_count})',
        ncols=80, leave=False
    ):
        inputs = inputs.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)
        probilities = model.predict(inputs)

        probilities = probilities.data.float()
        labels = labels.data.float()
        if all_probs is None:
            all_probs = probilities
            all_labels = labels
        else:
            all_probs = torch.cat((all_probs, probilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    n_correct = torch.sum(torch.squeeze(predict).float() == all_labels)
    n_total = all_labels.size()[0]
    accuracy = float(n_correct) / n_total

    eval_time = time() - start_time

    # update sampler pseudo-label
    if cfg.target_sampler:
        cfg.target_sampler.update_predicted_probs(all_probs)

    if accuracy > cfg.max_accuracy:
        # new highest accuracy
        cfg.logger.warning(
            f'  New Highest Accuracy: {accuracy*100:.4f}% !!!!'
        )
        cfg.max_accuracy = accuracy
        cfg.max_accuracy_epoch = cfg.epoch_count
        save_state(info_list=[f'acc{accuracy*100:.2f}'])

    cfg.logger.info(
        f'Evaluation Epoch ({cfg.epoch_count}) '
        f'Iteration [{cfg.iter_count}/{args.total_iters}] ends. '
        f'Time elapsed: {eval_time:.2f}s, '
        f'Eval Acc: {accuracy*100:.4f}% ({n_correct}/{n_total}); '
        f'Highest Acc: {cfg.max_accuracy*100:.4f}% @ Epoch ({cfg.max_accuracy_epoch}) '
        f'Iteration [{cfg.max_accuracy_epoch * args.iter_per_epoch}]'
    )

    if (
        (cfg.iter_count >= args.total_iters and cfg.epoch_count >= args.total_epochs) or \
        cfg.epoch_count >= (cfg.max_accuracy_epoch + cfg.COOL_DOWN_INTERVAL // args.iter_per_epoch)
    ):
        # save and quit if:
        #  1. have trained enough iterations and epochs
        #  2. enough epochs have passed since the last occurence of highest accuracy
        cfg.logger.warning(
            '>>>>>>>>>>>>>>>>>>>> '
            'Quitting... '
            f'Highest accuracy: {cfg.max_accuracy:.6f} @ Epoch ({cfg.max_accuracy_epoch}) Iteration [{cfg.max_accuracy_epoch * args.iter_per_epoch}].'
        )
        save_state(info_list=['fin'])
        with open(args.chart_path, 'a') as chart:
            info_strings = []
            for arg in vars(cfg.args):
                info_strings.append(f'{arg}={str(getattr(args, arg))}')
            info_strings.append(f'accuracy={cfg.max_accuracy}')
            chart.write(','.join(info_strings))
            chart.write('\n')
        exit(0)

    return


def save_state(info_list=[], subdir=''):
    if cfg.epoch_count < 15 and cfg.iter_count < 3000:
        return
    save_dir = os.path.join(args.checkpoint_root, args.network_alias, args.dataset_alias, subdir)
    os.system(f'mkdir -p {save_dir}')
    current_state_dict = {
        'args': args,
        'stats': (cfg.epoch_count, cfg.iter_count, cfg.eval_accuracies, cfg.max_accuracy, cfg.max_accuracy_epoch),
        'model_dict': model.state_dict(),
    }
    os.system(f'mkdir -p {args.checkpoint_root}')
    extra_info = ''
    for info_string in info_list:
        extra_info += f'{info_string}.'
    checkpoint_path = os.path.join(
        save_dir,
        f'{args.train_session_name}-ep{cfg.epoch_count:0>3}.{extra_info}ckpt'
    )
    torch.save(current_state_dict, checkpoint_path)

    cfg.logger.info(f'  Saved the state of epoch {cfg.epoch_count} to: {checkpoint_path}')


def main():
    # print arguments
    cfg.logger.info('Arguments:')
    for arg in vars(cfg.args):
        cfg.logger.info(f'  {arg}: {str(getattr(args, arg))}')

    cfg.logger.info('-' * cfg.DASHLINE_LENGTH)
    cfg.logger.info('Begin training...')
    evaluate()
    while True:
        train_epoch()

