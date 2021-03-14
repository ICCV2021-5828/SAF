
from .GRL import GradientReversalLayer
from .MDD import MDDNet
from .SAF import SAFnet


def get_model(args):
    if args.network == 'MDD':
        model = MDDNet(
            base_net=args.backbone,
            bottleneck_dim=args.width,
            width=args.width,
            class_num=args.class_num,
        )
        args.operation_flags.add('MDD')
    elif args.network == 'SAF':
        model = SAFnet(
            base_net=args.backbone,
            bottleneck_dim=args.width,
            width=args.width,
            class_num=args.class_num,
        )
        args.operation_flags.add('MDD')
        args.operation_flags.add('MIX')
    else:
        raise ValueError

    return model
