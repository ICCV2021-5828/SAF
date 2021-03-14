
from torch import nn
import torch.nn.functional as F

from .. import cfg
from .MDD import MDDNet


class SAFnet(MDDNet):
    '''variation of the mixup'''
    def __init__(self, base_net, bottleneck_dim, width, class_num, amplify_coeff=2):
        super(SAFnet, self).__init__(base_net=base_net, use_bottleneck=True, bottleneck_dim=bottleneck_dim, width=width, class_num=class_num)
        self.feature_dim = self.base_network.output_num()
        self.class_num = class_num
        self.amplify_coeff = amplify_coeff
        self._build_mixup_layers()

    def _build_mixup_layers(self):
        self.fc_a = nn.Sequential(
            nn.Linear(self.feature_dim, 384),
            nn.ReLU(inplace=True),
        )
        self.fc_b = nn.Sequential(
            nn.Linear(self.feature_dim, 384),
            nn.ReLU(inplace=True),
        )

        self.fc_weight = nn.Sequential(
            nn.Linear(384, 1),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

        self.parameter_list += [
            {'params': self.fc_a.parameters(), 'lr_ratio': 1, 'name': 'fc_a'},
            {'params': self.fc_b.parameters(), 'lr_ratio': 1, 'name': 'fc_b'},
            {'params': self.fc_weight.parameters(), 'lr_ratio': 1, 'name': 'fc_weight'},
        ]

    def _extract_features(self, x):
        return self.base_network(x)

    def _classify_features(self, features):
        features = self.bottleneck_layer(features)
        return self.classifier_layer(features)

    def _classify_features_adv(self, features):
        features = self.bottleneck_layer(features)
        return self.classifier_layer_2(features)

    def SAF_forward(self, **kwargs):
        tx = kwargs['tx']
        tbs = tx.shape[0]
        assert tbs % 2 == 0, 'Batch size must be even!'

        tx = self._extract_features(tx)
        naive_tx = self._classify_features(tx)
        naive_ty = naive_tx.detach().max(dim=1)[1]

        tx_1 = tx
        ty_1 = naive_ty
        ty_1 = F.one_hot(ty_1, num_classes=self.class_num)

        tx_2 = tx.flip(dims=(0,))
        ty_2 = naive_ty.flip(dims=(0,))
        ty_2 = F.one_hot(ty_2, num_classes=self.class_num)

        tx_a = self.fc_a(tx_1)
        tx_b = self.fc_b(tx_2)
        tx_w = self.fc_weight(tx_a + tx_b)

        tx = tx_w * tx_1 + (1 - tx_w) * tx_2
        tx = self._classify_features(tx)
        ty = tx_w * ty_1 + (1 - tx_w) * ty_2
        ty = ty * self.amplify_coeff

        return naive_tx, naive_ty, tx, ty

