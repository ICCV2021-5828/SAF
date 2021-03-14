
from torch import nn

from .backbone import network_dict
from .GRL import GradientReversalLayer


class MDDNet(nn.Module):
    def __init__(
        self,
        base_net='ResNet50',
        use_bottleneck=True,
        bottleneck_dim=1024,
        width=1024,
        class_num=31
    ):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReversalLayer
        self.bottleneck_layer_list = [
            nn.Linear(self.base_network.output_num(), bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        ]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, class_num)
        ]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, class_num)
        ]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self._initialization()

        ## collect parameters
        self.parameter_list = [
            {'params': self.base_network.parameters(), 'lr_ratio': 0.1, 'name': 'base_network'},
            {'params': self.bottleneck_layer.parameters(), 'lr_ratio': 1, 'name': 'bottleneck_layer'},
            {'params': self.classifier_layer.parameters(), 'lr_ratio': 1, 'name': 'classifier_layer'},
            {'params': self.classifier_layer_2.parameters(), 'lr_ratio': 1, 'name': 'classifier_layer_2'}
        ]

    def _initialization(self):
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for layer in self.classifier_layer:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                # layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.0)
        for layer in self.classifier_layer_2:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                # layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.0)

    def get_parameter_ratio_dict(self):
        param_ratio_dict = {}
        for param in self.parameter_list:
            param_ratio_dict[param['name']] = param['lr_ratio']
        return param_ratio_dict

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)

        outputs = self.classifier_layer(features)

        features_adv = self.grl_layer.apply(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

    def predict(self, inputs):
        return self.forward(inputs)[2]

    def predict_with_feature_representation(self, inputs):
        fr, _, pred, _ = self.forward(inputs)
        return fr, pred
