import torch.nn as nn
import torch.nn.functional as F
from utils.model_config import CONFIGS_

import collections

#################################
##### Neural Network model #####
#################################
class Net(nn.Module):
    def __init__(self, dataset='mnist', model='cnn'):
        super(Net, self).__init__()
        # define network layers
        print("Creating model for {}".format(dataset))
        self.dataset = dataset
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim=CONFIGS_[dataset]
        print('Network configs:', configs)
        self.named_layers, self.layers, self.layer_names =self.build_network(
            configs, input_channel, self.output_dim)
        self.n_parameters = len(list(self.parameters()))
        self.n_share_parameters = len(self.get_encoder())

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def build_network(self, configs, input_channel, output_dim):
        layers = nn.ModuleList()
        named_layers = {}
        layer_names = []
        kernel_size, stride, padding = 3, 2, 1
        for i, x in enumerate(configs):
            if x == 'F':
                layer_name='flatten{}'.format(i)
                layer=nn.Flatten(1)
                layers+=[layer]
                layer_names+=[layer_name]
            elif x == 'M':
                pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
                layer_name = 'pool{}'.format(i)
                layers += [pool_layer]
                layer_names += [layer_name]
            else:
                cnn_name = 'encode_cnn{}'.format(i)
                cnn_layer = nn.Conv2d(input_channel, x, stride=stride, kernel_size=kernel_size, padding=padding)
                named_layers[cnn_name] = [cnn_layer.weight, cnn_layer.bias]

                bn_name = 'encode_batchnorm{}'.format(i)
                bn_layer = nn.BatchNorm2d(x)
                named_layers[bn_name] = [bn_layer.weight, bn_layer.bias]

                relu_name = 'relu{}'.format(i)
                relu_layer = nn.ReLU(inplace=True)# no parameters to learn

                layers += [cnn_layer, bn_layer, relu_layer]
                layer_names += [cnn_name, bn_name, relu_name]
                input_channel = x

        # finally, classification layer
        fc_layer_name1 = 'encode_fc1'
        fc_layer1 = nn.Linear(self.hidden_dim, self.latent_dim)
        layers += [fc_layer1]
        layer_names += [fc_layer_name1]
        named_layers[fc_layer_name1] = [fc_layer1.weight, fc_layer1.bias]

        fc_layer_name = 'decode_fc2'
        fc_layer = nn.Linear(self.latent_dim, self.output_dim)
        layers += [fc_layer]
        layer_names += [fc_layer_name]
        named_layers[fc_layer_name] = [fc_layer.weight, fc_layer.bias]
        return named_layers, layers, layer_names


    def get_parameters_by_keyword(self, keyword='encode'):
        params=[]
        for name, layer in zip(self.layer_names, self.layers):
            if keyword in name:
                #layer = self.layers[name]
                params += [layer.weight, layer.bias]
        return params

    def get_encoder(self):
        return self.get_parameters_by_keyword("encode")

    def get_decoder(self):
        return self.get_parameters_by_keyword("decode")

    def get_shared_parameters(self, detach=False):
        return self.get_parameters_by_keyword("decode_fc2")

    def get_learnable_params(self):
        return self.get_encoder() + self.get_decoder()

    def forward(self, x, start_layer_idx = 0, logit=False):
        """
        :param x:
        :param logit: return logit vector before the last softmax layer
        :param start_layer_idx: if 0, conduct normal forward; otherwise, forward from the last few layers (see mapping function)
        :return:
        """
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
        restults={}
        z = x
        for idx in range(start_layer_idx, len(self.layers)):
            layer_name = self.layer_names[idx]
            layer = self.layers[idx]
            z = layer(z)

        if self.output_dim > 1:
            restults['output'] = F.log_softmax(z, dim=1)
        else:
            restults['output'] = z
        if logit:
            restults['logit']=z
        return restults

    def mapping(self, z_input, start_layer_idx=-1, logit=True):
        z = z_input
        n_layers = len(self.layers)
        for layer_idx in range(n_layers + start_layer_idx, n_layers):
            layer = self.layers[layer_idx]
            z = layer(z)
        if self.output_dim > 1:
            out=F.log_softmax(z, dim=1)
        result = {'output': out}
        if logit:
            result['logit'] = z
        return result
