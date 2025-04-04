import os
import torch
from .modeling.architectures.base_model import BaseModel

class BaseOCRV20:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()


    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)

    def read_pytorch_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError('{} is not existed.'.format(weights_path))
        weights = torch.load(weights_path)
        return weights

    def get_out_channels(self, weights):
        if list(weights.keys())[-1].endswith('.weight') and len(list(weights.values())[-1].shape) == 2:
            out_channels = list(weights.values())[-1].numpy().shape[1]
        else:
            out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        # print('weights is loaded.')

    def load_pytorch_weights(self, weights_path):
        self.net.load_state_dict(torch.load(weights_path, weights_only=True))
        # print('model is loaded: {}'.format(weights_path))

    def inference(self, inputs):
        with torch.no_grad():
            infer = self.net(inputs)
        return infer
