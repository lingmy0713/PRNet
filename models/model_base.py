import torch
import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()

        self.args = args

        self.device = torch.device('cpu' if args.b_cpu else 'cuda')
        self.n_gpu = args.n_gpu
        self.b_save_all_models = args.b_save_all_models

    def forward(self, x):
        pass

