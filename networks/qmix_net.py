import torch.nn as nn
import torch
import torch.nn.functional as F


class Qmix_Net(nn.Module):
    def __init__(self, args):
        super(Qmix_Net, self).__init__()

        self.args = args            

        # args.state_shape
        # args.hyper_hidden_dim


