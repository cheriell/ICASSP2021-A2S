
import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class ConvBlock(nn.Module):
    """Convolutional block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1),
                              bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=(1, 1),
                              bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x



## Model initialisation functions
## Code borrowed from https://github.com/bytedance/piano_transcription/blob/master/pytorch/models.py .

def init_layer(layer):
    """Initialise a linear or convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialise a batch norm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def init_gru(rnn):
    """Initialize a GRU layer. """
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)
