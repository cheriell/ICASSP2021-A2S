import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio2score.scores.symbols import SOS, EOS


class ConvStack(nn.Module):
    """Convolutional stack.

        Args:
            in_channels: number of channels in the input feature.
            freq_bins: number of frequency bins in the input feature.
            output_size: output size of the convolutional stack.
        """

    def __init__(self, in_channels, freq_bins, output_size):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=20, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1), 
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=20, 
                               out_channels=20, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1), 
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=20, 
                               out_channels=40, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1), 
                               bias=False)
        self.conv4 = nn.Conv2d(in_channels=40, 
                               out_channels=40, 
                               kernel_size=(3, 3), 
                               stride=(1, 1), 
                               padding=(1, 1), 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(40)
        self.bn4 = nn.BatchNorm2d(40)
        
        self.out = nn.Linear(freq_bins*40, output_size, bias=False)
        self.out_bn = nn.BatchNorm1d(output_size)
        
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        
        init_layer(self.out)
        init_bn(self.out_bn)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        
        x1 = F.avg_pool2d(x, kernel_size=(1, 2))
        x2 = F.max_pool2d(x, kernel_size=(1, 2))
        x = x1 + x2
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
                    # [batch_size, 40, freq_bins, spectrogram_max_length/2]
        
        x = x.transpose(1, 3).flatten(2)
                    # [batch_size, spectrogram_max_length/2, freq_bins*40]
        x = F.relu(self.out_bn(self.out(x).transpose(1, 2)).transpose(1, 2))
                    # [batch_size, spectrogram_max_length/2, output_size]
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        
        return x
        
        
class Seq2Seq(nn.Module):
    """Sequence-to-sequence architecture.

        Args:
            input_size: encoder input feature size
            hidden_size: hidden size of the rnn layers
            encoder_max_length: maximum length of the input sequence
            vocab_sizes: vocabulary sizes of each row of the output representation.
        """

    def __init__(self, input_size, hidden_size, encoder_max_length, vocab_sizes):
        super().__init__()
        
        self.n_out = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        self.encoder_max_length = encoder_max_length
        
        self.encoder = EncoderRNN(input_size=input_size, hidden_size=hidden_size)
        self.decoder = DecoderRNN(hidden_size=hidden_size, encoder_max_length=encoder_max_length, vocab_sizes=vocab_sizes)
        
        
    def forward(self, inputs, target_length, batch_size, device, target=None, teacher_forcing_ratio=0):
        # inputs: [batch_size, encoder_max_length, conv_stack_output_size]
        # target: None or [batch_size, n_out, target_length]
        
        encoder_outputs, hidden = self.encoder(inputs)  # [batch_size, encoder_max_length, hidden_size*2], [2, batch_size, hidden_size]
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        decoder_input = torch.ones(batch_size, self.n_out, 1, device=device) * SOS
        all_probs = [torch.zeros(batch_size, target_length, self.vocab_sizes[i], device=device) for i in range(self.n_out)]
        all_attn_weights = torch.zeros(batch_size, target_length, self.encoder_max_length, device=device)
        
        EOS_labels = torch.zeros(batch_size, device=device)
        
        for di in range(target_length):
            if EOS_labels.sum() == batch_size:
                break
            probs, hidden, attn_weights = self.decoder(decoder_input.long(), hidden, encoder_outputs)
            # probs: List[[batch_size, 1, vocab_sizes[i]]]
            # hidden: [2, batch_size, hidden_size]
            # attn_weights: [batch_size, 1, encoder_max_length]
            
            for i in range(self.n_out):
                all_probs[i][:,di,:] = probs[i][:,0,:]
            all_attn_weights[:,di,:] = attn_weights[:,0,:]
            
            if use_teacher_forcing:
                decoder_input[:,:,0] = target[:,:,di]
            else:
                preds = [probs[i].topk(1)[1] for i in range(self.n_out)]  # List[[batch_size, 1, 1]]
                decoder_input = torch.cat(preds, dim=1)  # [batch_size, n_out, 1]
            # check EOS and add label
            for batch_index in range(batch_size):
                if (decoder_input[batch_index] == EOS).nonzero().sum() > 0:
                    EOS_labels[batch_index] = 1
        
        return all_probs, all_attn_weights
        

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.init_weight()
        
    def init_weight(self):
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, x):
        # x: [batch_size, max_length, input_size]
        x, hidden = self.gru(x)  # x: [batch_size, max_length, hidden_size*2], hidden: [4, batch_size, hidden_size]
        hidden1 = F.tanh(self.fc(torch.cat((hidden[0], hidden[1]), dim=1)))  # [batch_size, hidden_size]
        hidden2 = F.tanh(self.fc(torch.cat((hidden[2], hidden[3]), dim=1)))  # [batch_size, hidden_size]
        hidden = torch.cat((hidden1.unsqueeze(0), hidden2.unsqueeze(0)), dim=0)  # [2, batch_size, hidden_size]
        return x, hidden


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.attn = nn.Linear(hidden_size*4, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.attn)
        init_layer(self.v)
        
    def forward(self, hidden, encoder_output):
        # hidden: [2, batch_size, hidden_size], encoder_output: [batch_size, max_length, hidden_size*2]
        max_length = encoder_output.shape[1]
        
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)  # [batch_size, hidden_size*2]
        hidden = hidden.unsqueeze(1).repeat(1, max_length, 1)  # [batch_size, max_length, hidden_size*2]
        
        energy = F.tanh(self.attn(torch.cat((hidden, encoder_output), dim=2)))  # [batch_size, max_length, hidden_size]
        attention = self.v(energy).squeeze(2)  # [batch_size, max_length]
        
        return F.softmax(attention, dim=1)
    
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, encoder_max_length, vocab_sizes):
        super().__init__()
        
        self.n_out = len(vocab_sizes)
        embedding_sizes = [int(s ** 0.7) for s in vocab_sizes]
        self.embedding_size = np.sum(embedding_sizes)
        
        self.embedding_layers = nn.ModuleList([nn.Embedding(vocab_sizes[i], embedding_sizes[i]) for i in range(self.n_out)])
        self.attention_layer = AttentionLayer(hidden_size)
        self.gru = nn.GRU(input_size=self.embedding_size+hidden_size*2, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.out = nn.ModuleList([nn.Linear(hidden_size*3, vocab_sizes[i]) for i in range(self.n_out)])
        
        self.init_weight()
        
    def init_weight(self):
        init_gru(self.gru)
        for i in range(self.n_out):
            init_layer(self.out[i])
            
    def forward(self, decoder_input, hidden, encoder_outputs):
        # decoder_input: [batch_size, n_out, 1], hidden: [2, batch_size, hidden_size], encoder_outputs: [batch_size, encoder_max_length, hidden_size*2]
        embeddings = [self.embedding_layers[i](decoder_input[:,i,:]) for i in range(self.n_out)]  # List[[batch_size, 1, embedding_sizes[i]]]
        embedded = torch.cat(embeddings, dim=2)  # [batch_size, 1, embedding_size]
        embedded = F.dropout(embedded, p=0.1, training=self.training, inplace=False)
        
        attn_weights = self.attention_layer(hidden, encoder_outputs).unsqueeze(1)  # [batch_size, 1, encoder_max_length]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_size*2]
        rnn_input = torch.cat([embedded, context], dim=2)  # [batch_size, 1, embedding_size + hidden_size*2]
        
        output, hidden = self.gru(rnn_input, hidden)  # [batch_size, 1, hidden_size], [2, batch_size, hidden_size]
        output_combine = torch.cat([output, context], dim=2)  # [batch_size, 1, hidden_size*3]
        
        probs = [self.out[i](output_combine) for i in range(self.n_out)]   # List[[batch_size, 1, vocab_sizes[i]]]
        probs = [F.log_softmax(p, dim=2) for p in probs]  # List[[batch_size, 1, vocab_sizes[i]]]
        
        return probs, hidden, attn_weights
        

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
