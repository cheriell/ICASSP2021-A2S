

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from audio2score.settings import Constants, TrainingParam
from audio2score.models.basemodel import BaseModel
from audio2score.models.containers import ConvBlock
from audio2score.models.containers import init_layer, init_bn, init_gru
from audio2score.utilities.evaluation_utils import Eval


class PianorollTranscriptionModel(BaseModel):
    """Initialise model.

        Args:
            in_channels: number of channels of the input feature.
            freq_bins: number of frequency bins of the input feature.
        """

    def __init__(self, in_channels: int, freq_bins: int):
        super(PianorollTranscriptionModel, self).__init__()

        # convolutional blocks
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=20)
        self.conv_block2 = ConvBlock(in_channels=20, out_channels=40)
        
        # flatten convolutional layer output and feed output to a linear layer, followed by a batch normalisation
        self.fc3 = nn.Linear(in_features=freq_bins*40, out_features=200, bias=False)
        self.bn3 = nn.BatchNorm1d(200)
        
        # 2 bi-GRU layers followed by a time-distributed dense output layer
        self.gru = nn.GRU(input_size=200, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(200, 88, bias=True)
        
        # initialise model weights and loss function
        self.init_weight()
        self.init_loss_function()
        
    def init_weight(self):
        """Initialise model weights"""
        init_layer(self.fc3)
        init_bn(self.bn3)
        init_gru(self.gru)
        init_layer(self.fc)

    def init_loss_function(self):
        """Initialise loss function"""
        self.loss_fn = nn.MSELoss()

    def forward(self, spectrogram: torch.Tensor):
        """Get model output.

        Parameters:
            spectrogram: (dtype: float, shape: (batch_size, in_channels, freq_bins, input_length)), audio spectrogram as input.
        Returns:
            pianoroll: (torch.Tensor, dtype: float, shape: (batch_size, 88, input_length)), pianoroll output
        """
        conv_hidden = self.conv_block1(spectrogram)  # [batch_size, 20, freq_bins, input_length]
        conv_hidden = F.dropout(conv_hidden, p=0.2, training=self.training, inplace=True)  # same as above
        conv_output = self.conv_block2(conv_hidden)  # [batch_size, 40, freq_bins, input_length]
        conv_output = F.dropout(conv_output, p=0.2, training=self.training, inplace=True)  # same as above
        
        conv_output = conv_output.transpose(1, 3).flatten(2)  # [batch_size, input_length, freq_bins*40]
        linear_output = F.relu(self.bn3(self.fc3(conv_output).transpose(1, 2)).transpose(1, 2))  # [batch_size, input_length, 200]
        linear_output = F.dropout(linear_output, p=0.5, training=self.training, inplace=True)  # same as above
        
        rnn_output, _ = self.gru(linear_output)  # [batch_size, input_length, 200]
        rnn_output = F.dropout(rnn_output, p=0.5, training=self.training, inplace=False)  # same as above
        pianoroll = F.elu(self.fc(rnn_output).transpose(1, 2))  # [batch_size, 88, input_length]
        
        return pianoroll

    def prepare_batch_data(self, batch):
        spectrogram, pianoroll, pianoroll_mask = batch
        spectrogram = spectrogram.float()
        pianoroll_targ = pianoroll.float()
        pianoroll_mask = pianoroll_mask.float()
        return (spectrogram, pianoroll_mask), pianoroll_targ  # input_data, target_data

    def predict(self, input_data):
        spectrogram, pianoroll_mask = input_data
        pianoroll_pred = self(spectrogram) * pianoroll_mask  # add mask to ignore paddings
        return pianoroll_pred  # output_data
        
    def get_loss(self, output_data, target_data):
        loss = self.loss_fn(output_data, target_data)
        return loss

    def evaluate(self, output_data, target_data):
        pianoroll_pred = output_data
        pianoroll_targ = target_data
        # F-measure
        ps, rs, fs, accs = [], [], [], []
        results_n_on, results_n_onoff = [], []
        for i in range(TrainingParam.batch_size):
            pr_target = (pianoroll_targ[i,:,:] > Constants.velocity_threshold).bool()
            pr_output = (pianoroll_pred[i,:,:] > Constants.velocity_threshold).bool()
            
            # framewise evaluation
            p, r, f, acc = Eval.framewise_evaluation(pr_output, pr_target)
            ps, rs, fs, accs = ps + [p], rs + [r], fs + [f], accs + [acc]

            # notewise evaluation
            result_n_on, result_n_onoff = Eval.notewise_evaluation(pr_target, pr_output, hop_time=0.01)
            if result_n_on is not None:
                results_n_on.append(result_n_on)
                results_n_onoff.append(result_n_onoff)

        ps_n_on, rs_n_on, fs_n_on = list(zip(*results_n_on))
        ps_n_onoff, rs_n_onoff, fs_n_onoff = list(zip(*results_n_onoff))

        # return logs
        logs = {'epoch': self.current_epoch, 
                'precision': np.mean(ps), 
                'recall': np.mean(rs), 
                'f-score': np.mean(fs), 
                'accuracy': np.mean(accs),
                'precision_n_on': np.mean(ps_n_on),
                'recall_n_on': np.mean(rs_n_on),
                'f-score_n_on': np.mean(fs_n_on),
                'precision_n_onoff': np.mean(ps_n_onoff),
                'recall_n_onoff': np.mean(rs_n_onoff),
                'f-score_n_onoff': np.mean(fs_n_onoff)}
        return logs