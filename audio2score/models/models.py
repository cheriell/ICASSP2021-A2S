
import sys
from typing import Optional
import numpy as np
import functools
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from audio2score.settings import Constants, TrainingParam
from audio2score.models.basemodel import BaseModel
from audio2score.models.containers import ConvBlock, ConvStack, Seq2Seq
from audio2score.models.containers import init_layer, init_bn, init_gru
from audio2score.scores.symbols import vocab_size_name, vocab_size_octave, vocab_size_tie, \
                                        vocab_size_duration, vocab_size, PAD, EOS
from audio2score.scores.reshapedscore import ScoreReshaped
from audio2score.scores.lilypondscore import ScoreLilyPond
from audio2score.utilities.evaluation_utils import Eval


class JointTranscriptionModel(BaseModel):
    """Joint transcription model.

        Args:
            score_type: 'Reshaped' or 'LilyPond'.
            in_channels: number of channels of the input feature.
            freq_bins: number of frequency bins of the input feature.
            conv_stack_output_size: hidden size of the convolutional stack output layer.
            hidden_size: hidden size of the GRU layers in the sequence-to-sequence architectures.
            encoder_max_length: length of the encoder input sequence.
        """

    def __init__(self, score_type: str = 'Reshaped',
                    in_channels: int = 1,
                    freq_bins: int = 480,
                    conv_stack_output_size: int = 200,
                    hidden_size: int = 100,
                    encoder_max_length: int = Constants.spectrogram_max_length // 2):
        super().__init__()

        self.score_type = score_type
        if score_type == 'Reshaped':
            vocab_sizes = [vocab_size_name] * 5 + [vocab_size_octave] * 5 + [vocab_size_tie] * 5 + [vocab_size_duration]
        elif score_type == 'LilyPond':
            vocab_sizes = [vocab_size]
        else:
            sys.exit('Error score type!')
        self.n_out = len(vocab_sizes)

        self.conv_stack = ConvStack(in_channels, freq_bins, conv_stack_output_size)

        self.gru_pianoroll = nn.GRU(input_size=200, hidden_size=100, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(200, 88, bias=True)

        self.seq2seq_right = Seq2Seq(input_size=conv_stack_output_size,
                                    hidden_size=hidden_size,
                                    encoder_max_length=encoder_max_length,
                                    vocab_sizes=vocab_sizes)
        self.seq2seq_left = Seq2Seq(input_size=conv_stack_output_size,
                                    hidden_size=hidden_size,
                                    encoder_max_length=encoder_max_length,
                                    vocab_sizes=vocab_sizes)

        self.init_weight()
        self.init_loss_function()
        self.init_parameters()

    def init_weight(self):
        """Initialise model weights"""
        init_gru(self.gru_pianoroll)
        init_layer(self.fc)

    def init_loss_function(self):
        """Initialise loss function"""
        self.loss_fn_pianoroll = nn.MSELoss()
        self.loss_fn_score = nn.NLLLoss(ignore_index=PAD)

    def init_parameters(self):
        self.learning_rate = TrainingParam.learning_rate
        self.weight_decay = 1e-3
        self.schedular_step_size = 1
        self.schedular_gamma = 0.98
        self.moniter_metric = 'valid_wer'

    def forward(self, spectrogram: torch.Tensor, 
                    target_length_right: torch.Tensor, 
                    target_length_left: torch.Tensor,
                    right: Optional[torch.Tensor] = None,
                    left: Optional[torch.Tensor] = None,
                    teacher_forcing_ratio: float = 0.):

        batch_size = spectrogram.shape[0]

        conv_stack_output = self.conv_stack(spectrogram)
                                # [batch_size, encoder_max_length, conv_stack_output_size]
                                
        rnn_output, _ = self.gru_pianoroll(conv_stack_output)  # [batch_size, encoder_max_length, 200]
        rnn_output = F.dropout(rnn_output, p=0.5, training=self.training, inplace=False)  # same as above
        pianoroll = F.elu(self.fc(rnn_output).transpose(1, 2))  # [batch_size, 88, encoder_max_length]
        
        probs_right, attn_weights_right = self.seq2seq_right(inputs=conv_stack_output, 
                                                             target_length=target_length_right, 
                                                             batch_size=batch_size,
                                                             device=self.device,
                                                             target=right,
                                                             teacher_forcing_ratio=teacher_forcing_ratio)
        # probs_right: List[[batch_size, target_length_right, vocab_sizes[i]]]
        # attn_weights_right: [batch_size, target_length_right, encoder_max_length]
        
        probs_left, attn_weights_left = self.seq2seq_left(inputs=conv_stack_output,
                                                         target_length=target_length_left,
                                                         batch_size=batch_size,
                                                         device=self.device,
                                                         target=left,
                                                         teacher_forcing_ratio=teacher_forcing_ratio)
        # probs_left: List[[batch_size, target_length_left, vocab_sizes[i]]]
        # attn_weights_left: [batch_size, target_length_left, encoder_max_length]
        
        return pianoroll, probs_right, probs_left, attn_weights_right, attn_weights_left

    def prepare_batch_data(self, batch):
        spectrogram_pad, spectrogram_length, pianoroll, pianoroll_mask, score_index_right_pad, score_index_left_pad = batch

        spectrogram_pad = spectrogram_pad.float()  # [batch_size, freq_bins, spectrogram_max_length]

        pianoroll_targ = pianoroll.float()
        pianoroll_mask = pianoroll_mask.float()

        score_index_right_pad = score_index_right_pad.long()  # [batch_size, 16, score_max_length_right]
        score_index_left_pad = score_index_left_pad.long()  # [batch_size, 16, score_max_length_left]
        
        input_data = (spectrogram_pad, pianoroll_mask, score_index_right_pad, score_index_left_pad)
        target_data = (pianoroll_targ, spectrogram_length, score_index_right_pad, score_index_left_pad)
        return input_data, target_data

    def predict(self, input_data):
        spectrogram, pianoroll_mask, right, left = input_data
        if self.training:
            pianoroll_pred, probs_right, probs_left, _, _ = self(spectrogram=spectrogram,
                                        target_length_right=right.shape[2],
                                        target_length_left=left.shape[2],
                                        right=right,
                                        left=left,
                                        teacher_forcing_ratio=0.5)
        else:
            pianoroll_pred, probs_right, probs_left, _, _ = self(spectrogram=spectrogram,
                                        target_length_right=right.shape[2],
                                        target_length_left=left.shape[2],
                                        teacher_forcing_ratio=0.)
        pianoroll_pred = pianoroll_pred * pianoroll_mask  # add mask to ignore paddings
        return pianoroll_pred, probs_right, probs_left  # output_data

    def get_loss(self, output_data, target_data):
        pianoroll_pred, probs_right, probs_left = output_data
        pianoroll_targ, spectrogram_length, right, left = target_data

        loss_pianoroll = self.loss_fn_pianoroll(pianoroll_pred, pianoroll_targ) * 2
        loss_right = [self.loss_fn_score(probs_right[i].permute(0,2,1), right[:,i,:]) for i in range(len(probs_right))]
        loss_left = [self.loss_fn_score(probs_left[i].permute(0,2,1), left[:,i,:]) for i in range(len(probs_left))]
        loss = functools.reduce(lambda x,y:x+y, [loss_pianoroll / torch.max(spectrogram_length) * Constants.spectrogram_max_length] + loss_right + loss_left)
        return loss

    def evaluate(self, output_data, target_data):
        pianoroll_pred, probs_right, probs_left = output_data
        pianoroll_targ, _, right, left = target_data

        # pianoroll (F-measure)
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

        # score
        accs_right, accs_left = [], []
        wers_right, wers_left = [], []
        for sample_index in range(TrainingParam.batch_size):
            right_length = torch.min((right[sample_index] == EOS).nonzero()[:,1])
            left_length = torch.min((left[sample_index] == EOS).nonzero()[:,1])
            
            preds_right = [probs_right[i][sample_index, :right_length].topk(1)[1] for i in range(self.n_out)]
                                                    # List[[right_length, 1]]
            preds_right = torch.cat(preds_right, dim=1).transpose(0, 1)  # [n_out, right_length]
            preds_left = [probs_left[i][sample_index,:left_length].topk(1)[1] for i in range(self.n_out)]
                                                    # List[[left_length, 1]]
            preds_left = torch.cat(preds_left, dim=1).transpose(0, 1)  # [n_out, left_length]
            
            acc_right = torch.sum(preds_right == right[sample_index,:,:right_length]).item() / preds_right.shape.numel()
            acc_left = torch.sum(preds_left == left[sample_index,:,:left_length]).item() / preds_left.shape.numel()
            accs_right.append(acc_right)
            accs_left.append(acc_left)

            # get predicted score and calculate WER
            right_length_pred = torch.min((preds_right == EOS).nonzero()[:,1]) \
                                if (preds_right == EOS).nonzero()[:,1].shape.numel() > 0 \
                                else right.shape[2]
            left_length_pred = torch.min((preds_left == EOS).nonzero()[:,1]) \
                                if (preds_left == EOS).nonzero()[:,1].shape.numel() > 0 \
                                else left.shape[2]
            preds_right = preds_right[:,:right_length_pred]
            preds_left = preds_left[:,:left_length_pred]

            if self.score_type == 'Reshaped':
                score_right_raw = ScoreReshaped.from_index_matrix(right[sample_index,:,:right_length].cpu().detach().numpy())
                score_left_raw = ScoreReshaped.from_index_matrix(left[sample_index,:,:left_length].cpu().detach().numpy())
                score_right_raw = ScoreLilyPond.from_m21(score_right_raw.to_m21())
                score_left_raw = ScoreLilyPond.from_m21(score_left_raw.to_m21())
                
                score_right = ScoreReshaped.from_index_matrix(preds_right.cpu().detach().numpy())
                score_left = ScoreReshaped.from_index_matrix(preds_left.cpu().detach().numpy())
                score_right = ScoreLilyPond.from_m21(score_right.to_m21())
                score_left = ScoreLilyPond.from_m21(score_left.to_m21())
                
            elif self.score_type == 'LilyPond':
                score_right_raw = ScoreLilyPond.from_index_matrix(right[sample_index,:,:right_length].cpu().detach().numpy())
                score_left_raw = ScoreLilyPond.from_index_matrix(left[sample_index,:,:left_length].cpu().detach().numpy())
                
                score_right = ScoreLilyPond.from_index_matrix(preds_right.cpu().detach().numpy())
                score_left = ScoreLilyPond.from_index_matrix(preds_left.cpu().detach().numpy())
            wer_right = Eval.wer_evaluation(score_right_raw.to_string(), score_right.to_string())
            wer_left = Eval.wer_evaluation(score_left_raw.to_string(), score_left.to_string())
            wers_right.append(wer_right)
            wers_left.append(wer_left)

        # return logs
        logs = {'epoch': self.current_epoch, 
                # pianoroll
                'precision': np.mean(ps), 
                'recall': np.mean(rs), 
                'f-score': np.mean(fs), 
                'accuracy': np.mean(accs),
                'precision_n_on': np.mean(ps_n_on),
                'recall_n_on': np.mean(rs_n_on),
                'f-score_n_on': np.mean(fs_n_on),
                'precision_n_onoff': np.mean(ps_n_onoff),
                'recall_n_onoff': np.mean(rs_n_onoff),
                'f-score_n_onoff': np.mean(fs_n_onoff),
                # score
                'acc_right': np.mean(accs_right),
                'acc_left': np.mean(accs_left),
                'acc': np.mean(accs_right + accs_left),
                'wer_right': np.mean(wers_right),
                'wer_left': np.mean(wers_left),
                'wer': np.mean(wers_right + wers_left)}
        return logs
        

class Audio2ScoreTranscriptionModel(BaseModel):
    """Audio-to-score transcription model.

        Args:
            score_type: 'Reshaped' or 'LilyPond'.
            in_channels: number of channels of the input feature.
            freq_bins: number of frequency bins of the input feature.
            conv_stack_output_size: hidden size of the convolutional stack output layer.
            hidden_size: hidden size of the GRU layers in the sequence-to-sequence architectures.
            encoder_max_length: length of the encoder input sequence.
        """

    def __init__(self, score_type: str = 'Reshaped',
                    in_channels: int = 1,
                    freq_bins: int = 480,
                    conv_stack_output_size: int = 200,
                    hidden_size: int = 100,
                    encoder_max_length: int = Constants.spectrogram_max_length // 2):
        super().__init__()

        self.score_type = score_type
        if score_type == 'Reshaped':
            vocab_sizes = [vocab_size_name] * 5 + [vocab_size_octave] * 5 + [vocab_size_tie] * 5 + [vocab_size_duration]
        elif score_type == 'LilyPond':
            vocab_sizes = [vocab_size]
        else:
            sys.exit('Error score type!')
        self.n_out = len(vocab_sizes)
        
        self.conv_stack = ConvStack(in_channels, freq_bins, conv_stack_output_size)
        self.seq2seq_right = Seq2Seq(input_size=conv_stack_output_size,
                                    hidden_size=hidden_size,
                                    encoder_max_length=encoder_max_length,
                                    vocab_sizes=vocab_sizes)
        self.seq2seq_left = Seq2Seq(input_size=conv_stack_output_size,
                                    hidden_size=hidden_size,
                                    encoder_max_length=encoder_max_length,
                                    vocab_sizes=vocab_sizes)

        self.init_loss_function()
        self.init_parameters()

    def init_loss_function(self):
        self.loss_fn = nn.NLLLoss(ignore_index=PAD)

    def init_parameters(self):
        self.learning_rate = TrainingParam.learning_rate
        self.weight_decay = 1e-3
        self.schedular_step_size = 1
        self.schedular_gamma = 0.98
        self.moniter_metric = 'valid_wer'

    def forward(self, spectrogram, target_length_right, target_length_left,
                    right=None, left=None, teacher_forcing_ratio=0.):

        batch_size = spectrogram.shape[0]

        conv_stack_output = self.conv_stack(spectrogram)
                                # [batch_size, encoder_max_length, conv_stack_output_size]
        
        probs_right, attn_weights_right = self.seq2seq_right(inputs=conv_stack_output, 
                                                             target_length=target_length_right, 
                                                             batch_size=batch_size,
                                                             device=self.device,
                                                             target=right,
                                                             teacher_forcing_ratio=teacher_forcing_ratio)
        # probs_right: List[[batch_size, target_length_right, vocab_sizes[i]]]
        # attn_weights_right: [batch_size, target_length_right, encoder_max_length]
        
        probs_left, attn_weights_left = self.seq2seq_left(inputs=conv_stack_output,
                                                         target_length=target_length_left,
                                                         batch_size=batch_size,
                                                         device=self.device,
                                                         target=left,
                                                         teacher_forcing_ratio=teacher_forcing_ratio)
        # probs_left: List[[batch_size, target_length_left, vocab_sizes[i]]]
        # attn_weights_left: [batch_size, target_length_left, encoder_max_length]
        
        return probs_right, probs_left, attn_weights_right, attn_weights_left
    
    def prepare_batch_data(self, batch):
        spectrogram_pad, _, score_index_right_pad, score_index_left_pad = batch
        spectrogram_pad = spectrogram_pad.float()  # [batch_size, freq_bins, spectrogram_max_length]
        score_index_right_pad = score_index_right_pad.long()  # [batch_size, 16, score_max_length_right]
        score_index_left_pad = score_index_left_pad.long()  # [batch_size, 16, score_max_length_left]
        
        input_data = (spectrogram_pad, score_index_right_pad, score_index_left_pad)
        target_data = (score_index_right_pad, score_index_left_pad)
        return input_data, target_data

    def predict(self, input_data):
        spectrogram, right, left = input_data
        if self.training:
            probs_right, probs_left, _, _ = self(spectrogram=spectrogram,
                                        target_length_right=right.shape[2],
                                        target_length_left=left.shape[2],
                                        right=right,
                                        left=left,
                                        teacher_forcing_ratio=0.5)
        else:
            probs_right, probs_left, _, _ = self(spectrogram=spectrogram,
                                        target_length_right=right.shape[2],
                                        target_length_left=left.shape[2],
                                        teacher_forcing_ratio=0.)
        return probs_right, probs_left  # output_data

    def get_loss(self, output_data, target_data):
        probs_right, probs_left = output_data
        right, left = target_data

        loss_right = [self.loss_fn(probs_right[i].permute(0,2,1), right[:,i,:]) for i in range(len(probs_right))]
        loss_left = [self.loss_fn(probs_left[i].permute(0,2,1), left[:,i,:]) for i in range(len(probs_left))]
        loss = functools.reduce(lambda x,y:x+y, loss_right + loss_left)
        return loss

    def evaluate(self, output_data, target_data):
        probs_right, probs_left = output_data
        right, left = target_data

        accs_right, accs_left = [], []
        wers_right, wers_left = [], []
        for sample_index in range(TrainingParam.batch_size):
            right_length = torch.min((right[sample_index] == EOS).nonzero()[:,1])
            left_length = torch.min((left[sample_index] == EOS).nonzero()[:,1])
            
            preds_right = [probs_right[i][sample_index, :right_length].topk(1)[1] for i in range(self.n_out)]
                                                    # List[[right_length, 1]]
            preds_right = torch.cat(preds_right, dim=1).transpose(0, 1)  # [n_out, right_length]
            preds_left = [probs_left[i][sample_index,:left_length].topk(1)[1] for i in range(self.n_out)]
                                                    # List[[left_length, 1]]
            preds_left = torch.cat(preds_left, dim=1).transpose(0, 1)  # [n_out, left_length]
            
            acc_right = torch.sum(preds_right == right[sample_index,:,:right_length]).item() / preds_right.shape.numel()
            acc_left = torch.sum(preds_left == left[sample_index,:,:left_length]).item() / preds_left.shape.numel()
            accs_right.append(acc_right)
            accs_left.append(acc_left)

            # get predicted score and calculate WER
            right_length_pred = torch.min((preds_right == EOS).nonzero()[:,1]) \
                                if (preds_right == EOS).nonzero()[:,1].shape.numel() > 0 \
                                else right.shape[2]
            left_length_pred = torch.min((preds_left == EOS).nonzero()[:,1]) \
                                if (preds_left == EOS).nonzero()[:,1].shape.numel() > 0 \
                                else left.shape[2]
            preds_right = preds_right[:,:right_length_pred]
            preds_left = preds_left[:,:left_length_pred]

            if self.score_type == 'Reshaped':
                score_right_raw = ScoreReshaped.from_index_matrix(right[sample_index,:,:right_length].cpu().detach().numpy())
                score_left_raw = ScoreReshaped.from_index_matrix(left[sample_index,:,:left_length].cpu().detach().numpy())
                score_right_raw = ScoreLilyPond.from_m21(score_right_raw.to_m21())
                score_left_raw = ScoreLilyPond.from_m21(score_left_raw.to_m21())
                
                score_right = ScoreReshaped.from_index_matrix(preds_right.cpu().detach().numpy())
                score_left = ScoreReshaped.from_index_matrix(preds_left.cpu().detach().numpy())
                score_right = ScoreLilyPond.from_m21(score_right.to_m21())
                score_left = ScoreLilyPond.from_m21(score_left.to_m21())
                
            elif self.score_type == 'LilyPond':
                score_right_raw = ScoreLilyPond.from_index_matrix(right[sample_index,:,:right_length].cpu().detach().numpy())
                score_left_raw = ScoreLilyPond.from_index_matrix(left[sample_index,:,:left_length].cpu().detach().numpy())
                
                score_right = ScoreLilyPond.from_index_matrix(preds_right.cpu().detach().numpy())
                score_left = ScoreLilyPond.from_index_matrix(preds_left.cpu().detach().numpy())
            wer_right = Eval.wer_evaluation(score_right_raw.to_string(), score_right.to_string())
            wer_left = Eval.wer_evaluation(score_left_raw.to_string(), score_left.to_string())
            wers_right.append(wer_right)
            wers_left.append(wer_left)

        logs = {'epoch': self.current_epoch,
                'acc_right': np.mean(accs_right),
                'acc_left': np.mean(accs_left),
                'acc': np.mean(accs_right + accs_left),
                'wer_right': np.mean(wers_right),
                'wer_left': np.mean(wers_left),
                'wer': np.mean(wers_right + wers_left)}

        return logs


class PianorollTranscriptionModel(BaseModel):
    """pianoroll transcription model.

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
        self.init_parameters()
        
    def init_weight(self):
        """Initialise model weights"""
        init_layer(self.fc3)
        init_bn(self.bn3)
        init_gru(self.gru)
        init_layer(self.fc)

    def init_loss_function(self):
        """Initialise loss function"""
        self.loss_fn = nn.MSELoss()

    def init_parameters(self):
        self.learning_rate = TrainingParam.learning_rate
        self.weight_decay = 2e-5
        self.schedular_step_size = 2
        self.schedular_gamma = 0.9
        self.moniter_metric = 'valid_loss'

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