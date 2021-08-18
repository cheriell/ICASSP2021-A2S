from typing import Optional
import numpy as np
import torch


from audio2score.scores.symbols import EOS
from audio2score.scores.reshapedscore import ScoreReshaped
from audio2score.scores.lilypondscore import ScoreLilyPond
from audio2score.models.models import Audio2ScoreTranscriptionModel, JointTranscriptionModel
from audio2score.settings import Constants



class Audio2ScoreTranscriber(object):
    """Audio to score tanscriber, transcribe using pretrained model.

        Args:
            model_checkpoint: path to the pre-trained model checkpoint.
            score_type: "Reshaped" or "LilyPond".
            model_type: "joint" or "audio2score".
            gpu: gpu device id.
        """

    def __init__(self, model_checkpoint: str, 
                        score_type: Optional[str] = 'Reshaped', 
                        model_type: Optional[str] = 'joint',
                        gpu: Optional[int] = None):

        self.device = f'cuda:{gpu}' if gpu else 'cpu'
        self.score_type = score_type
        self.model_type = model_type
        if model_type == 'audio2score':
            self.model = Audio2ScoreTranscriptionModel.load_from_checkpoint(model_checkpoint,
                                                                    score_type=score_type)
        elif model_type == 'joint':
            self.model = JointTranscriptionModel.load_from_checkpoint(model_checkpoint,
                                                                    score_type=score_type)
        self.model.to(self.device)
        self.model.eval()

    def transcribe_one_bar_from_spectrogram(self, spectrogram: np.ndarray):
        """Transcribe one bar from the VQT spectrogram.

            Args:
                spectrogram: VQT spectrogram in numpy array.
            Returns:
                score_right: (ScoreReshaped or ScoreLilyPond) predicted score representation for right hand
                score_left: (ScoreReshaped or ScoreLilyPond) predicted score representation for left hand
            """
        # prepare spectrogram
        spectrogram_pad = np.zeros((spectrogram.shape[0], Constants.spectrogram_max_length), dtype=float)
        spectrogram_pad[:,:spectrogram.shape[1]] = spectrogram
        spectrogram_pad = torch.Tensor(spectrogram_pad).to(self.device).float().unsqueeze(0).unsqueeze(1)
                        # [1, 1, freq_bins, spectrogram_max_length]

        if self.score_type == 'Reshaped':
            max_length_right, max_length_left = Constants.score_max_length_reshaped
        elif self.score_type == 'LilyPond':
            max_length_right, max_length_left = Constants.score_max_length_lilypond
        
        output_data = self.model(spectrogram=spectrogram_pad,
                                                target_length_right=max_length_right, 
                                                target_length_left=max_length_left)
        if self.model_type == 'audio2score':
            probs_right, probs_left, _, _ = output_data
        elif self.model_type == 'joint':
            _, probs_right, probs_left, _, _ = output_data
        
        preds_right = [probs_right[i][0].topk(1)[1] for i in range(self.model.n_out)]
                        # List[[max_length_right, 1]]
        preds_left = [probs_left[i][0].topk(1)[1] for i in range(self.model.n_out)]
                        # List[[max_length_left, 1]]
        preds_right = torch.cat(preds_right, dim=1).transpose(0, 1)
                        # [n_out, max_length_right]
        preds_left = torch.cat(preds_left, dim=1).transpose(0, 1)
                        # [n_out, max_length_left]
        
        right_length = torch.min((preds_right == EOS).nonzero()[:,1]) \
                        if (preds_right == EOS).nonzero()[:,1].shape.numel() > 0 \
                        else max_length_right
        left_length = torch.min((preds_left == EOS).nonzero()[:,1]) \
                        if (preds_left == EOS).nonzero()[:,1].shape.numel() > 0 \
                        else max_length_left
        preds_right = preds_right[:,:right_length]
        preds_left = preds_left[:,:left_length]
        
        if self.score_type == 'Reshaped':
            score_right = ScoreReshaped.from_index_matrix(preds_right.cpu().detach().numpy())
            score_left = ScoreReshaped.from_index_matrix(preds_left.cpu().detach().numpy())
        elif self.score_type == 'LilyPond':
            score_right = ScoreLilyPond.from_index_matrix(preds_right.cpu().detach().numpy())
            score_left = ScoreLilyPond.from_index_matrix(preds_left.cpu().detach().numpy())

        return score_right, score_left

