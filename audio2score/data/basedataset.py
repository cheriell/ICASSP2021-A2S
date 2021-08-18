import os
import pickle
import numpy as np
import pandas as pd
import torch

from audio2score.settings import Constants
from audio2score.scores.symbols import PAD, EOS
from audio2score.settings import SpectrogramSetting


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset.

        Shared utility functions.
        """

    def __init__(self):
        super().__init__()


    def init_downbeats_and_length(self):
        # initialise downbeats for each piece and dataset length
        self.downbeats_list = []
        self.length = 0
        for i, row in self.metadata.iterrows():
            downbeats = pickle.load(open(row['downbeats_file'], 'rb'))
            self.downbeats_list.append(downbeats)
            self.length += len(downbeats) - 2  # ignore the final bar.


    @staticmethod
    def get_spectrogram(row: pd.Series, bar_index: int, spectrogram_setting: SpectrogramSetting):
        """Get spectrogram for the item.
            
            Args:
                row: music piece information as row in the metadata.
                bar_index: bar index within the music piece.
                spectrogram_setting: spectrogram setting.
            Returns:
                spectrogram_pad: padded spectrogram as model input feature.
                spectrogram_length: valid length in the padded spectrogram.
            """

        spectrogram_file = os.path.join(row['spectrograms_folder'], 
                                f'{spectrogram_setting.to_string()}.pkl')
        spectrogram = pickle.load(open(spectrogram_file+f'.{bar_index}.pkl', 'rb'))
        
        spectrogram_length = min(spectrogram.shape[-1], Constants.spectrogram_max_length)
        spectrogram_pad = np.zeros((spectrogram.shape[0], Constants.spectrogram_max_length), dtype=float)
        spectrogram_pad[:,:spectrogram_length] = spectrogram[:,:spectrogram_length]
        spectrogram_pad = np.expand_dims(spectrogram_pad, axis=0)

        return spectrogram_pad, spectrogram_length

    @staticmethod
    def get_pianoroll(row: pd.Series, bar_index: int):
        """Get pianoroll and pianoroll mask of the item.
            
            Args:
                row: music piece information as row in the metadata.
                bar_index: bar index within the music piece.
            Returns:
                pianoroll_pad: padded pianoroll for target data.
                pianoroll_mask: pianoroll mask with 1s for pianoroll data and 0s for paddings.
            """
        pianoroll = pickle.load(open(row['pianoroll_file']+f'.{bar_index}.pkl', 'rb'))
        pianoroll = pianoroll[:, ::2]

        pianoroll_pad = np.zeros((88, Constants.spectrogram_max_length//2), dtype=float)
        pianoroll_pad[:, :pianoroll.shape[1]] = pianoroll
        pianoroll_mask = np.zeros((88, Constants.spectrogram_max_length//2), dtype=float)
        pianoroll_mask[:, :pianoroll.shape[1]] = 1.

        return pianoroll_pad, pianoroll_mask


    @staticmethod
    def get_score(row: pd.Series, bar_index: int, score_type: str):
        """Get score representation as index matrix for the item.

            Args:
                row: music piece information as row in the metadata.
                bar_index: bar index within the music piece.
                spectrogram_setting: spectrogram setting.
            Returns:
                score_index_right_pad: padded score representation as indexes.
                score_index_left_pad: padded score representation as indexes.
            """

        if score_type == 'Reshaped':
            score_folder = row['score_reshaped_folder']
            score_max_length_right, score_max_length_left = Constants.score_max_length_reshaped
        elif score_type == 'LilyPond':
            score_folder = row['score_lilypond_folder']
            score_max_length_right, score_max_length_left = Constants.score_max_length_lilypond

        score_right, score_left = pickle.load(open(os.path.join(score_folder, f'{bar_index}.pkl'), 'rb'))
        score_index_right = score_right.to_index_matrix()
        score_index_left = score_left.to_index_matrix()

        score_index_right_pad = np.ones((score_index_right.shape[0], score_max_length_right), dtype=int) * PAD
        score_index_left_pad = np.ones((score_index_left.shape[0], score_max_length_left), dtype=int) * PAD
        score_index_right_pad[:,:score_index_right.shape[1]] = score_index_right
        score_index_left_pad[:,:score_index_left.shape[1]] = score_index_left
        score_index_right_pad[:,score_index_right.shape[1]] = EOS
        score_index_left_pad[:,score_index_left.shape[1]] = EOS

        return score_index_right_pad, score_index_left_pad
