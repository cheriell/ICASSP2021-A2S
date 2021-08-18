import os
from typing import Any
import pickle
import torch
import pandas as pd
import numpy as np

from audio2score.data.basedataset import BaseDataset
from audio2score.settings import Constants, SpectrogramSetting


class JointTranscriptionDataset(BaseDataset):
    """joint transcription dataset.
    
        Args:
            metadata: metadata to the dataset.
            spectrogram_setting: the spectrogram setting (type and parameters).
            score_type: score representation type, "Reshaped" or "LilyPond".
        """
    def __init__(self, metadata: pd.DataFrame, 
                        spectrogram_setting: SpectrogramSetting, 
                        score_type: str):
        super().__init__()

        self.metadata = metadata
        self.spectrogram_setting = spectrogram_setting
        self.score_type = score_type

        self.init_downbeats_and_length()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        for i, row in self.metadata.iterrows():
            # index to bar_index
            if index >= len(self.downbeats_list[i]) - 2:
                index -= len(self.downbeats_list[i]) - 2
                continue

            spectrogram_pad, spectrogram_length = self.get_spectrogram(row, 
                                                    bar_index=index,
                                                    spectrogram_setting=self.spectrogram_setting)
            pianoroll_pad, pianoroll_mask = self.get_pianoroll(row, bar_index=index)
            score_index_right_pad, score_index_left_pad = self.get_score(row, 
                                                    bar_index=index, 
                                                    score_type=self.score_type)

            return spectrogram_pad, spectrogram_length, pianoroll_pad, pianoroll_mask, score_index_right_pad, score_index_left_pad


class Audio2ScoreTranscriptionDataset(BaseDataset):
    """audio2score transcription dataset.

        Args:
            metadata: metadata to the dataset.
            spectrogram_setting: the spectrogram setting (type and parameters).
            score_type: score representation type, "Reshaped" or "LilyPond".
        """
    def __init__(self, metadata: pd.DataFrame, 
                        spectrogram_setting: SpectrogramSetting, 
                        score_type: str):
        super().__init__()

        self.metadata = metadata
        self.spectrogram_setting = spectrogram_setting
        self.score_type = score_type

        self.init_downbeats_and_length()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        for i, row in self.metadata.iterrows():
            # index to bar_index
            if index >= len(self.downbeats_list[i]) - 2:
                index -= len(self.downbeats_list[i]) - 2
                continue

            spectrogram_pad, spectrogram_length = self.get_spectrogram(row, 
                                                    bar_index=index,
                                                    spectrogram_setting=self.spectrogram_setting)
            score_index_right_pad, score_index_left_pad = self.get_score(row, 
                                                    bar_index=index, 
                                                    score_type=self.score_type)

            return spectrogram_pad, spectrogram_length, score_index_right_pad, score_index_left_pad
            


class PianorollTranscriptionDataset(torch.utils.data.Dataset):
    """pianoroll transcription dataset.

        Args:
            metadata: metadata to the dataset.
            spectrogram_setting: the spectrogram setting (type and parameters).
        """

    def __init__(self, metadata: pd.DataFrame, spectrogram_setting: SpectrogramSetting):
        super().__init__()
        self.metadata = metadata
        self.spectrogram_setting = spectrogram_setting

        # calculate dataset length
        durations = self.metadata['duration'].to_numpy()
        self.n_segments_all = np.ceil(durations / Constants.segment_length).astype(int)
        self.length = np.sum(self.n_segments_all)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        for i, row in self.metadata.iterrows():
            # check if we are looking for a segment in current piece, if not, move to the next.
            if index >= self.n_segments_all[i]:
                index -= self.n_segments_all[i]
                continue

            # get item data
            # load spectrogram and pianorolls
            spectrogram_file = os.path.join(row['spectrograms_folder'], 
                                    f'{self.spectrogram_setting.to_string()}.pkl')
            pianoroll_file = row['pianoroll_file']
            spectrogram_full = pickle.load(open(spectrogram_file, 'rb'))
            if len(spectrogram_full.shape) == 2:
                spectrogram_full = np.expand_dims(spectrogram_full, axis=0)
            pianoroll_full = pickle.load(open(pianoroll_file, 'rb'))

            # get segment
            start = index * Constants.input_length
            end = start + Constants.input_length
            spectrogram = spectrogram_full[:, :, start:min(spectrogram_full.shape[2], end)]
            pianoroll = pianoroll_full[:, start:min(pianoroll_full.shape[1], end)]

            # initialise padded spectrogram and pianoroll
            spectrogram_padded = np.zeros((spectrogram_full.shape[0], spectrogram_full.shape[1], Constants.input_length), dtype=float)
            pianoroll_padded = np.zeros((88, Constants.input_length), dtype=float)
            pianoroll_mask = np.zeros((88, Constants.input_length), dtype=float)
            # overwrite with values
            spectrogram_padded[:, :, :spectrogram.shape[2]] = spectrogram
            pianoroll_padded[:, :pianoroll.shape[1]] = pianoroll
            pianoroll_mask[:, :pianoroll.shape[1]] = 1.

            return spectrogram_padded, pianoroll_padded, pianoroll_mask



