import os
import sys
import pickle
from typing import Any, Optional
import pandas as pd
import pretty_midi as pm
pm.pretty_midi.MAX_TICK = 1e10
import torch
import pytorch_lightning as pl

from audio2score.settings import Constants, SpectrogramSetting, TrainingParam
from audio2score.utilities.spectrogram_utils import SpectrogramUtil
from audio2score.utilities.utils import mkdir
from audio2score.scores.helper import get_downbeats_and_end_time, split_bars
from audio2score.scores.reshapedscore import ScoreReshaped
from audio2score.scores.lilypondscore import ScoreLilyPond

class BaseDataModule(pl.LightningDataModule):
    """Base Datamodule - DO NOT CREATE DATA MODULE HERE!
    
        Shared functions.
        """

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_metadata(dataset_folder: str,
                    feature_folder: str,
                    split: str,
                    three_train_pianos: Optional[bool] = True):
        """Get metadata for the dataset.

            Args:
                dataset_folder: folder to the MuseSyn dataset.
                feature_folder: folder to save pre-calculated features.
                split: train/test/valid split.
                three_train_pianos: whether or not to use only three pianos for model training
            Returns:
                metadata: (pd.DataFrame) dataset metadata.
            """

        if three_train_pianos:
            pianos = Constants.pianos if split == 'test' else Constants.pianos[:-1]
        else:
            pianos = Constants.pianos

        print(f'Get {split} metadata, {len(pianos)} pianos')

        metadata_file = f'metadata/cache/{split}-pianos={len(pianos)}.csv'
        if os.path.exists(metadata_file):
            return pd.read_csv(metadata_file)

        metadata = []
        for i, row in pd.read_csv(f'metadata/{split}.txt').iterrows():
            name = row['name']
            for piano in pianos:
                # get information for each piece
                midi_file = os.path.join(dataset_folder, 'midi', name+'.mid')
                audio_file = os.path.join(dataset_folder, 'flac', piano, name+'.flac')
                score_file = os.path.join(dataset_folder, 'xml', name+'.xml')

                spectrograms_folder = os.path.join(feature_folder, 'spectrograms', piano, name)
                pianoroll_file = os.path.join(feature_folder, 'pianoroll', name+'.pkl')
                downbeats_file = os.path.join(feature_folder, 'downbeats', name+'.pkl')
                score_reshaped_folder = os.path.join(feature_folder, 'score_reshaped', name)
                score_lilypond_folder = os.path.join(feature_folder, 'score_lilypond', name)
                duration = pm.PrettyMIDI(midi_file).get_end_time()

                # udpate metadata
                metadata.append({'name': name, 
                                'piano': piano, 
                                'midi_file': midi_file, 
                                'audio_file': audio_file, 
                                'score_file': score_file,
                                'split': split, 
                                'spectrograms_folder': spectrograms_folder, 
                                'pianoroll_file': pianoroll_file, 
                                'downbeats_file': downbeats_file,
                                'score_reshaped_folder': score_reshaped_folder,
                                'score_lilypond_folder': score_lilypond_folder,
                                'duration': duration})

        # to DataFrame and save metadata
        metadata = pd.DataFrame(metadata)
        mkdir(os.path.split(metadata_file)[0])
        metadata.to_csv(metadata_file)
        return metadata

    @staticmethod
    def prepare_spectrograms(metadata: pd.DataFrame, 
                            spectrogram_setting: Any):
        """Calculate spectrograms and save the pre-calculated features.

            Args:
                metadata: metadata to the dataset.
                spectrogram_setting: the spectrogram setting (type and parameters).
            Returns:
                No return, save pre-calculated features instead.
            """

        for i, row in metadata.iterrows():
            print(f'Preparing spectrogram {i+1}/{len(metadata)}', end='\r')
            
            # get audio file and spectrogram file
            audio_file = row['audio_file']
            spectrogram_file = os.path.join(row['spectrograms_folder'], 
                                    f'{spectrogram_setting.to_string()}.pkl')

            # if already calculated, skip
            if os.path.exists(spectrogram_file):
                continue

            # calculate spectrogram
            if spectrogram_setting.type == 'STFT':
                spectrogram = SpectrogramUtil.STFT_from_file(audio_file, 
                                    win_length=spectrogram_setting.win_length)
            elif spectrogram_setting.type == 'Mel':
                spectrogram = SpectrogramUtil.melspectrogram_from_file(audio_file, 
                                    win_length=spectrogram_setting.win_length, 
                                    n_mels=spectrogram_setting.n_mels)
            elif spectrogram_setting.type == 'CQT':
                spectrogram = SpectrogramUtil.CQT_from_file(audio_file, 
                                    bins_per_octave=spectrogram_setting.bins_per_octave, 
                                    n_octaves=spectrogram_setting.n_octaves)
            elif spectrogram_setting.type == 'HCQT':
                spectrogram = SpectrogramUtil.HCQT_from_file(audio_file, 
                                    bins_per_octave=spectrogram_setting.bins_per_octave, 
                                    n_octaves=spectrogram_setting.n_octaves, 
                                    n_harms=spectrogram_setting.n_harms)
            elif spectrogram_setting.type == 'VQT':
                spectrogram = SpectrogramUtil.VQT_from_file(audio_file, 
                                    bins_per_octave=spectrogram_setting.bins_per_octave, 
                                    n_octaves=spectrogram_setting.n_octaves, 
                                    gamma=spectrogram_setting.gamma)

            # save feature
            mkdir(row['spectrograms_folder'])
            pickle.dump(spectrogram, open(spectrogram_file, 'wb'), protocol=2)
                
        print()

    @staticmethod
    def prepare_pianorolls(metadata: pd.DataFrame):
        """Calculate pianorolls and save pre-calculated feature.

            Args:
                metadata: metadata to the dataset.
            Returns:
                No return, save pre-calculated features instead.
            """

        for i, row in metadata.iterrows():
            print(f'Preparing pianoroll {i+1}/{len(metadata)}', end='\r')

            # get midi file and pianoroll file
            midi_file = row['midi_file']
            pianoroll_file = row['pianoroll_file']

            # if already calculated, skip
            if os.path.exists(pianoroll_file):
                continue

            # calculate pianoroll and save feature
            midi_data = pm.PrettyMIDI(midi_file)
            pianoroll = midi_data.get_piano_roll(fs=1./Constants.hop_time)[21:21+88]  # 88 piano keys
            mkdir(os.path.split(pianoroll_file)[0])
            pickle.dump(pianoroll, open(pianoroll_file, 'wb'), protocol=2)

        print()

    @staticmethod
    def prepare_downbeats(metadata: pd.DataFrame):
        """Calculate downbeats and endtime for each piece

            Args:
                metadata: metadata to the dataset.
            Returns:
                No return, save pre-calculated features instead.
            """

        for i, row in metadata.iterrows():
            print(f'Preparing downbeats {i+1}/{len(metadata)}', end='\r')

            midi_file = row['midi_file']
            downbeats_file = row['downbeats_file']
            if os.path.exists(downbeats_file):
                continue

            # calculate downbeats
            midi_data = pm.PrettyMIDI(midi_file)
            downbeats, _ = get_downbeats_and_end_time(midi_data)
            mkdir(os.path.split(downbeats_file)[0])
            pickle.dump(downbeats, open(downbeats_file, 'wb'), protocol=2)

        print()

    @staticmethod
    def prepare_scores(metadata: pd.DataFrame):
        """Calculate score representations for each piece

            Args:
                metadata: metadata to the dataset.
            Returns:
                No return, save pre-calculated features instead.
            """

        for i, row in metadata.iterrows():
            print(f'Preparing scores {i+1}/{len(metadata)}', end='\r')

            score_file = row['score_file']
            downbeats_file = row['downbeats_file']
            score_reshaped_folder = row['score_reshaped_folder']
            score_lilypond_folder = row['score_lilypond_folder']
            mkdir(score_reshaped_folder)
            mkdir(score_lilypond_folder)

            downbeats = pickle.load(open(downbeats_file, 'rb'))
            if os.path.exists(os.path.join(score_reshaped_folder, f'{len(downbeats)-2}.pkl')) and os.path.exists(os.path.join(score_lilypond_folder, f'{len(downbeats)-2}.pkl')):
                continue
            score_m21_list = split_bars(score_file, len(downbeats))

            for bar_index, score_m21 in enumerate(score_m21_list):
                score_reshaped_right = ScoreReshaped.from_m21(score_m21.right)
                score_reshaped_left = ScoreReshaped.from_m21(score_m21.left)
                score_lilypond_right = ScoreLilyPond.from_m21(score_m21.right)
                score_lilypond_left = ScoreLilyPond.from_m21(score_m21.left)
                pickle.dump(tuple([score_reshaped_right, score_reshaped_left]), 
                            open(os.path.join(score_reshaped_folder, f'{bar_index}.pkl'), 'wb'), 
                            protocol=2)
                pickle.dump(tuple([score_lilypond_right, score_lilypond_left]), 
                            open(os.path.join(score_lilypond_folder, f'{bar_index}.pkl'), 'wb'), 
                            protocol=2)

        print()

    @staticmethod
    def split_spectrograms(metadata: pd.DataFrame, 
                        spectrogram_setting: SpectrogramSetting):
        """Split spectrograms into bars.

            Args:
                meatadata: metadata to the dataset.
                spectrogram_setting: spectrogram setting.
            Returns:
                No return.
            """

        for i, row in metadata.iterrows():
            print(f'Split spectrogram into bars {i+1}/{len(metadata)}', end='\r')

            downbeats_file = row['downbeats_file']
            spectrogram_file = os.path.join(row['spectrograms_folder'], 
                                    f'{spectrogram_setting.to_string()}.pkl')
            
            downbeats = pickle.load(open(downbeats_file, 'rb'))
            if os.path.exists(spectrogram_file+f'.{len(downbeats)-2}.pkl'):
                continue

            spectrogram_full = pickle.load(open(spectrogram_file, 'rb'))
            for bar_index in range(len(downbeats)-1):
                # get start and end time frames
                start_time, end_time = downbeats[bar_index], downbeats[bar_index+1]
                start_frame, end_frame = int(start_time / 0.01), int(end_time / 0.01)
                # get and save spectrogram
                spectrogram = spectrogram_full[:, start_frame:end_frame+1]
                pickle.dump(spectrogram, open(spectrogram_file+f'.{bar_index}.pkl', 'wb'), protocol=2)
        print()
                
    @staticmethod
    def split_pianorolls(metadata: pd.DataFrame):
        """Split pianorolls into bars.

            Args:
                metadata: meatadata to the dataset.
            Returns:
                No return.
            """

        for i, row in metadata.iterrows():
            print(f'Split pianoroll into bars {i+1}/{len(metadata)}', end='\r')

            downbeats_file = row['downbeats_file']
            pianoroll_file = row['pianoroll_file']

            downbeats = pickle.load(open(downbeats_file, 'rb'))
            if os.path.exists(pianoroll_file+f'.{len(downbeats)-2}.pkl'):
                continue

            pianoroll_full = pickle.load(open(pianoroll_file, 'rb'))
            for bar_index in range(len(downbeats)-1):
                # get start and end time frames
                start_time, end_time = downbeats[bar_index], downbeats[bar_index+1]
                start_frame, end_frame = int(start_time / 0.01), int(end_time / 0.01)
                # get and save spectrogram
                spectrogram = pianoroll_full[:, start_frame:end_frame+1]
                pickle.dump(spectrogram, open(pianoroll_file+f'.{bar_index}.pkl', 'wb'), protocol=2)
        print()
  
    def train_dataloader(self):
        # Override train_dataloader
        print('Get train dataloader')
        dataset = self.get_train_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset,          
                            batch_size=TrainingParam.batch_size, 
                            sampler=sampler, 
                            drop_last=True)
        return data_loader
    
    def val_dataloader(self):
        # Override val_dataloader
        print('Get validation dataloader')
        dataset = self.get_valid_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset, 
                            batch_size=TrainingParam.batch_size, 
                            sampler=sampler, 
                            drop_last=True)
        return data_loader

    def test_dataloader(self):
        # Override test_dataloader
        print('Get test dataloader')
        dataset = self.get_test_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset, 
                            batch_size=TrainingParam.batch_size, 
                            sampler=sampler, 
                            drop_last=True)
        return data_loader
