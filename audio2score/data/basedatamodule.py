import os
import sys
import pickle
from typing import Any
import pandas as pd
import pretty_midi as pm
import torch
import pytorch_lightning as pl

from audio2score.settings import Constants, SpectrogramSetting, TrainingParam
from audio2score.utilities.spectrogram_utils import SpectrogramUtil
from audio2score.utilities.utils import mkdir

class BaseDataModule(pl.LightningDataModule):
    """Base Datamodule - DO NOT CREATE DATA MODULE HERE!
    
        Shared functions.
        """
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_metadata(dataset_folder: str,
                    feature_folder: str,
                    split: str):
        """Get metadata for the dataset.

        Args:
            dataset_folder: folder to the MuseSyn dataset.
            feature_folder: folder to save pre-calculated features.
            split: train/test/valid split.
        Returns:
            metadata: (pd.DataFrame) dataset metadata.
        """
        print(f'Get {split} metadata')
        pianos = Constants.pianos if split == 'test' else Constants.pianos[:-1]

        metadata = []
        for i, row in pd.read_csv(f'metadata/{split}.txt').iterrows():
            name = row['name']
            for piano in pianos:
                # get information for each piece
                midi_file = os.path.join(dataset_folder, 'midi', name+'.mid')
                audio_file = os.path.join(dataset_folder, 'flac', piano, name+'.flac')
                spectrograms_folder = os.path.join(feature_folder, 'spectrograms', piano, name)
                pianoroll_file = os.path.join(feature_folder, 'pianoroll', name+'.pkl')
                duration = pm.PrettyMIDI(midi_file).get_end_time()

                # udpate metadata
                metadata.append({'name': name, 
                                'piano': piano, 
                                'midi_file': midi_file, 
                                'audio_file': audio_file, 
                                'split': split, 
                                'spectrograms_folder': spectrograms_folder, 'pianoroll_file': pianoroll_file, 
                                'duration': duration})

        # to DataFrame and save metadata
        metadata = pd.DataFrame(metadata)
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
                
        print('\ndone!')

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

        print('\ndone!')
        
    def train_dataloader(self):
        """Override train_dataloader"""
        print('Get train dataloader')
        dataset = self.get_train_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset,          
                            batch_size=TrainingParam.batch_size, 
                            sampler=sampler, 
                            drop_last=True)
        return data_loader
    
    def val_dataloader(self):
        """Override val_dataloader"""
        print('Get validation dataloader')
        dataset = self.get_valid_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset, 
                            batch_size=TrainingParam.batch_size, 
                            sampler=sampler, 
                            drop_last=True)
        return data_loader

    def test_dataloader(self):
        """Override test_dataloader"""
        print('Get test dataloader')
        dataset = self.get_test_dataset()
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.dataloader.DataLoader(dataset, 
                            batch_size=TrainingParam.batch_size, 
                            sampler=sampler, 
                            drop_last=True)
        return data_loader