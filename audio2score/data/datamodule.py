

from typing import Any

from audio2score.data.basedatamodule import BaseDataModule
from audio2score.data.datasets import PianorollTranscriptionDataset, Audio2ScoreTranscriptionDataset, JointTranscriptionDataset


class JointTranscriptionDataModule(BaseDataModule):
    """Joint Transcription DataModule.
    
        Args:
            spectrogram_setting: the spectrogram setting (type and parameters).
            score_type: 'Reshaped' or 'LilyPond'.
            dataset_folder: folder to the MuseSyn dataset.
            feature_folder: folder to save pre-calculated features.
        """
    def __init__(self, spectrogram_setting: Any,
                score_type: str,
                dataset_folder: str,
                feature_folder: str):
        super().__init__()

        self.spectrogram_setting = spectrogram_setting
        self.score_type = score_type
        self.metadata_train = self.get_metadata(dataset_folder, feature_folder, 'train', three_train_pianos=True)
        self.metadata_valid = self.get_metadata(dataset_folder, feature_folder, 'valid', three_train_pianos=True)
        self.metadata_test = self.get_metadata(dataset_folder, feature_folder, 'test', three_train_pianos=True)

    def prepare_data(self):
        # Override prepare_data
        for metadata, split in [(self.metadata_train, 'train'), 
                        (self.metadata_valid, 'validation'),
                        (self.metadata_test, 'test')]:
            print('prepare', split, 'set')
            self.prepare_spectrograms(metadata, self.spectrogram_setting)
            self.prepare_downbeats(metadata)
            self.prepare_scores(metadata)
            self.split_spectrograms(metadata, self.spectrogram_setting)
            self.split_pianorolls(metadata)

    def get_train_dataset(self):
            return JointTranscriptionDataset(self.metadata_train, self.spectrogram_setting, self.score_type)

    def get_valid_dataset(self):
        return JointTranscriptionDataset(self.metadata_valid, self.spectrogram_setting, self.score_type)

    def get_test_dataset(self):
        return JointTranscriptionDataset(self.metadata_test, self.spectrogram_setting, self.score_type)
            


class Audio2ScoreTranscriptionDataModule(BaseDataModule):
    """Audio2score Transcription DataModule.

        Args:
            spectrogram_setting: the spectrogram setting (type and parameters).
            score_type: 'Reshaped' or 'LilyPond'.
            dataset_folder: folder to the MuseSyn dataset.
            feature_folder: folder to save pre-calculated features.
        """
    def __init__(self, spectrogram_setting: Any,
                score_type: str,
                dataset_folder: str,
                feature_folder: str):
        super().__init__()

        self.spectrogram_setting = spectrogram_setting
        self.score_type = score_type
        self.metadata_train = self.get_metadata(dataset_folder, feature_folder, 'train', three_train_pianos=True)
        self.metadata_valid = self.get_metadata(dataset_folder, feature_folder, 'valid', three_train_pianos=True)
        self.metadata_test = self.get_metadata(dataset_folder, feature_folder, 'test', three_train_pianos=True)
    
    def prepare_data(self):
        # Override prepare_data
        for metadata, split in [(self.metadata_train, 'train'), 
                        (self.metadata_valid, 'validation'),
                        (self.metadata_test, 'test')]:
            print('prepare', split, 'set')
            self.prepare_spectrograms(metadata, self.spectrogram_setting)
            self.prepare_downbeats(metadata)
            self.prepare_scores(metadata)
            self.split_spectrograms(metadata, self.spectrogram_setting)
            self.split_pianorolls(metadata)

    def get_train_dataset(self):
        return Audio2ScoreTranscriptionDataset(self.metadata_train, self.spectrogram_setting, self.score_type)

    def get_valid_dataset(self):
        return Audio2ScoreTranscriptionDataset(self.metadata_valid, self.spectrogram_setting, self.score_type)

    def get_test_dataset(self):
        return Audio2ScoreTranscriptionDataset(self.metadata_test, self.spectrogram_setting, self.score_type)


class PianorollTranscriptionDataModule(BaseDataModule):
    """Pianoroll Transcription DataModule.

        Args:
            spectrogram_setting: the spectrogram setting (type and parameters).
            dataset_folder: folder to the MuseSyn dataset.
            feature_folder: folder to save pre-calculated features.
        """
    def __init__(self, spectrogram_setting: Any,
                dataset_folder: str,
                feature_folder: str):
        super().__init__()

        self.spectrogram_setting = spectrogram_setting
        self.metadata_train = self.get_metadata(dataset_folder, feature_folder, 'train', three_train_pianos=False)
        self.metadata_valid = self.get_metadata(dataset_folder, feature_folder, 'valid', three_train_pianos=False)
        self.metadata_test = self.get_metadata(dataset_folder, feature_folder, 'test', three_train_pianos=False)

    def prepare_data(self):
        # Override prepare_data
        for metadata in [self.metadata_train, 
                        self.metadata_valid, 
                        self.metadata_test]:
            self.prepare_spectrograms(metadata, self.spectrogram_setting)
            self.prepare_pianorolls(metadata)
            
    def get_train_dataset(self):
        return PianorollTranscriptionDataset(self.metadata_train, self.spectrogram_setting)

    def get_valid_dataset(self):
        return PianorollTranscriptionDataset(self.metadata_valid, self.spectrogram_setting)

    def get_test_dataset(self):
        return PianorollTranscriptionDataset(self.metadata_test, self.spectrogram_setting)
