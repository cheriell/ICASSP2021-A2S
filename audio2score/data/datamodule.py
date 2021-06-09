

from typing import Any

from audio2score.data.basedatamodule import BaseDataModule
from audio2score.data.prtransdataset import PianorollTranscriptionDataset

class PianorollTranscriptionDataModule(BaseDataModule):
    """DataModule.

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
        self.metadata_train = self.get_metadata(dataset_folder, feature_folder, 'train')
        self.metadata_valid = self.get_metadata(dataset_folder, feature_folder, 'valid')
        self.metadata_test = self.get_metadata(dataset_folder, feature_folder, 'test')

    def prepare_data(self):
        """Override prepare_data()"""
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
