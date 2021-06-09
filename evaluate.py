import warnings
warnings.filterwarnings('ignore')
import argparse
import pytorch_lightning as pl

from audio2score.data.datamodule import PianorollTranscriptionDataModule
from audio2score.models.models import PianorollTranscriptionModel
from audio2score.settings import SpectrogramSetting


def test_audio2pr(args):
    spectrogram_settings = SpectrogramSetting()

    datamodule = PianorollTranscriptionDataModule(spectrogram_settings, 
                            args.dataset_folder, 
                            args.feature_folder)
    model = PianorollTranscriptionModel.load_from_checkpoint(args.model_checkpoint, 
                            in_channels=spectrogram_settings.channels,
                            freq_bins=spectrogram_settings.freq_bins)
    trainer = pl.Trainer(gpus=[2], reload_dataloaders_every_epoch=True)
    trainer.test(model, datamodule=datamodule)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='task',
                        help='select from [audio2pr] [audio2score]')
    
    parser_audio2pr = subparsers.add_parser('audio2pr')
    parser_audio2pr.add_argument('--dataset_folder',
                        type=str,
                        help='dataset path')
    parser_audio2pr.add_argument('--feature_folder',
                        type=str,
                        help='folder to save pre-calculated features')
    parser_audio2pr.add_argument('--model_checkpoint',
                        type=str,
                        help='pre-trained model checkpoint')

    args = parser.parse_args()

    if args.task == 'audio2pr':
        test_audio2pr(args)