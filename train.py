import warnings
warnings.filterwarnings('ignore')
import argparse
import pytorch_lightning as pl

from audio2score.data.datamodule import PianorollTranscriptionDataModule, Audio2ScoreTranscriptionDataModule, JointTranscriptionDataModule
from audio2score.models.models import PianorollTranscriptionModel, Audio2ScoreTranscriptionModel, JointTranscriptionModel
from audio2score.settings import SpectrogramSetting


def train(args, spectrogram_setting):
    # get datamodule, model and experiment_name
    if args.task == 'audio2pr':
        datamodule = PianorollTranscriptionDataModule(spectrogram_setting=spectrogram_setting, 
                                dataset_folder=args.dataset_folder, 
                                feature_folder=args.feature_folder)
        model = PianorollTranscriptionModel(in_channels=spectrogram_setting.channels,
                                freq_bins=spectrogram_setting.freq_bins)
        experiment_name = f'audio2pr-{spectrogram_setting.to_string()}'

    elif args.task == 'audio2score':
        datamodule = Audio2ScoreTranscriptionDataModule(spectrogram_setting=spectrogram_setting,
                                score_type=args.score_type,
                                dataset_folder=args.dataset_folder,
                                feature_folder=args.feature_folder)
        model = Audio2ScoreTranscriptionModel(score_type=args.score_type)
        experiment_name = f'audio2score-{args.score_type}'

    elif args.task == 'joint':
        datamodule = JointTranscriptionDataModule(spectrogram_setting=spectrogram_setting,
                                score_type=args.score_type,
                                dataset_folder=args.dataset_folder,
                                feature_folder=args.feature_folder)
        model = JointTranscriptionModel(score_type=args.score_type)
        experiment_name = f'joint-{args.score_type}'

    # get trainer and train
    logger = pl.loggers.TensorBoardLogger('tensorboard_logs', 
                                        name=experiment_name, 
                                        default_hp_metric=False)
    trainer = pl.Trainer(gpus=1, 
                        logger=logger,
                        log_every_n_steps=20,
                        reload_dataloaders_every_epoch=True,
                        auto_select_gpus=True)
    trainer.fit(model, datamodule=datamodule)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='task',
                        help='select from [audio2pr] [audio2score]')
    
    # pianoroll transcription parser
    parser_audio2pr = subparsers.add_parser('audio2pr')

    # score transcription parser
    parser_audio2score = subparsers.add_parser('audio2score')
    parser_audio2score.add_argument('--score_type',
                        type=str,
                        help='select from [Reshaped] [LilyPond]')

    # joint trasncription parser
    parser_joint = subparsers.add_parser('joint')
    parser_joint.add_argument('--score_type',
                        type=str,
                        help='select from [Reshaped] [LilyPond]')

    # common parser arguments
    for subparser in [parser_audio2pr, parser_audio2score, parser_joint]:
        subparser.add_argument('--dataset_folder',
                                type=str,
                                help='dataset path')
        subparser.add_argument('--feature_folder',
                                type=str,
                                help='folder to save pre-calculated features')

    args = parser.parse_args()

    # train
    spectrogram_setting = SpectrogramSetting()
    train(args, spectrogram_setting)
