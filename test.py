import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import pickle
import pandas as pd
import numpy as np
import music21 as m21
import pytorch_lightning as pl

from audio2score.data.datamodule import PianorollTranscriptionDataModule, Audio2ScoreTranscriptionDataModule, JointTranscriptionDataModule
from audio2score.models.models import PianorollTranscriptionModel
from audio2score.settings import SpectrogramSetting
from audio2score.transcribers import Audio2ScoreTranscriber
from audio2score.scores.helper import combine_bars_from_score_reshaped, combine_bars_from_score_lilypond
from audio2score.scores.lilypondscore import ScoreLilyPond
from audio2score.utilities.utils import mkdir
from audio2score.utilities.evaluation_utils import Eval


output_path = 'outputs/'  # folder to save predicted scores and evaluation results
mkdir(output_path)


def test_audio2pr(args):
    spectrogram_setting = SpectrogramSetting()

    datamodule = PianorollTranscriptionDataModule(spectrogram_setting, 
                            args.dataset_folder, 
                            args.feature_folder)
    model = PianorollTranscriptionModel.load_from_checkpoint(args.model_checkpoint, 
                            in_channels=spectrogram_setting.channels,
                            freq_bins=spectrogram_setting.freq_bins)
    trainer = pl.Trainer(gpus=1,
                        reload_dataloaders_every_epoch=True,
                        auto_select_gpus=True)
    trainer.test(model, datamodule=datamodule)


def test_audio2score(args):
    spectrogram_setting = SpectrogramSetting()

    if args.task == 'audio2score':
        DataModule = Audio2ScoreTranscriptionDataModule
    elif args.task == 'joint':
        DataModule = JointTranscriptionDataModule
    datamodule = DataModule(spectrogram_setting,
                            args.score_type,
                            args.dataset_folder,
                            args.feature_folder)
    transcriber = Audio2ScoreTranscriber(model_checkpoint=args.model_checkpoint,
                            score_type=args.score_type,
                            model_type=args.task,
                            gpu=0)
                            
    evaluation_results = pd.DataFrame(columns=['wer-right', 'wer-left', 'wer',
                            'mv2h-multipitch', 'mv2h-voice', 'mv2h-meter', 'mv2h-value', 'mv2h'])

    for i, row in datamodule.metadata_test.iterrows():
        print(f'Evaluating test set {i+1}/{len(datamodule.metadata_test)}')
        if i == 2: break
        
        downbeats_file = row['downbeats_file']
        spectrograms_folder = row['spectrograms_folder']
        score_folder = row['score_reshaped_folder'] if args.score_type == 'Reshaped' \
                                                    else row['score_lilypond_folder']
        piano = row['piano']
        name = row['name']

        # get ground truth downbeats
        downbeats = pickle.load(open(downbeats_file, 'rb'))

        # get predicted score and target score by bar
        score_right_list_pred, score_left_list_pred = [], []
        score_right_list_targ, score_left_list_targ = [], []
        quarter_lengths = []
        key_signatures, time_signatures = [], []

        for bar_index in range(len(downbeats)-2):  # ignore final bar
            print(f'\ttranscribing bar {bar_index+1}/{len(downbeats)-2}', end='\r')
            if bar_index == 2: break

            # predicted score
            spectrogram = pickle.load(open(os.path.join(spectrograms_folder, 
                                            f'{spectrogram_setting.to_string()}.pkl') + \
                                            f'.{bar_index}.pkl', 'rb'))
            score_right_pred, score_left_pred = transcriber.transcribe_one_bar_from_spectrogram(spectrogram)
            # target score
            score_right_targ, score_left_targ = pickle.load(open(os.path.join(score_folder, f'{bar_index}.pkl'), 'rb'))
            # quarter_lengths
            quarter_length = score_right_targ.to_m21().quarterLength

            score_right_list_pred.append(score_right_pred)
            score_left_list_pred.append(score_left_pred)
            score_right_list_targ.append(score_right_targ)
            score_left_list_targ.append(score_left_targ)
            quarter_lengths.append(quarter_length)
            key_signatures.append(m21.key.KeySignature(sharps=0))
            time_signatures.append(m21.meter.TimeSignature(f'{quarter_length}/4'))

        print('\n\tgetting predicted score and evaluate')
        if args.score_type == 'Reshaped':
            score_targ = combine_bars_from_score_reshaped(key_signatures, time_signatures, 
                                    score_right_list_targ, score_left_list_targ, quarter_lengths)
            score_pred = combine_bars_from_score_reshaped(key_signatures, time_signatures, 
                                    score_right_list_pred, score_left_list_pred, quarter_lengths)
        elif args.score_type == 'LilyPond':
            score_targ = combine_bars_from_score_lilypond(key_signatures, time_signatures, 
                                    score_right_list_targ, score_left_list_targ, quarter_lengths)
            score_pred = combine_bars_from_score_lilypond(key_signatures, time_signatures, 
                                    score_right_list_pred, score_left_list_pred, quarter_lengths)

        # save target and predicted scores
        score_targ_file = os.path.join(output_path, args.score_type, piano, f'{name}_targ.mid')
        score_pred_file = os.path.join(output_path, args.score_type, piano, f'{name}_pred.mid')
        mkdir(os.path.split(score_targ_file)[0])
        score_targ.write('midi', score_targ_file)
        score_pred.write('midi', score_pred_file)
        
        # evaluate
        wer_right, wer_left = evaluate_word_error_rate(score_right_list_pred, score_left_list_pred,
                                                    score_right_list_targ, score_left_list_targ, args.score_type)
        mv2h_result = Eval.mv2h_evaluation(score_targ_file, score_pred_file, args.MV2H_path)
        
        # update evaluation results
        evaluation_results.loc[i] = [wer_right, wer_left, np.mean([wer_right, wer_left]),
                    mv2h_result['Multi-pitch'], mv2h_result['Voice'], mv2h_result['Meter'], mv2h_result['Value'],
                    np.mean([mv2h_result['Multi-pitch'], mv2h_result['Voice'], mv2h_result['Meter'], mv2h_result['Value']])]

    print(np.mean(evaluation_results))
    evaluation_results_file = os.path.join(output_path, args.score_type, 'evaluation_results.csv')
    evaluation_results.to_csv(evaluation_results_file, index=False)


def evaluate_word_error_rate(score_right_list_pred,
                        score_left_list_pred,
                        score_right_list_targ,
                        score_left_list_targ,
                        score_type):

    # convert to ScoreLilyPond
    if score_type == 'Reshaped':
        score_right_list_pred = [ScoreLilyPond.from_m21(score.to_m21()) for score in score_right_list_pred]
        score_left_list_pred = [ScoreLilyPond.from_m21(score.to_m21()) for score in score_left_list_pred]
        score_right_list_targ = [ScoreLilyPond.from_m21(score.to_m21()) for score in score_right_list_targ]
        score_left_list_targ = [ScoreLilyPond.from_m21(score.to_m21()) for score in score_left_list_targ]

    wer_right = Eval.wer_evaluation(target=' | '.join([score.to_string() for score in score_right_list_targ]),
                                    output=' | '.join([score.to_string() for score in score_right_list_pred]))
    wer_left = Eval.wer_evaluation(target=' | '.join([score.to_string() for score in score_left_list_targ]),
                                    output=' | '.join([score.to_string() for score in score_left_list_pred]))

    return wer_right, wer_left


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='task',
                        help='select from [audio2pr] [audio2score] [joint]')
    
    # pianoroll transcription parser
    parser_audio2pr = subparsers.add_parser('audio2pr')

    # audio2score transcription parser
    parser_audio2score = subparsers.add_parser('audio2score')
    parser_audio2score.add_argument('--score_type',
                        type=str,
                        help='select from [Reshaped] [LilyPond]')
    parser_audio2score.add_argument('--MV2H_path',
                        type=str,
                        help='path to the MV2H metric, e.g. "path/to/MV2H/bin"')
    
    # joint transcription parser
    parser_joint = subparsers.add_parser('joint')
    parser_joint.add_argument('--score_type',
                        type=str,
                        help='select from [Reshaped] [LilyPond]')
    parser_joint.add_argument('--MV2H_path',
                        type=str,
                        help='path to the MV2H metric, e.g. "path/to/MV2H/bin"')

    # common parser arguments
    for subparser in [parser_audio2pr, parser_audio2score, parser_joint]:
        subparser.add_argument('--dataset_folder',
                                type=str,
                                help='dataset path')
        subparser.add_argument('--feature_folder',
                                type=str,
                                help='folder to save pre-calculated features')
        subparser.add_argument('--model_checkpoint',
                                type=str,
                                help='pre-trained model checkpoint')

    args = parser.parse_args()

    # evaluate
    if args.task == 'audio2pr':
        test_audio2pr(args)
    elif args.task in ['audio2score', 'joint']:
        test_audio2score(args)