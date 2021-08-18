#!/usr/bin/env bash

#################################################
### EXAMPLE SHELL SCRIPT

# REMEMBER to change these paths to where you save the dataset/feature!
# ALSO REMEMBER to change the paths to the model checkpoints before you run evaluation code.

DATASET_FOLDER="/import/research_c4dm/ll307/MuseSyn"
FEATURE_FOLDER="/import/c4dm-datasets/A2S_transcription/MuseSyn"
MV2H_PATH="/import/c4dm-datasets/A2S_transcription/working/MV2H/bin"


#################################################
### Experiment 1. test different time-frequency representations
###     For different spectrograms (and their parameters), change SpectrogramSetting in 'audio2score/settings.py'.
###     A list of spectrogram settings tested in experiments are in 'metadata/spectrogram_settings.csv'.

# train pianoroll transcription model
python train.py audio2pr --dataset_folder $DATASET_FOLDER --feature_folder $FEATURE_FOLDER

# evaluate pianoroll transcription model
MODEL_CHECKPOINT="path/to/model_checkpoint.ckpt"
python test.py audio2pr --dataset_folder $DATASET_FOLDER --feature_folder $FEATURE_FOLDER --model_checkpoint $MODEL_CHECKPOINT


#################################################
### Experiment 2. test different score representations
###     Use the best spectrogram setting (the VQT one) as described in the paper.
###     Specify SCORE_TYPE to be "Reshaped" or "LilyPond".

SCORE_TYPE="Reshaped"  # select from ["Reshaped", "LilyPond"]

# train audio2score transcription model
python train.py audio2score --score_type $SCORE_TYPE --dataset_folder $DATASET_FOLDER --feature_folder $FEATURE_FOLDER

# evaluate audio2score transcription model
MODEL_CHECKPOINT="path/to/model_checkpoint.ckpt"
python test.py audio2score --score_type $SCORE_TYPE --MV2H_path $MV2H_PATH --dataset_folder $DATASET_FOLDER --feature_folder $FEATURE_FOLDER --model_checkpoint $MODEL_CHECKPOINT


##################################################
### Joint multi-pitch detection and score transcription
###     Use the best spectrogram setting (the VQT one) as described in the paper.

# train joint transcription model
python train.py joint --score_type "Reshaped" --dataset_folder $DATASET_FOLDER --feature_folder $FEATURE_FOLDER

# evaluate joint transcription model
MODEL_CHECKPOINT="path/to/model_checkpoint.ckpt"
python test.py joint --score_type "Reshaped" --MV2H_path $MV2H_PATH --dataset_folder $DATASET_FOLDER --feature_folder $FEATURE_FOLDER --model_checkpoint $MODEL_CHECKPOINT