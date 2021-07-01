#!/usr/bin/env bash

#################################################
### paths
### REMEMBER to change these path to where you save the dataset/feature/model!
DATASET_FOLDER="dataset/folder"
FEATURE_FOLDER="feature/folder"
MODEL_CHECKPOINT="path/to/model_checkpoint.ckpt"


#################################################
### Experiment 1. test different time-frequency representations
###     For different spectrograms (and their parameters), change SpectrogramSetting in 'audio2score/settings'.
###     A list of spectrogram settings tested in experiments are in 'metadata/spectrogram_settings.csv'

# # train pianoroll transcription model
# python train.py audio2pr --dataset_folder $DATASET_FOLDER --feature_folder $FEATURE_FOLDER

# # test pianoroll transcription model
# python evaluate.py audio2pr --dataset_folder $DATASET_FOLDER --feature_folder $FEATURE_FOLDER --model_checkpoint $MODEL_CHECKPOINT


#################################################
### Experiment 2. test different score representations
###
### TBC in July.

