#!/usr/bin/env bash

#################################################
### paths
### REMEMBER to change these path to where you save the dataset/feature/model!
DATASET_FOLDER="/import/research_c4dm/ll307/MuseSyn"
FEATURE_FOLDER="/import/c4dm-datasets/A2S_transcription/MuseSyn"
MODEL_CHECKPOINT="/import/c4dm-datasets/A2S_transcription/working/icassp-a2s-draft/lightning_logs/version_14/checkpoints/epoch=4-valid_loss=208.58.ckpt"


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
### TBC in this month (June 2021)

