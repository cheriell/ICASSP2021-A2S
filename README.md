# ICASSP-A2S

Accompanying code for paper

- Lele Liu, Veronica Morfi and Emmanouil Benetos, "Joint Multi-pitch Detection and Score Transcription for Polyphonic Piano Music", IEEE International Conference on Acoustics, Speech and Signal Processing, Canada, Jun 2021.

## Environment setup

This project uses pytorch and python 3, it's recommended you first create a python 3 virtual environment

    python3 -m venv ICASSP2021-A2S-ENV
    source ICASSP2021-A2S-ENV/bin/activate
    git clone https://github.com/cheriell/ICASSP-A2S

Run the following command to install the python packages required in this project

    cd ICASSP-A2S
    pip install -r requirements.txt

In this project, we use the [MV2H metric](https://github.com/apmcleod/MV2H) (McLeod et al., 2018) for Audio-to-Score transcription, please refer to the original github repository for installation details.

Before running, enable shell scripts

    chmod +x runme.sh
    chmod +x audio2score/utilities/evaluate_midi_mv2h.sh 

## Data

We use the `MuseSyn` dataset for our experiments. To download the dataset, please refer to: [MuseSyn: A dataset for complete automatic piano music transcription research](https://zenodo.org/record/4527460).

## Running

Please refer to `runme.sh` for examples of relavant commands for model training and evaluation. Before you run the script, please remember to change the relavant path on top of the shell script to where you save your datasets, features, models and MV2H metric. Uncomment commands to run the script.

### Multi-pitch detection with different time-frequency representations

To train a multi-pitch detection model, use the following command.

    python train.py audio2pr --dataset_folder path/to/MuseSyn --feature_folder path/to/MuseSyn/features

For evaluation, run

    python test.py audio2pr --dataset_folder path/to/MuseSyn --feature_folder path/to/MuseSyn/features --model_checkpoint model_checkpoint_file

For different time-frequency representations, please modify the spectrogram settings in file `audio2score/settings.py`. A list of tested spectrogram settings in the paper are given in `metadata/spectrogram_settings.csv`.

### Audio-to-Score transcription with different score representations

To train a single-task audio-to-score transcription model, run

    python train.py audio2score --score_type score_type --dataset_folder path/to/MuseSyn --feature_folder path/to/MuseSyn/features

`score_type` should be `Reshaped` or `LilyPond`.

For evaluation, run

    python test.py audio2score --score_type score_type --MV2H_path path/to/MV2H/bin --dataset_folder path/to/MuseSyn --feature_folder path/to/MuseSyn/features --model_checkpoint model_checkpoint_file

### Joint Transcription

To train a joint transcrition model, run

    python train.py joint --score_type "Reshaped" --dataset_folder path/to/MuseSyn --feature_folder path/to/MuseSyn/features

For evaluation, run

    python test.py joint --score_type "Reshaped" --MV2H_path path/to/MV2H/bin --dataset_folder path/to/MuseSyn --feature_folder path/to/MuseSyn/features --model_checkpoint model_checkpoint_file


# NOTES

remove transformer related python files