# ICASSP-A2S

Accompanying code for paper

Lele Liu, Veronica Morfi and Emmanouil Benetos, "Joint Multi-pitch Detection and Score Transcription for Polyphonic Piano Music", IEEE International Conference on Acoustics, Speech and Signal Processing, Canada, Jun 2021.

## Environment

This project uses pytorch and python 3, it's recommended you first create a python 3 virtual environment

    $ python3 -m venv ICASSP2021-A2S-ENV
    $ cd ICASSP2021-A2S-ENV
    $ source bin/activate
    $ git clone https://github.com/cheriell/ICASSP-A2S

Install the python packages required in this project

    $ pip install -r requirements.txt

## Data

We use the `MuseSyn` dataset for our experiments. To download the dataset, please refer to: [MuseSyn: A dataset for complete automatic piano music transcription research](https://zenodo.org/record/4527460).

## Running

Please refer to `runme.sh` for relavant commands for model training and evaluation. Before you run the script, please remember to change the relavant path on top of the shell script to where you save your dataset/features/models. Uncomment commands to run the script.

