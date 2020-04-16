# Audio samples
All audio samples referenced in the project document can be accessed via [our GitHub](https://github.com/TalSk/cs236781-project) under `src/results` directory.

# Project sources structure 

In this subsection we will detail the structure of our code in the `src/` directory: 

- The directory `ddsp/` contains the code of the DDSP library along with its preprocessor.  No changes were done to the original code. 

- `TasNet/` contains the code of the MCTN, where the main differences are changes to the decoder, some hyperparameters and general refactoring of the original code (that was originally written for TensorFlow v1). 

- `results/` contains the output of all experiments held, with its subdirectories and file names detailing the specific experiment. 

- `training/` contains all scripts that were used in the pre-training of the models. 

	* `our_solo_instrumnet.gin` – a base gin file for the DDSP’s autoencoder training. 

	* `py-sbatch_{bass, drums, vocals}` – bash files that were used for training. They ran DDSP autoencoder training using the gin file above, with a path to the relevant .tfrecord file in our dataset directory. 

	* `py-sbatch_{bass, drums, vocals}_resynth` – bash files that were used to resynthesize the relevant instrument using the resynth.py evaluation file. 

- `resynth.py` that runs an evaluation by passing a single instrument audio file through the pre-trained DDSP autoencoder. 

- `eval-mctn.py` that runs an evaluation by passing a given audio file through the pre-trained MCTN model. 

- `eval_ddsp.py` that runs an evaluation by passing extracted fundamental frequencies and loudness of each source (for example, those extracted by the MCTN model) through each of the pre-trained DDSP autoencoders. 

- `eval_mctn_f0_loudness.py` that evaluates audio features of the MCTN output.

- `eval_loudness_f0_spectral_sdr` that evaluates audio features of audio files using the DDSP autoencoders.

# Installation
Run the following commands to create a conda environment called "ddsp" capable of running our code:

`conda create --name ddsp python=3.7
conda activate ddsp
conda install jupyterlab
conda install -c conda-forge librosa
conda install -c conda-forge tqdm`

## Environment installation

For windows: `conda install -c msys2 m2w64-libsndfile`

For linux: `sudo apt-get install libsndfile-dev` or `conda install -c conda-forge libsndfile`

And then: 
```
pip install --upgrade pip
pip install apache_beam
sudo apt-get install ffmpeg libavcodec-extra
pip install --upgrade ddsp
```

To run the code on a GPU using TensorFlow, drivers from Nvidia [need to be installed](https://developer.nvidia.com/rdp/cudnn-download).

# Preprocessing & training of the DDSP
The dataset was taken from the [DSD100 dataset](https://www.sisec17.audiolabs-erlangen.de/#/dataset) (Specifically, James May, tracks 20, 21, 70 and 71).

To prepare the tfrecord file, use the DDSP utility file `prepare_tfrecord.py` which can be run instantly after installing DDSP.

To train the DDSP autoencoder the utility file `ddsp_run.py` which can also be run instantly after installing DDSP.
We used `src/training/our_solo_instrumnet.gin` as the GIN configuration file.

For example, this is the command line we run on the server for the vocal DDSP decoder training:
```
ddsp_run \
  --mode=train \
  --model_dir=/home/amitz/cs236781/project/trained/vocals/ \
  --gin_file=/home/amitz/cs236781/project/our_solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='/home/amitz/cs236781/project/Dataset/Training/vocals/vocals.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr
```

# Preprocessing & training of the MCTN
The data is expected be in the following structure:
```
dataset/s1/file1.wav
dataset/s2/file1.wav
dataset/s3/file1.wav
dataset/mix/file1.wav
```

Note that the file in every directory that belongs to the same song must have the same filename!

Then, by running `python ./tasnet_main.py -dd <PATH TO PREPROCESSED DATASET DIR>`
the script will create the preprocessed data files if they don't exist, and then proceed to training the model.

# Running the entire model
This is separated to two parts, the first is the run the MCTN model to extract audio features:
`python eval_mctn.py`

This script expects the MCTN checkpoint to be in `./mctn_ckpt` directory and the mixed input tfrecord file to in `./mctn_data/infer.tfr` path.
It then writes output - a numpy array - to the `./results` directory with the file names `{vocals, bass, drums}_{f0_hz, loudness_db}.npy`.

Then, one can pass these extracted values through the pre-trained DDSP auto-encoders by running
`python eval_ddsp.py`

This script expects the autoencoders checkpoints to be in the directories `./ddsp_{vocals, bass, drums}_ckpt` containing a gin file with the expected names as can seen in the script.
It will then output the resynthesized, seperated audio WAV files to the `./results/Full model - test` directory.