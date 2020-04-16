import argparse
import os
import logging
import sys
import gin
import ddsp.training

import numpy as np

from scipy.io import wavfile


def outputToWav(rawOutput, resultPath, sample_rate=16000):
    if len(rawOutput.shape) == 2:
        rawOutput = rawOutput[0]

    normalizer = float(np.iinfo(np.int16).max)
    array_of_ints = np.array(
        np.asarray(rawOutput) * normalizer, dtype=np.int16)
    wavfile.write(resultPath, sample_rate, array_of_ints)

def main():
    parser = argparse.ArgumentParser(description='Audio Separation Evaluation - DDSP part')
    parser.add_argument('-vod', '--ddsp_vocals_ckpt_dir', type=str, default='./ddsp_vocals_ckpt')
    parser.add_argument('-bad', '--ddsp_bass_ckpt_dir', type=str, default='./ddsp_bass_ckpt')
    parser.add_argument('-drd', '--ddsp_drums_ckpt_dir', type=str, default='./ddsp_drums_ckpt')
    parser.add_argument('-ld', '--log_dir', type=str, default='./ddsp_log')
    parser.add_argument('-rd', '--results_dir', type=str, default='./results')

    args = parser.parse_args()
    args.log_file = os.path.join(args.log_dir, 'log.txt')

    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    ddsp_vocals_gin_path = os.path.join(args.ddsp_vocals_ckpt_dir, "operative_config-13200.gin")
    ddsp_bass_gin_path = os.path.join(args.ddsp_bass_ckpt_dir, "operative_config-11100.gin")
    ddsp_drums_gin_path = os.path.join(args.ddsp_drums_ckpt_dir, "operative_config-0.gin")

    # Prepare input from files
    logger.info("Loading data from mctn files")
    audio_features_vocals = {
        'loudness_db': np.load(os.path.join(args.results_dir, "vocals_loudness_db.npy"), allow_pickle=False),
        'f0_hz': np.load(os.path.join(args.results_dir, "vocals_f0_hz.npy"), allow_pickle=False),
    }
    audio_features_bass = {
        'loudness_db': np.load(os.path.join(args.results_dir, "bass_loudness_db.npy"), allow_pickle=False),
        'f0_hz': np.load(os.path.join(args.results_dir, "bass_f0_hz.npy"), allow_pickle=False),
    }
    audio_features_drums = {
        'loudness_db': np.load(os.path.join(args.results_dir, "drums_loudness_db.npy"), allow_pickle=False),
        'f0_hz': np.load(os.path.join(args.results_dir, "drums_f0_hz.npy"), allow_pickle=False),
    }

    # Load each autoencoder, and resynthesize
    logger.info("====Resynthesizing vocals===")
    gin.clear_config()
    gin.parse_config_file(ddsp_vocals_gin_path)

    time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Additive.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = audio_features_vocals['f0_hz'].shape[1]  # audio_features_vocals['f0_hz'].shape[1]  # int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        'Additive.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'DefaultPreprocessor.time_steps = {}'.format(time_steps),
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    vocals_ddsp = ddsp.training.models.Autoencoder(name="vocals_ae")
    vocals_ddsp.restore(args.ddsp_vocals_ckpt_dir)
    vocals_gen = vocals_ddsp(audio_features_vocals, training=False)

    print("====Resynthesizing bass===")
    gin.clear_config()
    gin.parse_config_file(ddsp_bass_gin_path)

    time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Additive.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = audio_features_bass['f0_hz'].shape[1]  # int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        'Additive.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'DefaultPreprocessor.time_steps = {}'.format(time_steps),
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    bass_ddsp = ddsp.training.models.Autoencoder(name="bass_ae")
    bass_ddsp.restore(args.ddsp_bass_ckpt_dir)
    bass_gen = vocals_ddsp(audio_features_bass, training=False)

    print("====Resynthesizing drums===")
    gin.clear_config()
    gin.parse_config_file(ddsp_drums_gin_path)

    time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Additive.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = audio_features_drums['f0_hz'].shape[1]  # int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        'Additive.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'DefaultPreprocessor.time_steps = {}'.format(time_steps),
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    drums_ddsp = ddsp.training.models.Autoencoder(name="drums_ae")
    drums_ddsp.restore(args.ddsp_drums_ckpt_dir)
    drums_gen = vocals_ddsp(audio_features_drums, training=False)

    print(vocals_gen.shape)
    print(bass_gen.shape)
    print(drums_gen.shape)

    # TODO: Add metrics evaluation.

    logger.info("Saving to wav")
    outputToWav(rawOutput=vocals_gen, resultPath=os.path.join(args.results_dir, "sep_vocals.wav"))
    outputToWav(rawOutput=bass_gen, resultPath=os.path.join(args.results_dir, "sep_bass.wav"))
    outputToWav(rawOutput=drums_gen, resultPath=os.path.join(args.results_dir, "sep_drums.wav"))
    logger.info("Done")


if __name__ == "__main__":
    main()