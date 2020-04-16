#!/usr/bin/env python3
import ddsp.training
import gin
from matplotlib import pyplot as plt
import numpy as np
import os
from pydub import AudioSegment
import librosa
import base64
import io
import tempfile

import tensorflow as tf
from scipy.io import wavfile
import sys

DEFAULT_SAMPLE_RATE=16000

def audio_bytes_to_np(wav_data,
                      sample_rate=DEFAULT_SAMPLE_RATE,
                      normalize_db=0.1):
    """Convert audio file data (in bytes) into a numpy array.

    Saves to a tempfile and loads with librosa.
    Args:
    wav_data: A byte stream of audio data.
    sample_rate: Resample recorded audio to this sample rate.
    normalize_db: Normalize the audio to this many decibels. Set to None to skip
    normalization step.

    Returns:
    An array of the recorded audio at sample_rate.
    """
    # Parse and normalize the audio.
    audio = AudioSegment.from_file(io.BytesIO(wav_data), format="wave")
    audio.remove_dc_offset()
    if normalize_db is not None:
        audio.normalize(headroom=normalize_db)
    # Save to tempfile and load with librosa.
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_wav_file:
        fname = temp_wav_file.name
        audio.export(fname, format='wav')
        audio_np, unused_sr = librosa.load(fname, sr=sample_rate)
    return audio_np.astype(np.float32)

def outputToWav(rawOutput, resultPath, sample_rate=DEFAULT_SAMPLE_RATE):
    # If batched, take first element.
    if len(rawOutput.shape) == 2:
        rawOutput = rawOutput[0]

    normalizer = float(np.iinfo(np.int16).max)
    array_of_ints = np.array(
        np.asarray(rawOutput) * normalizer, dtype=np.int16)
    wavfile.write(resultPath, sample_rate, array_of_ints)

def main():
    AUDIO_PATH = sys.argv[1]
    MODEL_DIR = sys.argv[2]
    RESULT_PATH = sys.argv[3]
    print(sys.argv)

    audio = audio_bytes_to_np(open(AUDIO_PATH, "rb").read(),
                                   sample_rate=DEFAULT_SAMPLE_RATE,
                                   normalize_db=None)
    audio = audio[np.newaxis, :]
    audio_features = ddsp.training.eval_util.compute_audio_features(audio)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)

    # Parse the gin config.
    gin_file = os.path.join(MODEL_DIR, 'operative_config-0.gin')
    gin.parse_config_file(gin_file)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Additive.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    print("===Trained model===")
    print("Time Steps", time_steps_train)
    print("Samples", n_samples_train)
    print("Hop Size", hop_size)
    print("\n===Resynthesis===")
    print("Time Steps", time_steps)
    print("Samples", n_samples)
    print('')

    gin_params = [
        'Additive.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'DefaultPreprocessor.time_steps = {}'.format(time_steps),
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)


    # Trim all input vectors to correct lengths 
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
        audio_features['audio'] = audio_features['audio'][:, :n_samples]

    # Load model
    model = ddsp.training.models.Autoencoder()
    model.restore(MODEL_DIR)

    # Resynthesize audio.
    audio_gen = model(audio_features, training=False)
    outputToWav(audio_gen, RESULT_PATH)

if __name__ == "__main__":
    main()