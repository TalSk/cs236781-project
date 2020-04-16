import argparse
import numpy as np
import librosa
from ddsp import spectral_ops

import tensorflow.compat.v1 as tf


def _calc_sdr(s_hat, s):
    s_target = (tf.tensordot(s, s_hat) * s) / tf.norm(s)
    e_noice = s_hat - s_target
    
    return 10 * tf.log(tf.norm(s_target) / tf.norm(e_noice)) / tf.log(10.0)



def main():
    parser = argparse.ArgumentParser(description='Evaluate loudness and basic frequency (F0) L1 difference between a synthesized wav file to its original wav file')
    parser.add_argument('-sf', '--synthesized_file', type=str)
    parser.add_argument('-of', '--original_file', type=str)
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000)
    parser.add_argument('-fr', '--frame_rate', type=int, default=250)
    parser.add_argument('-s', '--include_spectral', type=int, default=1)
    parser.add_argument('-f0', '--include_f0', type=int, default=1)
    parser.add_argument('-ld', '--include_ld', type=int, default=1)
    parser.add_argument('-sdr', '--include_sdr', type=int, default=1)
    
    args = parser.parse_args()
    
    synth_audio, _ = librosa.load(args.synthesized_file, args.sample_rate)
    original_audio, _ = librosa.load(args.original_file, args.sample_rate)

    synth_audio_samples = synth_audio.shape[0]
    original_audio_samples = original_audio.shape[0]

    if synth_audio_samples < original_audio_samples:
        print(f"Trimming original audio samples from {original_audio_samples} to {synth_audio_samples}")
        original_audio_samples = synth_audio_samples
        original_audio = original_audio[:original_audio_samples]

    elif original_audio_samples < synth_audio_samples:
        print(f"Trimming synthesized audio samples from {synth_audio_samples} to {original_audio_samples}")
        synth_audio_samples = original_audio_samples
        synth_audio = synth_audio[:synth_audio_samples]

    if args.include_sdr:
        print(f"SDR: {_calc_sdr(synth_audio, original_audio)}")

    if args.include_f0:
        print("Calculating F0 for synthesized audio")
        synth_f0 = spectral_ops.compute_f0(synth_audio, args.sample_rate, args.frame_rate)[0]

        print("Calculating F0 for original audio")
        original_f0 = spectral_ops.compute_f0(original_audio, args.sample_rate, args.frame_rate)[0]
        f0_l1 = np.mean(abs(synth_f0 - original_f0))
        print(f"Average F0 L1: {f0_l1}")

    if args.include_ld:
        print("Calculating loudness for synthesized audio")
        synth_loudness = spectral_ops.compute_loudness(synth_audio, args.sample_rate, args.frame_rate)

        print("Calculating loudness for original audio")
        original_loudness = spectral_ops.compute_loudness(original_audio, args.sample_rate, args.frame_rate)

        loudness_l1 = np.mean(abs(synth_loudness - original_loudness))
        print(f"Average Loudness L1: {loudness_l1}")

    if args.include_spectral:
        from ddsp import losses
        loss_obj = losses.SpectralLoss(mag_weight=1.0, logmag_weight=1.0)
        spectral_loss = loss_obj(synth_audio, original_audio)
        print(f"Average Multi-scale spectrogram loss: {spectral_loss}")

if __name__ == '__main__':
    main()