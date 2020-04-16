import argparse
import numpy as np
import librosa
from ddsp import spectral_ops

def main():
    parser = argparse.ArgumentParser(description='Evaluate loudness and basic frequency (F0) L1 difference between a synthesized wav file to its original wav file')
    parser.add_argument('-sf', '--synthesized_file', type=str)
    parser.add_argument('-of', '--original_file', type=str)
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000)
    parser.add_argument('-fr', '--frame_rate', type=int, default=250)
    args = parser.parse_args()
    
    synth_audio, _ = librosa.load(args.synthesized_file, args.sample_rate)
    original_audio, _ = librosa.load(args.original_file, args.sample_rate)

    synth_audio_samples = synth_audio.shape[0]
    original_audio_samples = original_audio.shape[0]

    if synth_audio_samples < original_audio_samples:
        print(f"Trimming original audio samples from {original_audio_samples} to {synth_audio_samples}")
        original_audio_samples = synth_audio_samples
        original_audio = original_audio[:original_audio_samples]

    print("Calculating F0 for synthesized audio")
    synth_f0 = spectral_ops.compute_f0(synth_audio, args.sample_rate, args.frame_rate)[0]
    print("Calculating loudness for synthesized audio")
    synth_loudness = spectral_ops.compute_loudness(synth_audio, args.sample_rate, args.frame_rate)

    avg_synth_f0 = np.mean(synth_f0)
    avg_synth_loudness = np.mean(synth_loudness)
    print(f"Average F0 of synth file: {avg_synth_f0}")
    print(f"Average Loudness of synth file: {avg_synth_loudness}")

    print("Calculating F0 for original audio")
    original_f0 = spectral_ops.compute_f0(original_audio, args.sample_rate, args.frame_rate)[0]
    print("Calculating loudness for original audio")
    original_loudness = spectral_ops.compute_loudness(original_audio, args.sample_rate, args.frame_rate)

    avg_original_f0 = np.mean(original_f0)
    avg_original_loudness = np.mean(original_loudness)
    print(f"Average F0 of original file: {avg_original_f0}")
    print(f"Average Loudness of original file: {avg_original_loudness}")


    f0_l1 = abs(avg_synth_f0 - avg_original_f0)
    loudness_l1 = abs(avg_synth_loudness - avg_original_loudness)
    print(f"Average F0 L1: {f0_l1}")
    print(f"Average Loudness L1: {loudness_l1}")

if __name__ == '__main__':
    main()