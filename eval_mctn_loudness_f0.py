import argparse
import os
import logging
import librosa
import numpy as np
from ddsp import spectral_ops

def main():
    parser = argparse.ArgumentParser(description='Evaluate MCTN output loudness & F0 of every instrument')
    parser.add_argument('-ld', '--log_dir', type=str, default='./mctn_log')
    parser.add_argument('-rd', '--results_dir', type=str, default='./results')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000)
    parser.add_argument('-fr', '--frame_rate', type=int, default=250)
    parser.add_argument('-osd', '--original_sound_dir', type=str, default='./results/DDSP - Same artist - Test')

    args = parser.parse_args()
    args.log_file = os.path.join(args.log_dir, 'log.txt')

    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

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
    
    # Calc average loudness & F0 diff for every instrument
    for instrument, audio_features in [("bass", audio_features_bass), ("drums", audio_features_drums), ("vocals", audio_features_vocals)]:
        original_audio, _ = librosa.load(os.path.join(args.original_sound_dir, f"original_{instrument}.wav"), args.sample_rate)

        synth_audio_samples = audio_features_vocals["f0_hz"].shape[1] * args.sample_rate // args.frame_rate
        original_audio_samples = original_audio.shape[0]

        if synth_audio_samples < original_audio_samples:
            logging.info(f"Trimming original {instrument} audio samples from {original_audio_samples} to {synth_audio_samples}")
            original_audio_samples = synth_audio_samples
            original_audio = original_audio[:original_audio_samples]

        # Assuming only 1 batch
        synth_f0 = audio_features["f0_hz"][0]
        synth_loudness = audio_features["loudness_db"][0]

        logging.info(f"Calculating F0 for {instrument} original audio")
        original_f0 = spectral_ops.compute_f0(original_audio, args.sample_rate, args.frame_rate)[0]
        logging.info(f"Calculating loudness for {instrument} original audio")
        original_loudness = spectral_ops.compute_loudness(original_audio, args.sample_rate, args.frame_rate)

        f0_l1 = np.mean(abs(synth_f0 - original_f0))
        loudness_l1 = np.mean(abs(synth_loudness - original_loudness))
        logging.info(f"Average {instrument} F0 L1: {f0_l1}")
        logging.info(f"Average {instrument} Loudness L1: {loudness_l1}")

if __name__ == "__main__":
    main()