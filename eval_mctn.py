import argparse
import logging
import os
import sys

import numpy as np
import tensorflow.compat.v1 as tfv1
import tensorflow as tf

from TasNet import tasnet_tf, tasnet_dataloader


def main():
    tfv1.disable_eager_execution()
    parser = argparse.ArgumentParser(description='Audio Separation Evaluation - MCTN part')
    parser.add_argument('-dd', '--mctn_data_dir', type=str, default='./mctn_data')
    parser.add_argument('-md', '--mctn_ckpt_dir', type=str, default='./mctn_ckpt')
    parser.add_argument('-ld', '--log_dir', type=str, default='./mctn_log')
    parser.add_argument('-rd', '--results_dir', type=str, default='./results')
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000)
    parser.add_argument('-fr', '--frame_rate', type=int, default=250)
    parser.add_argument('-cl', '--calc_loss', type=int, default=0)
    parser.add_argument('-osd', '--original_sound_dir', type=str, default='./results/DDSP - Same artist - Test')

    args = parser.parse_args()
    args.log_file = os.path.join(args.log_dir, 'log.txt')

    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    # Prepare MCTN model and dataloader.
    # Expects a file named "infer.tfr" in mctn_data_dir
    with tfv1.variable_scope("model") as scope:
        mctn_eval_dataloader = tasnet_dataloader.TasNetDataLoader("infer", data_dir=args.mctn_data_dir,
                                                                  batch_size=1, sample_rate=args.sample_rate,
                                                                  frame_rate=args.frame_rate)  # TODO: Fill correct values

        infer_model = tasnet_tf.TasNet("infer", mctn_eval_dataloader, n_speaker=3, N=128,
                                       L=64, B=256, H=512, P=3, X=8,
                                       R=1, sample_rate_hz=args.sample_rate, frame_rate_hz=args.frame_rate,
                                       weight_f0=0.1)  # TODO: Fill correct values

    # Load pre-trained MCTN
    mctn = tfv1.train.Saver()
    config = tfv1.ConfigProto(
        #  device_count={'GPU': 0}
    )
    config.allow_soft_placement = True

    whole_output = []
    total_loss = 0

    with tfv1.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(args.mctn_ckpt_dir)
        assert ckpt
        logging.info('Loading MCTN model from %s', ckpt.model_checkpoint_path)
        mctn.restore(sess, ckpt.model_checkpoint_path)

        sess.run(mctn_eval_dataloader.iterator.initializer)
        i = 0
        while True:
            try:
                # Pass mixed.wav through the pre-trained MCTN to extract f0 and loudness
                mctn_loss, mctn_outputs = sess.run(
                    fetches=[
                        infer_model.loss, infer_model.outputs
                    ])
                whole_output += [mctn_outputs]

                logging.info(f'MCTN loss on mixed input: {mctn_loss}')
                total_loss += mctn_loss

                i += 1
                # if i == 5:
                #     break
            except tfv1.errors.OutOfRangeError:
                logging.info(f'Done aggregating results. Total average loss: {total_loss / len(whole_output)}')
                break

        bass_pos = 0
        drums_pos = 1  # TODO: Change
        vocals_pos = 2

        # Prepare input to be saved for each of the DDSP autoencoders.
        whole_f0 = []
        whole_lds = []
        for output in whole_output:
            whole_f0 += [output[0]]
            whole_lds += [tf.raw_ops.Pack(values=output[1])]

        whole_f0_stacked = tf.concat(whole_f0, axis=2)

        whole_lds_stacked = tf.concat(whole_lds, axis=2)

        audio_features_vocals = {
            'loudness_db': whole_lds_stacked[vocals_pos, :, :],
            'f0_hz': whole_f0_stacked[vocals_pos, :, :]
        }
        audio_features_bass = {
            'loudness_db': whole_lds_stacked[bass_pos, :, :],
            'f0_hz': whole_f0_stacked[bass_pos, :, :]
        }
        audio_features_drums = {
            'loudness_db': whole_lds_stacked[drums_pos, :, :],
            'f0_hz': whole_f0_stacked[drums_pos, :, :]
        }

        np.save(os.path.join(args.results_dir, 'vocals_f0_hz.npy'), sess.run(audio_features_vocals['f0_hz']),
                allow_pickle=False)
        np.save(os.path.join(args.results_dir, 'vocals_loudness_db.npy'), sess.run(audio_features_vocals['loudness_db']),
                allow_pickle=False)
        np.save(os.path.join(args.results_dir, 'bass_f0_hz.npy'), sess.run(audio_features_bass['f0_hz']),
                allow_pickle=False)
        np.save(os.path.join(args.results_dir, 'bass_loudness_db.npy'), sess.run(audio_features_bass['loudness_db']),
                allow_pickle=False)
        np.save(os.path.join(args.results_dir, 'drums_f0_hz.npy'), sess.run(audio_features_drums['f0_hz']),
                allow_pickle=False)
        np.save(os.path.join(args.results_dir, 'drums_loudness_db.npy'), sess.run(audio_features_drums['loudness_db']),
                allow_pickle=False)

        if args.calc_loss == 1:
            for instrument, audio_features in [("bass", audio_features_bass), ("drums", audio_features_drums), ("vocals", audio_features_vocals)]:
                original_audio, _ = librosa.load(os.path.join(args.original_sound_dir, f"original_{instrument}.wav"), args.sample_rate)

                synth_audio_samples = audio_features_vocals["f0_hz"].shape[1]
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
