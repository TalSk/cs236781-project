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

    bass_pos = 0
    drums_pos = 1
    vocals_pos = 2

    f0_loss_bass = 0
    f0_loss_drums = 0
    f0_loss_vocals = 0
    ld_loss_bass = 0
    ld_loss_drums = 0
    ld_loss_vocals = 0

    with tfv1.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(args.mctn_ckpt_dir)
        assert ckpt
        logging.info('Loading MCTN model from %s', ckpt.model_checkpoint_path)
        mctn.restore(sess, ckpt.model_checkpoint_path)

        sess.run(mctn_eval_dataloader.iterator.initializer)
        while True:
            try:
                # Pass mixed.wav through the pre-trained MCTN to extract f0 and loudness
                mctn_loss, mctn_outputs, losses = sess.run(
                    fetches=[
                        infer_model.loss, infer_model.outputs, infer_model.losses
                    ])


                whole_output += [mctn_outputs]

                logging.info(f'MCTN loss on mixed input: {mctn_loss}')
                total_loss += mctn_loss

                f0_loss_bass += losses[0][bass_pos]
                f0_loss_drums += losses[0][drums_pos]
                f0_loss_vocals += losses[0][vocals_pos]
                ld_loss_bass += losses[1][bass_pos]
                ld_loss_drums += losses[1][drums_pos]
                ld_loss_vocals += losses[1][vocals_pos]

            except tfv1.errors.OutOfRangeError:
                logging.info(f'Done aggregating results. Total average loss: {total_loss / len(whole_output)}')
                break

        logging.info(f"Bass f0 av. loss: {(f0_loss_bass / len(whole_output))}")
        logging.info(f"Drums f0 av. loss: {(f0_loss_drums / len(whole_output))}")
        logging.info(f"Vocals f0 av. loss: {(f0_loss_vocals / len(whole_output))}")
        logging.info(f"Bass ld av. loss: {(ld_loss_bass / len(whole_output))}")
        logging.info(f"Drums ld av. loss: {(ld_loss_drums / len(whole_output))}")
        logging.info(f"Vocals ld av. loss: {(ld_loss_vocals / len(whole_output))}")
        logging.info(f"Bass av. loss: {(0.9 * ld_loss_bass + 0.1 * f0_loss_bass) / len(whole_output)}")
        logging.info(f"Drums av. loss: {(0.9 * ld_loss_drums + 0.1 * f0_loss_drums) / len(whole_output)}")
        logging.info(f"Vocals av. loss: {((0.9 * ld_loss_vocals + 0.1 * f0_loss_vocals) / len(whole_output))}")
        input()  # TODO

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

if __name__ == "__main__":
    main()
