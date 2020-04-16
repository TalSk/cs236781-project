import argparse
import logging
import os
import sys
# import gin
import ddsp.training

import numpy as np
import tensorflow.compat.v1 as tfv1
import tensorflow as tf

from scipy.io import wavfile
from TasNet import tasnet_tf, tasnet_dataloader


def get_mctn_model_layers(N, L, B, C, R, X, H):
    layers = {
        "conv1d_encoder":
            tfv1.keras.layers.Conv1D(
                filters=N,
                kernel_size=L,
                strides=L,
                activation=tfv1.nn.relu,
                name="encode_conv1d"),
        "bottleneck":
            tfv1.keras.layers.Conv1D(B, 1, 1),
        "1d_deconv":
            tfv1.keras.layers.Dense(L, use_bias=False),
        "f0_deconv": tfv1.keras.Sequential((tfv1.keras.layers.Dense(2 * N, activation="relu", use_bias=False),
                                            tfv1.keras.layers.Dense(2 * N, activation="relu", use_bias=False),
                                            tfv1.keras.layers.Dense(128, activation="relu", use_bias=False))),
        "loudness_deconv": tfv1.keras.Sequential((tfv1.keras.layers.Dense(2 * N, use_bias=False),
                                                  tfv1.keras.layers.Dense(2 * N, use_bias=False),
                                                  tfv1.keras.layers.Dense(1, use_bias=False)))
    }
    for i in range(C):
        layers["1x1_conv_decoder_{}".format(i)] = \
            tfv1.keras.layers.Conv1D(N, 1, 1)
    for r in range(R):
        for x in range(X):
            now_block = "block_{}_{}_".format(r, x)
            layers[now_block + "first_1x1_conv"] = tfv1.keras.layers.Conv1D(
                filters=H, kernel_size=1)
            layers[now_block + "first_PReLU"] = tfv1.keras.layers.PReLU(
                shared_axes=[1])
            layers[now_block + "second_PReLU"] = tfv1.keras.layers.PReLU(
                shared_axes=[1])
            layers[now_block + "second_1x1_conv"] = tfv1.keras.layers.Conv1D(
                filters=B, kernel_size=1)
    return layers


def outputToWav(rawOutput, resultPath, sample_rate=16000):
    if len(rawOutput.shape) == 2:
        rawOutput = rawOutput[0]

    normalizer = float(np.iinfo(np.int16).max)
    array_of_ints = np.array(
        np.asarray(rawOutput) * normalizer, dtype=np.int16)
    wavfile.write(resultPath, sample_rate, array_of_ints)


def main():
    tfv1.disable_eager_execution()
    parser = argparse.ArgumentParser(description='Audio Separation Evaluation')
    parser.add_argument('-dd', '--mctn_data_dir', type=str, default='./mctn_data')
    parser.add_argument('-md', '--mctn_ckpt_dir', type=str, default='./mctn_ckpt')
    parser.add_argument('-vod', '--ddsp_vocals_ckpt_dir', type=str, default='./ddsp_vocals_ckpt')
    parser.add_argument('-bad', '--ddsp_bass_ckpt_dir', type=str, default='./ddsp_bass_ckpt')
    parser.add_argument('-drd', '--ddsp_drums_ckpt_dir', type=str, default='./ddsp_drums_ckpt')
    parser.add_argument('-ld', '--log_dir', type=str, default='./log')
    parser.add_argument('-rd', '--results_dir', type=str, default='./results')

    args = parser.parse_args()
    args.log_file = os.path.join(args.log_dir, 'log.txt')

    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    mctn_model_path = os.path.join(args.mctn_ckpt_dir, "MCTN")  # TODO: Fill ext
    ddsp_vocals_gin_path = os.path.join(args.ddsp_vocals_ckpt_dir, "operative_config-0.gin")
    ddsp_bass_gin_path = os.path.join(args.ddsp_bass_ckpt_dir, "operative_config-0.gin")
    ddsp_drums_gin_path = os.path.join(args.ddsp_drums_ckpt_dir, "operative_config-0.gin")

    # Prepare MCTN model and dataloader.
    # Expects a file named "infer.tfr" in mctn_data_dir
    with tfv1.variable_scope("model") as scope:
        mctn_eval_dataloader = tasnet_dataloader.TasNetDataLoader("infer", data_dir=args.mctn_data_dir,
                                                                  batch_size=2, sample_rate=16000,
                                                                  frame_rate=250)  # TODO: Fill correct values
        layers = get_mctn_model_layers(N=128, B=256, H=512, X=8, R=1, C=3, L=1)  # TODO: Fill correct values

        infer_model = tasnet_tf.TasNet("infer", mctn_eval_dataloader, layers=layers, n_speaker=3, N=128,
                                       L=1, B=256, H=512, P=3, X=8,
                                       R=1, sample_rate_hz=16000, frame_rate_hz=250,
                                       weight_f0=0.5)  # TODO: Fill correct values

    # Load pre-trained MCTN
    mctn = tfv1.train.Saver()
    config = tfv1.ConfigProto(
        #  device_count={'GPU': 0}
    )
    config.allow_soft_placement = True
    with tfv1.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(args.mctn_ckpt_dir)
        assert ckpt
        logging.info('Loading MCTN model from %s', ckpt.model_checkpoint_path)
        mctn.restore(sess, ckpt.model_checkpoint_path)

        sess.run(mctn_eval_dataloader.iterator.initializer)

        # Pass mixed.wav through the pre-trained MCTN to extract f0 and loudness
        print("hi")
        input()
        mctn_loss, mctn_outputs = sess.run(
            fetches=[
                infer_model.loss, infer_model.outputs
            ])

        logging.info(f'MCTN loss on mixed input: {mctn_loss}')

    # Prepare input to each of the DDSP autoencoders.
    vocals_pos = 2
    bass_pos = 0
    drums_pos = 1  # TODO: Change

    audio_features_vocals = {
        'loudness_db': mctn_outputs[1][vocals_pos],
        'f0_hz': mctn_outputs[0][vocals_pos, :, :]
    }
    audio_features_bass = {
        'loudness_db': mctn_outputs[1][bass_pos],
        'f0_hz': mctn_outputs[0][bass_pos, :, :]
    }
    audio_features_drums = {
        'loudness_db': mctn_outputs[1][drums_pos],
        'f0_hz': mctn_outputs[0][drums_pos, :, :]
    }

    # Load all 3 autoencoders
    # gin_file = os.path.join(args.model_dir, 'operative_config-0.gin')
    # gin.parse_config_file(gin_file)
    vocals_ddsp = ddsp.training.models.Autoencoder()
    vocals_ddsp.restore(args.ddsp_vocals_ckpt_dir)
    bass_ddsp = ddsp.training.models.Autoencoder()
    bass_ddsp.restore(args.ddsp_bass_ckpt_dir)
    drums_ddsp = ddsp.training.models.Autoencoder()
    drums_ddsp.restore(args.ddsp_drums_ckpt_dir)

    # Resynthesize
    vocals_gen = vocals_ddsp(audio_features_vocals, training=False)
    bass_gen = vocals_ddsp(audio_features_bass, training=False)
    drums_gen = vocals_ddsp(audio_features_drums, training=False)

    # TODO: Add metrics evaluation.

    # Save to results
    outputToWav(rawOutput=vocals_gen, resultPath=os.path.join(args.results_dir, "sep_vocals.wav"))
    outputToWav(rawOutput=bass_gen, resultPath=os.path.join(args.results_dir, "sep_bass.wav"))
    outputToWav(rawOutput=drums_gen, resultPath=os.path.join(args.results_dir, "sep_drums.wav"))


if __name__ == "__main__":
    main()
