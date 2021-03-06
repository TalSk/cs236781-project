import tensorflow.compat.v1 as tf
import numpy as np
import ddsp

class TasNet:
    def __init__(self, mode, dataloader, n_speaker, N, L, B, H, P, X,
                 R, sample_rate_hz, frame_rate_hz, weight_f0, layers=None):
        self.mode = mode
        self.dataloader = dataloader
        
        self.C = n_speaker
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.P = P
        self.X = X
        self.R = R

        self.sample_rate = sample_rate_hz
        self.frame_rate = frame_rate_hz
        self.weight_f0 = weight_f0

        self.dtype = tf.float32

        if layers is None:
            self.layers = self.build_layers()
        else:
            self.layers = layers

        self._build_graph()

    def build_layers(self):
        layers = {
            "conv1d_encoder":
                tf.keras.layers.Conv1D(
                    filters=self.N,
                    kernel_size=self.L,
                    strides=self.L,
                    activation=tf.nn.relu,
                    name="encode_conv1d"),
            "bottleneck":
                tf.keras.layers.Conv1D(self.B, 1, 1),
            "1d_deconv":
                tf.keras.layers.Dense(self.L, use_bias=False),
            "f0_deconv": tf.keras.Sequential((tf.keras.layers.Dense(2 * self.N, activation="relu", use_bias=False),
                                              tf.keras.layers.LayerNormalization(),
                                              tf.keras.layers.Dense(2 * self.N, activation="relu", use_bias=False),
                                              tf.keras.layers.LayerNormalization(),
                                              tf.keras.layers.Dense(128, activation="relu", use_bias=False),
                                              tf.keras.layers.LayerNormalization()
                                              )),
            "loudness_deconv": tf.keras.Sequential((tf.keras.layers.Dense(2 * self.N, activation="relu", use_bias=False),
                                                    tf.keras.layers.LayerNormalization(),
                                                    tf.keras.layers.Dense(2 * self.N, activation="relu", use_bias=False),
                                                    tf.keras.layers.LayerNormalization(),
                                                    tf.keras.layers.Dense(1, use_bias=False)))
        }
        for i in range(self.C):
            layers["1x1_conv_decoder_{}".format(i)] = \
                tf.keras.layers.Conv1D(self.N, 1, 1)
        for r in range(self.R):
            for x in range(self.X):
                now_block = "block_{}_{}_".format(r, x)
                layers[now_block + "first_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.H, kernel_size=1)
                layers[now_block + "first_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1])
                layers[now_block + "second_PReLU"] = tf.keras.layers.PReLU(
                    shared_axes=[1])
                layers[now_block + "second_1x1_conv"] = tf.keras.layers.Conv1D(
                    filters=self.B, kernel_size=1)
        return layers

    def _build_graph(self):
        # audios: [batch_size, max_len]
        audios, f0s, loudness = self.dataloader.get_next()

        input_audio = audios[:, 0, :]

        self.single_audios = single_audios = tf.unstack(
            audios[:, 1:, :], axis=1)

        with tf.variable_scope("encoder"):
            # encoded_input: [batch_size, some len, N]
            encoded_input = self.layers["conv1d_encoder"](
                inputs=tf.expand_dims(input_audio, -1))
            self.encoded_len = int(4 * self.sample_rate // self.L)

        with tf.variable_scope("bottleneck"):
            # norm_input: [batch_size, some len, N]
            norm_input = self._channel_norm(encoded_input, "bottleneck")

            # block_input: [batch_size, some len, B]
            block_input = self.layers["bottleneck"](norm_input)

        for r in range(self.R):
            for x in range(self.X):
                now_block = "block_{}_{}_".format(r, x)
                with tf.variable_scope(now_block):
                    block_output = self.layers[now_block +
                                               "first_1x1_conv"](block_input)
                    block_output = self.layers[now_block +
                                               "first_PReLU"](block_output)
                    block_output = self._global_norm(block_output, "first")

                    block_output = self._depthwise_conv1d(block_output, x)
                    block_output = self.layers[now_block +
                                               "second_PReLU"](block_output)
                    block_output = self._global_norm(block_output, "second")
                    block_output = self.layers[now_block +
                                               "second_1x1_conv"](block_output)

                    block_input = block_output = block_output + block_input

        sep_output_list = [
            self.layers["1x1_conv_decoder_{}".format(i)](block_output)
            for i in range(self.C)
        ]

        # softmax
        probs = tf.nn.softmax(tf.stack(sep_output_list, axis=-1))
        prob_list = tf.unstack(probs, axis=-1)

        # C, B, T, N
        sep_output_list = [mask * encoded_input for mask in prob_list]

        # C, B, T, 128
        f0_deconved = [
            self.layers["f0_deconv"](sep_output)
            for sep_output in sep_output_list
        ]

        T_TO_F = (self.sample_rate // self.frame_rate) // self.L
        # C, B, F, 128
        if T_TO_F > 1:
            f0_deconved = [y[:, ::T_TO_F, :] for y in f0_deconved]

        # C, B, T
        loudness_deconved = [
            tf.squeeze(self.layers["loudness_deconv"](sep_output), axis=-1)
            for sep_output in sep_output_list
        ]
        if T_TO_F > 1:
            loudness_deconved = [y[:, ::T_TO_F] for y in loudness_deconved]

        probs = [tf.nn.softplus(y) + 1e-3 for y in f0_deconved]
        probs = [prob / tf.reduce_sum(prob, axis=-1, keepdims=True) for prob in probs]
        output_f0s = tf.squeeze(self._compute_f0_hz(probs), axis=-1)

        # C, B, F
        output_loudnesses = loudness_deconved

        self.outputs = (output_f0s, output_loudnesses)

        f0_loss = self._calc_f0_loss(f0s, output_f0s)
        loudness_loss = self._calc_loudness_loss(loudness, output_loudnesses)

        f0_diff = [tf.reduce_mean(tf.abs(f0s[:, i, :] - output_f0s[i, :, :])) for i in range(self.C)]
        ld_diff = [tf.reduce_mean(tf.abs(loudness[:, i, :] - output_loudnesses[i])) for i in range(self.C)]
        self.losses = (f0_diff, ld_diff)

        self.loss = self.weight_f0 * f0_loss + (1.0 - self.weight_f0) * loudness_loss
        self.inputs = (f0s, loudness)

    def _calc_f0_loss(self, gt_f0s, pred_f0s):
        list_difference = [gt_f0s[:, i, :] - pred_f0s[i, :, :] for i in range(self.C)]
        return tf.reduce_mean(tf.abs(list_difference))

    def _calc_loudness_loss(self, gt_lds, pred_lds):
        list_difference = [gt_lds[:, i, :] - pred_lds[i] for i in range(self.C)]
        return tf.reduce_mean(tf.abs(list_difference))

    def _compute_f0_hz(self, probs):
        """Computes the f0 in herz from a distribution over D MIDI intervals."""
        # probs: [B, T, D]
        depth = int(probs[0].shape[-1])

        midi_bins = tf.constant(
            1.0 * np.arange(depth).reshape((1, 1, -1)) / depth,
            dtype=tf.float32)  # [1, 1, D]

        f0_hz = [ddsp.core.midi_to_hz(
            (depth-1.0) * tf.reduce_sum(
            midi_bins * prob, axis=-1, keepdims=True)) for prob in probs]  # [B, T, 1]
        return f0_hz

    def _channel_norm(self, inputs, name):
        # inputs: [batch_size, some len, channel_size]
        with tf.variable_scope(name):
            channel_size = inputs.shape[-1]
            E = tf.reshape(
                tf.reduce_mean(inputs, axis=[2]), [-1, self.encoded_len, 1])
            Var = tf.reshape(
                tf.reduce_mean((inputs - E) ** 2, axis=[2]),
                [-1, self.encoded_len, 1])
            gamma = tf.get_variable(
                "gamma", shape=[1, 1, channel_size], dtype=self.dtype)
            beta = tf.get_variable(
                "beta", shape=[1, 1, channel_size], dtype=self.dtype)
            return ((inputs - E) / (Var + 1e-8) ** 0.5) * gamma + beta

    def _global_norm(self, inputs, name):
        # inputs: [batch_size, some len, channel_size]
        with tf.variable_scope(name):
            channel_size = inputs.shape[-1]
            E = tf.reshape(tf.reduce_mean(inputs, axis=[1, 2]), [-1, 1, 1])
            Var = tf.reshape(
                tf.reduce_mean((inputs - E) ** 2, axis=[1, 2]), [-1, 1, 1])
            gamma = tf.get_variable(
                "gamma", shape=[1, 1, channel_size], dtype=self.dtype)
            beta = tf.get_variable(
                "beta", shape=[1, 1, channel_size], dtype=self.dtype)
            return ((inputs - E) / (Var + 1e-8) ** 0.5) * gamma + beta

    def _depthwise_conv1d(self, inputs, x):
        inputs = tf.reshape(inputs, [-1, 1, self.encoded_len, self.H])
        filters = tf.get_variable(
            "dconv_filters", [1, self.P, self.H, 1], dtype=self.dtype)
        outputs = tf.nn.depthwise_conv2d(
            input=inputs,
            filter=filters,
            strides=[1, 1, 1, 1],
            padding='SAME',
            rate=[1, 2 ** x])
        return tf.reshape(outputs, [-1, self.encoded_len, self.H])
