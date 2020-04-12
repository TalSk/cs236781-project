import tensorflow.compat.v1 as tf
import numpy as np


class TasNet:
    def __init__(self, mode, dataloader, layers, n_speaker, N, L, B, H, P, X,
                 R, sample_rate_hz):
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
        self.dtype = tf.float32

        self.layers = layers

        self._build_graph()

    def _calc_sdr(self, s_hat, s):
        def norm(x):
            return tf.reduce_sum(x ** 2, axis=-1, keepdims=True)

        s_target = tf.reduce_sum(
            s_hat * s, axis=-1, keepdims=True) * s / norm(s)
        upp = norm(s_target)
        low = norm(s_hat - s_target)
        return 10 * tf.log(upp / low) / tf.log(10.0)

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
            self.encoded_len = (int(4 * self.sample_rate) - self.L) // (
                    self.L // 2) + 1

        with tf.variable_scope("bottleneck"):
            # norm_input: [batch_size, some len, N]
            norm_input = self._channel_norm(encoded_input, "bottleneck")

            # block_inptu: [batch_size, some len, B]
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

        sep_output_list = [mask * encoded_input for mask in prob_list]

        sep_output_list = [
            (self.layers["f0_deconv"](sep_output), self.layers["loudness_deconv"](sep_output))
            for sep_output in sep_output_list
        ]

        # self.outputs = outputs = [
        #     tf.signal.overlap_and_add(
        #         signal=sep_output,
        #         frame_step=self.L // 2,
        #     ) for sep_output in sep_output_list
        # ]

        # sdr1 = self._calc_sdr(outputs[0], single_audios[0]) + \
        #        self._calc_sdr(outputs[1], single_audios[1])
        #
        # sdr2 = self._calc_sdr(outputs[1], single_audios[0]) + \
        #        self._calc_sdr(outputs[0], single_audios[1])

        # sdr = tf.maximum(sdr1, sdr2)
        # self.loss = tf.reduce_mean(-sdr) / self.C
        probs = [tf.nn.softplus(y[0]) + 1e-3 for y in sep_output_list]
        probs = [prob / tf.reduce_sum(prob, axis=-1, keepdims=True) for prob in probs]
        output_f0s = self._compute_unit_midi(probs)
        output_loudnesses = [lds for _, lds in sep_output_list]

        self.outputs = []

        for i in range(len(output_f0s)):
            self.outputs += [output_f0s[i], output_loudnesses[i]]

        f0_loss = self._calc_f0_loss(f0s, output_f0s)
        loudness_loss = self._calc_loudness_loss(loudness, output_loudnesses)

        self.loss = f0_loss + loudness_loss

    def _calc_f0_loss(self, gt_f0s, pred_f0s):
        difference = gt_f0s - pred_f0s
        return tf.reduce_mean(tf.abs(difference))

    def _calc_loudness_loss(self, gt_lds, pred_lds):
        difference = gt_lds - pred_lds
        return tf.reduce_mean(tf.log(tf.abs(difference)))

    def _compute_unit_midi(self, probs):
        """Computes the midi from a distribution over the unit interval."""
        # probs: [B, T, D]
        depth = int(probs[0].shape[-1])

        unit_midi_bins = tf.constant(
            1.0 * np.arange(depth).reshape((1, 1, -1)) / depth,
            dtype=tf.float32)  # [1, 1, D]

        f0_unit_midi = [tf.reduce_sum(
            unit_midi_bins * prob, axis=-1, keepdims=True) for prob in probs]  # [B, T, 1]
        return f0_unit_midi

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
