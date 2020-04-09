# import math
# import numpy as np
# import tensorflow.compat.v2 as tf
#
# tfkl = tf.keras.layers
# EPS = 1e-8
#
# class ConvTasNet(tfkl.Layer):
#     def __init__(self, N, L, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear='relu', name="conv-tasnet"):
#         """
#         Args:
#             N: Number of filters in autoencoder
#             L: Length of the filters (in samples)
#             B: Number of channels in bottleneck 1 × 1-conv block
#             H: Number of channels in convolutional blocks
#             P: Kernel size in convolutional blocks
#             X: Number of convolutional blocks in each repeat
#             R: Number of repeats
#             C: Number of speakers
#             norm_type: BN, gLN, cLN
#             causal: causal or non-causal
#             mask_nonlinear: use which non-linear function to generate mask
#         """
#         super(ConvTasNet, self).__init__(name=name)
#         self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C
#         self.norm_type = norm_type
#         self.causal = causal
#         self.mask_nonlinear = mask_nonlinear
#
#         self.encoder = Encoder(L, N)
#         self.separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear)
#         self.decoder = Decoder(N, L)
#
#
#     def call(self, x):
#         pass
#
#
# class Encoder(tfkl.Layer):
#     def __init__(self, L, N, name="tasnet-encoder"):
#         super(Encoder, self).__init__(name=name)
#         # Hyper-parameter
#         self.L, self.N = L, N
#         # Components
#         # 50% overlap
#         self.conv1d_U = tfkl.Conv1D(N, kernel_size=L, stride=L // 2, activation=tf.nn.relu)
#
#     def call(self, mixture):
#         """
#         Args:
#             mixture: [M, T], M is batch size, T is #samples
#         Returns:
#             mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
#         """
#         mixture = tf.keras.backend.expand_dims(mixture, axis=1)
#         mixture_w = tf.nn.relu(self.conv1d_U(mixture))
#         return mixture_w
#
#
# class Decoder(tfkl.Layer):
#     def __init__(self, N, L, name="tasnet-decoder"):
#         super(Decoder, self).__init__(name=name)
#         # Hyper-parameter
#         self.N, self.L = N, L
#         # Components
#         self.basis_signals = tfkl.Dense(units=L, input_shape=(N,), use_bias=False)
#
#     def forward(self, mixture_w, est_mask):
#         """
#         Args:
#             mixture_w: [M, N, K]
#             est_mask: [M, C, N, K]
#         Returns:
#             est_source: [M, C, T]
#         """
#         # D = W * M
#         source_w = tf.keras.backend.expand_dims(mixture_w, axis=1) * est_mask  # [M, C, N, K]
#         source_w = tf.transpose(source_w, perm=[0, 1, 3, 2])
#         # S = DV
#         est_source = self.basis_signals(source_w)  # [M, C, K, L]
#         est_source = overlap_and_add(est_source, self.L // 2)  # M x C x T
#         return est_source
#
#
# def overlap_and_add(signal, frame_step):
#     """Reconstructs a signal from a framed representation.
#     Adds potentially overlapping frames of a signal with shape
#     `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
#     The resulting tensor has shape `[..., output_size]` where
#         output_size = (frames - 1) * frame_step + frame_length
#     Args:
#         signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
#         frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
#     Returns:
#         A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
#         output_size = (frames - 1) * frame_step + frame_length
#     Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
#     """
#     outer_dimensions = signal.size()[:-2]
#     frames, frame_length = signal.size()[-2:]
#
#     subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
#     subframe_step = frame_step // subframe_length
#     subframes_per_frame = frame_length // subframe_length
#     output_size = frame_step * (frames - 1) + frame_length
#     output_subframes = output_size // subframe_length
#
#     subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
#
#     frame = tf.range(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
#     frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
#     frame = frame.contiguous().view(-1)
#
#     result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
#     result.index_add_(-2, frame, subframe_signal)
#     result = result.view(*outer_dimensions, -1)
#     return result
#
#
# class TemporalConvNet(tfkl.Layer):
#     def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False,
#                  mask_nonlinear='relu', name="tasnet-tcn"):
#         """
#         Args:
#             N: Number of filters in autoencoder
#             B: Number of channels in bottleneck 1 × 1-conv block
#             H: Number of channels in convolutional blocks
#             P: Kernel size in convolutional blocks
#             X: Number of convolutional blocks in each repeat
#             R: Number of repeats
#             C: Number of speakers
#             norm_type: BN, gLN, cLN
#             causal: causal or non-causal
#             mask_nonlinear: use which non-linear function to generate mask
#         """
#         super(TemporalConvNet, self).__init__(name=name)
#         # Hyper-parameter
#         self.C = C
#         self.mask_nonlinear = mask_nonlinear
#
#         self.network = tf.keras.Sequential()
#         # Components
#         # [M, N, K] -> [M, N, K]
#         layer_norm = ChannelwiseLayerNorm(N)
#         self.network.add(layer_norm)
#         # [M, N, K] -> [M, B, K]
#         bottleneck_conv1x1 = tfkl.Conv1D(B, 1, use_bias=False, input_shape=(N,))
#         self.network.add(bottleneck_conv1x1)
#         # [M, B, K] -> [M, B, K]
#         temporal_conv_net = tf.keras.Sequential()
#         for r in range(R):
#             blocks = tf.keras.Sequential()
#             for x in range(X):
#                 dilation = 2 ** x
#                 padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
#                 blocks.add(TemporalBlock(B, H, P, stride=1,
#                                          padding=padding,
#                                          dilation=dilation,
#                                          norm_type=norm_type,
#                                          causal=causal))
#             temporal_conv_net.add(blocks)
#         self.network.add(temporal_conv_net)
#         # [M, B, K] -> [M, C*N, K]
#         mask_conv1x1 = tfkl.Conv1D(C * N, 1, use_bias=False, input_shape=(B,))
#         self.network.add(mask_conv1x1)
#
#     def forward(self, mixture_w):
#         """
#         Keep this API same with TasNet
#         Args:
#             mixture_w: [M, N, K], M is batch size
#         returns:
#             est_mask: [M, C, N, K]
#         """
#         M, N, K = mixture_w.size()
#         score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
#         score = score.view(M, self.C, N, K)  # [M, C*N, K] -> [M, C, N, K]
#         if self.mask_nonlinear == 'softmax':
#             est_mask = tf.nn.softmax(score, dim=1)
#         elif self.mask_nonlinear == 'relu':
#             est_mask = tf.nn.relu(score)
#         else:
#             raise ValueError("Unsupported mask non-linear function")
#         return est_mask
#
#
# class TemporalBlock(tfkl.Layer):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride, padding, dilation, norm_type="gLN", causal=False):
#         super(TemporalBlock, self).__init__()
#         # [M, B, K] -> [M, H, K]
#         self.net = tf.keras.Sequential()
#
#         conv1x1 = tfkl.Conv1D(out_channels, 1, use_bias=False, input_shape=(in_channels,))
#         self.net.add(conv1x1)
#         prelu = tfkl.PReLU()
#         self.net.add(prelu)
#         norm = chose_norm(norm_type, out_channels)
#         self.net.add(norm)
#         # [M, H, K] -> [M, B, K]
#         dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
#                                         stride, padding, dilation, norm_type,
#                                         causal)
#         self.net.add(dsconv)
#
#     def forward(self, x):
#         """
#         Args:
#             x: [M, B, K]
#         Returns:
#             [M, B, K]
#         """
#         residual = x
#         out = self.net(x)
#         # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
#         return out + residual  # look like w/o F.relu is better than w/ F.relu
#         # return F.relu(out + residual)
#
#
# class DepthwiseSeparableConv(tfkl.Layer):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride, padding, dilation, norm_type="gLN", causal=False, name="tasnet-depthsizeconv"):
#         super(DepthwiseSeparableConv, self).__init__()
#         # Use `groups` option to implement depthwise convolution
#         # [M, H, K] -> [M, H, K]
#         depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
#                                    stride=stride, padding=padding,
#                                    dilation=dilation,
#                                    use_bias=False, input_shape=)
#         if causal:
#             chomp = Chomp1d(padding)
#         prelu = nn.PReLU()
#         norm = chose_norm(norm_type, in_channels)
#         # [M, H, K] -> [M, B, K]
#         pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
#         # Put together
#         if causal:
#             self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm,
#                                      pointwise_conv)
#         else:
#             self.net = nn.Sequential(depthwise_conv, prelu, norm,
#                                      pointwise_conv)
#
#     def forward(self, x):
#         """
#         Args:
#             x: [M, H, K]
#         Returns:
#             result: [M, B, K]
#         """
#         return self.net(x)
#
#
#
# def chose_norm(norm_type, channel_size):
#     """The input of normlization will be (M, C, K), where M is batch size,
#        C is channel size and K is sequence length.
#     """
#     if norm_type == "gLN":
#         return GlobalLayerNorm(channel_size)
#     elif norm_type == "cLN":
#         return ChannelwiseLayerNorm(channel_size)
#     else: # norm_type == "BN":
#         # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
#         # along M and K, so this BN usage is right.
#         return tfkl.BatchNormalization(axis=1)
#
#
# class ChannelwiseLayerNorm(tfkl.Layer):
#     """Channel-wise Layer Normalization (cLN)"""
#
#     def __init__(self, channel_size, name="tasnet-channelwisenorm"):
#         super(ChannelwiseLayerNorm, self).__init__(name=name)
#         self.gamma = tf.Variable(tf.ones((1, channel_size, 1)))  # [1, N, 1]
#         self.beta = tf.Variable(tf.zeros((1, channel_size, 1)))  # [1, N, 1]
#
#     def forward(self, y):
#         """
#         Args:
#             y: [M, N, K], M is batch size, N is channel size, K is length
#         Returns:
#             cLN_y: [M, N, K]
#         """
#         mean = np.mean(y, dim=1, keepdim=True)  # [M, 1, K]
#         var = np.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
#         cLN_y = self.gamma * (y - mean) / np.power((var + EPS), 0.5) + self.beta
#         return cLN_y
#
#
# class GlobalLayerNorm(tfkl.Layer):
#     """Global Layer Normalization (gLN)"""
#     def __init__(self, channel_size, name="tasnet-globallayernorm"):
#         super(GlobalLayerNorm, self).__init__(name=name)
#         self.gamma = tf.Variable(tf.ones((1, channel_size, 1)))  # [1, N, 1]
#         self.beta = tf.Variable(tf.zeros((1, channel_size, 1)))  # [1, N, 1]
#
#     def forward(self, y):
#         """
#         Args:
#             y: [M, N, K], M is batch size, N is channel size, K is length
#         Returns:
#             gLN_y: [M, N, K]
#         """
#         # TODO: in torch 1.0, torch.mean() support dim list
#         mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
#         var = (np.power(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
#         gLN_y = self.gamma * (y - mean) / np.power(var + EPS, 0.5) + self.beta
#         return gLN_y