import tensorflow.compat.v1 as tf
import os
import logging
import librosa
import numpy as np
from tqdm import tqdm

from ddsp import spectral_ops

class TasNetDataLoader():
    def __init__(self, mode, data_dir, batch_size, sample_rate, frame_rate):
        if mode != "train" and mode != "valid" and mode != "infer":
            raise ValueError("mode: {} while mode should be "
                             "'train', 'valid', or 'infer'".format(mode))
        if not os.path.isdir(data_dir):
            raise ValueError("cannot find data_dir: {}".format(data_dir))

        self.wav_dir = os.path.join(data_dir, mode)
        self.tfr = os.path.join(data_dir, mode + '.tfr')
        self.mode = mode
        self.n_speaker = 3
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

        if not os.path.isfile(self.tfr):
            self._encode()

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def get_next(self):
        logging.info("Loading data from {}".format(self.tfr))
        with tf.name_scope("input"):
            dataset = tf.data.TFRecordDataset(self.tfr).map(self._decode)
            if self.mode == "train":
                dataset = dataset.shuffle(2000 + 3 * self.batch_size)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(self.batch_size * 5)
            self.iterator = dataset.make_initializable_iterator()
            return self.iterator.get_next()

    def _encode(self):
        logging.info("Writing {}".format(self.tfr))
        with tf.python_io.TFRecordWriter(self.tfr) as writer:
            s1_wav_dir = os.path.join(self.wav_dir, "s1")
            s2_wav_dir = os.path.join(self.wav_dir, "s2")
            s3_wav_dir = os.path.join(self.wav_dir, "s3")
            filenames = os.listdir(s1_wav_dir)
            for filename in tqdm(filenames):
                print("Preprocessing %s" % (os.path.join(s1_wav_dir, filename)))
                s1, _ = librosa.load(
                    os.path.join(s1_wav_dir, filename), self.sample_rate)
                s1_f0 = spectral_ops.compute_f0(s1, self.sample_rate, self.frame_rate)
                s1_loudness = spectral_ops.compute_loudness(s1, self.sample_rate, self.frame_rate)

                print("Preprocessing %s" % (os.path.join(s2_wav_dir, filename)))
                s2, _ = librosa.load(
                    os.path.join(s2_wav_dir, filename), self.sample_rate)
                s2_f0 = spectral_ops.compute_f0(s2, self.sample_rate, self.frame_rate)
                s2_loudness = spectral_ops.compute_loudness(s2, self.sample_rate, self.frame_rate)

                print("Preprocessing %s" % (os.path.join(s3_wav_dir, filename)))
                s3, _ = librosa.load(
                    os.path.join(s3_wav_dir, filename), self.sample_rate)
                s3_f0 = spectral_ops.compute_f0(s3, self.sample_rate, self.frame_rate)
                s3_loudness = spectral_ops.compute_loudness(s3, self.sample_rate, self.frame_rate)
                
                def padding(inputs):
                    return np.pad(
                        inputs, (int(2.55 * self.sample_rate), 0),
                        'constant',
                        constant_values=(0, 0))

                # mix, s1, s2 = padding(mix), padding(s1), padding(s2)

                def sample_to_frame(sample_num):
                    return int(self.frame_rate * sample_num / self.sample_rate)

                def write(l, r):
                    l_frame = sample_to_frame(l)
                    r_frame = sample_to_frame(r)

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "s1_audio": self._float_list_feature(s1[l:r]),
                                "s1_f0": self._float_list_feature(s1_f0[l_frame:r_frame]),
                                "s1_loudness": self._float_list_feature(s1_loudness[l_frame:r_frame]),

                                "s2_audio": self._float_list_feature(s2[l:r]),
                                "s2_f0": self._float_list_feature(s2_f0[l_frame:r_frame]),
                                "s2_loudness": self._float_list_feature(s2_loudness[l_frame:r_frame]),

                                "s3_audio": self._float_list_feature(s3[l:r]),
                                "s3_f0": self._float_list_feature(s3_f0[l_frame:r_frame]),
                                "s3_loudness": self._float_list_feature(s3_loudness[l_frame:r_frame]),
                            }))
                    writer.write(example.SerializeToString())

                now_length = s1.shape[-1]
                if now_length < int(4 * self.sample_rate):
                    continue
                target_length = int(4 * self.sample_rate)
                stride = int(4 * self.sample_rate)
                for i in range(0, now_length - target_length, stride):
                    write(i, i + target_length)
                # if now_length // target_length:
                #     write(now_length - target_length, now_length)

    def _decode(self, serialized_example):
        example = tf.parse_single_example(
            serialized_example,
            features={
                "s1_audio": tf.VarLenFeature(tf.float32),
                "s1_f0": tf.VarLenFeature(tf.float32),
                "s1_loudness": tf.VarLenFeature(tf.float32),

                "s2_audio": tf.VarLenFeature(tf.float32),
                "s2_f0": tf.VarLenFeature(tf.float32),
                "s2_loudness": tf.VarLenFeature(tf.float32),

                "s3_audio": tf.VarLenFeature(tf.float32),
                "s3_f0": tf.VarLenFeature(tf.float32),
                "s3_loudness": tf.VarLenFeature(tf.float32),
            },
        )
        s1_audio = tf.sparse_tensor_to_dense(example["s1_audio"])
        s2_audio = tf.sparse_tensor_to_dense(example["s2_audio"])
        s3_audio = tf.sparse_tensor_to_dense(example["s3_audio"])
        audios = tf.stack([s1_audio, s2_audio, s3_audio])
        f0s = tf.stack([example["s1_f0"], example["s2_f0"], example["s3_f0"]])
        loudness = tf.stack([example["s1_loudness"], example["s2_loudness"], example["s3_loudness"]])
        return audios, f0s, loudness