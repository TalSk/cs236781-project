import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import librosa
import numpy as np
import logging

from tasnet_tf import TasNet
from tasnet_utils import *
from tasnet_dataloader import TasNetDataLoader

def main():
    tf.disable_eager_execution()

    args, logger = setup()
    global_step = tf.Variable(0, trainable=False, name="global_step")
    train_dataloader = TasNetDataLoader("train", args.data_dir,
                                        args.batch_size, args.sample_rate, args.frame_rate)
    valid_dataloader = TasNetDataLoader("valid", args.data_dir,
                                        args.batch_size, args.sample_rate, args.frame_rate)

    with tf.variable_scope("model") as scope:
        train_model = TasNet("train", train_dataloader, args.C, args.N,
                                args.L, args.B, args.H, args.P, args.X,
                                args.R, args.sample_rate, args.frame_rate, args.weight_f0)
        scope.reuse_variables()
        valid_model = TasNet("valid", valid_dataloader, args.C, args.N,
                                args.L, args.B, args.H, args.P, args.X,
                                args.R, args.sample_rate, args.frame_rate, args.weight_f0,
                                layers=train_model.layers)

    print_num_of_trainable_parameters()
    trainable_variables = tf.trainable_variables()

    valid_loss = read_log(args.log_file)

    learning_rate = tf.placeholder(tf.float32, [])
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients = tf.gradients(train_model.loss, trainable_variables)
    update = opt.apply_gradients(
        zip(gradients, trainable_variables), global_step=global_step)

    saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count={'GPU': args.use_gpu}
    )
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:

        ckpt = tf.train.get_checkpoint_state(args.log_dir)
        if ckpt:
            logging.info('Loading model from %s', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.info('Loading model with fresh parameters')
            sess.run(tf.global_variables_initializer())

        lr = args.learning_rate
        no_improve_count = 0
        prev_loss = np.inf

        for epoch in range(1, args.max_epoch + 1):
            sess.run(train_dataloader.iterator.initializer)
            logging.info('-' * 20 + ' epoch {} '.format(epoch) + '-' * 25)

            train_iter_cnt, train_loss_sum = 0, 0
            while True:
                try:
                    cur_loss, _, cur_global_step, outputs, inputs = \
                        sess.run(
                            fetches=[train_model.loss, update, global_step, train_model.outputs,
                                        train_model.inputs],
                            feed_dict={learning_rate: lr}
                        )
                    train_loss_sum += cur_loss * args.batch_size
                    train_iter_cnt += args.batch_size
                except tf.errors.OutOfRangeError:
                    logging.info(
                        'step = {} , train loss = {:5f} , lr = {:5f}'.
                            format(cur_global_step,
                                    train_loss_sum / train_iter_cnt, lr))
                    break

            sess.run(valid_dataloader.iterator.initializer)
            valid_iter_cnt, valid_loss_sum = 0, 0
            while True:
                try:
                    cur_loss, = sess.run([valid_model.loss])
                    valid_loss_sum += cur_loss * args.batch_size
                    valid_iter_cnt += args.batch_size
                except tf.errors.OutOfRangeError:
                    epoch_loss = (valid_loss_sum / valid_iter_cnt)

                    if epoch_loss >= prev_loss:
                        no_improve_count += 1
                        if no_improve_count >= 3:
                            lr *= args.learning_rate_decrease
                    else:
                        no_improve_count = 0
                    prev_loss = epoch_loss

                    logging.info('validation loss = {:5f}'.format(epoch_loss))
                    if epoch_loss < valid_loss:
                        valid_loss = epoch_loss
                        saver.save(
                            sess,
                            args.checkpoint_path,
                            global_step=cur_global_step)
                    break

if __name__ == '__main__':
    main()