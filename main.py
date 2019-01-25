import tensorflow as tf
from utils import read_data
from libs.models.nets_factory import get_network
import model
import time
import os
import argparse
def main(args):
    # log_path = './log/'
    logs_path = './logs'
    model_path = './model'
    ckpt_path = './checkpoint/train.h5-1510s1'
    image_size = 40
    batch_size = 64
    epoch = 60
    epsilon = 1e-6
    #learning_rate = 1e-4
    x1 = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])
    x2 = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])
    y = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    pred = model.lst_unet(x1, x2, 3, True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                  initializer=tf.constant_initializer(0), trainable=False)
    # pred = get_network(x1, x2, weight_decay=0.001, is_training=True)
    #y_nonzero = y.nonzero
    mse_loss = tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(y, pred))+epsilon))
    tf.add_to_collection("loss", mse_loss)
    loss = tf.add_n(tf.get_collection("loss"))
    # #loss = tf.nn.l2_loss(y-pred)
    # global_step = tf.Variable(0)

    learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=batch_size,
                                                    decay_rate=0.94,  # 0.94
                                                    staircase=True)

    # opt = tf.train.MomentumOptimizer(learning_rate=learning_rate_node, momentum=0.99)  # 0.99
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # 0.99
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies([tf.group(*update_ops)]):
    #     train_opt = optimizer.minimize(mse_loss, global_step=global_step)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
    #opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.control_dependencies(update_ops):
        grads = opt.compute_gradients(loss)
        train_op = opt.apply_gradients(grads, global_step=global_step)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True
    sess = tf.InteractiveSession(config=tf_config)
    sess.run(tf.global_variables_initializer())
    var_list = tf.trainable_variables()
    if global_step is not None:
        var_list.append(global_step)
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)
    # saver = tf.train.Saver(max_to_keep=10)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    #saver = tf.train.Saver()


    # checkpoint_dir = model_path
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    # if ckpt and ckpt.model_checkpoint_path:
    #    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    #    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='optimizer')
    #    epoch_learning_rate = 1e-4
    # else:
    #
    # opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
    # epoch_learning_rate = 1e-3
    # sess.run(tf.global_variables_initializer())
    counter=0
    start_time = time.time()
    input1, input2, label = read_data(ckpt_path)
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #     ckpt = tf.train.get_checkpoint_state(model_path)
    #     if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #     else:
    #         sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    epoch_learning_rate = 1e-3
    for one_ep in range(epoch):
        # Run by batch images
        if one_ep == epoch * 0.3 or one_ep == epoch * 0.5:
            epoch_learning_rate = epoch_learning_rate * 0.1
        train_loss = 0.0
        batch_idxs = len(input1) // batch_size
        for idx in range(0, batch_idxs):
            batch_input1 = input1[idx*batch_size: (idx+1)*batch_size]
            batch_input2 = input2[idx*batch_size: (idx+1)*batch_size]
            batch_label = label[idx*batch_size: (idx+1)*batch_size]
            counter = counter + 1
            _, err = sess.run([train_op, loss], feed_dict={x1: batch_input1, x2: batch_input2, y: batch_label,
                                                               learning_rate: epoch_learning_rate})
            if counter % 20 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]"\
                      % ((one_ep + 1), counter, time.time() - start_time, err))

            if counter % 500 ==0:
                train_loss = train_loss / 500.0
                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss)])
                summary_writer.add_summary(summary=train_summary, global_step=epoch)
                summary_writer.flush()
                save_path = os.path.join(model_path, 'LST_CNN_itr%d.ckpt'%(counter))
                saver.save(sess, save_path)
                print('model parameter has been saved in %s'%save_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
	parser.add_argument("--gpu", choices=['0','1','2','3'], default='0', help="gpu_id")
	args = parser.parse_args()
	main(args)









