import tensorflow as tf
import cv2 as cv
import os
import model
import numpy as np
def testreal():
    t1_path = 't1.tif'
    t2_path = 't2.tif'
    mask_path = 'mask.tif'
    out_path = 't2r.tif'
    model_path = './models'
    image_size = 40
    sess = tf.Session()
    x1 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])
    x2 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])
    pred = model.lst_unet(x1, x2, 3, False)
    checkpoint_dir = model_path
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        input1 = cv.imread(t1_path, cv.IMREAD_UNCHANGED)
        input2 = cv.imread(t2_path, cv.IMREAD_UNCHANGED)
        mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
        mask = mask / 255.0
        input1 = input1.reshape([1, image_size, image_size, 1])
        input2 = input2.reshape([1, image_size, image_size, 1])
        mask = mask.reshape([1, image_size, image_size, 1])

        tmp_result = sess.run([pred], feed_dict={x1: input1, x2: input2})
        tmp_result = np.array(tmp_result)
        result = tmp_result.squeeze()
        mask = mask.squeeze()
        input2 = input2.squeeze()
        input1 = input1.squeeze()
        image = result * (1 - mask) + input2
        cv.imwrite(out_path, image)

if __name__ == '__main__':
    testreal()