# coding: utf-8

import tensorflow as tf
import time


def save_graph():
    # saver = tf.train.import_meta_graph("../checkpoint/yolov3_coco.ckpt.meta", clear_devices=True)
    saver = tf.train.import_meta_graph("../checkpoint/yolov3_coco_demo.ckpt.meta", clear_devices=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "../checkpoint/yolov3_coco.ckpt")
        saver.restore(sess, "../checkpoint/yolov3_coco_demo.ckpt")

        # save
        localtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        train_writer = tf.summary.FileWriter("../log/%s" % localtime, sess.graph)


if __name__ == "__main__":
    save_graph()
