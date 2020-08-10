#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: clausmichele
@modified : github.com/parthc-rob
@date : 08-07-2020
"""

import time
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
from cv_bridge import CvBridge
from collections import deque
from scipy import ndimage

import rospy
from sensor_msgs.msg import Image


def SpatialCNN(input, is_training=False, output_channels=3, reuse=tf.compat.v1.AUTO_REUSE):
    with tf.compat.v1.variable_scope('block1', reuse=reuse):
        output = tf.compat.v1.layers.conv2d(input, 128, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 20):
        with tf.compat.v1.variable_scope('block%d' % layers, reuse=reuse):
            output = tf.compat.v1.layers.conv2d(output,  64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.compat.v1.layers.batch_normalization(output, training=is_training))
    with tf.compat.v1.variable_scope('block20', reuse=reuse):
        output = tf.compat.v1.layers.conv2d(output, output_channels, 3, padding='same', use_bias=False)
    return input - output


def Temp3CNN(input, is_training=False, output_channels=3, reuse=tf.compat.v1.AUTO_REUSE):
    input_middle = input[:, :, :, 3:6]
    with tf.compat.v1.variable_scope('temp-block1', reuse=reuse):
        output = tf.compat.v1.layers.conv2d(input, 128, 3, padding='same', activation=tf.nn.leaky_relu)
    for layers in range(2, 20):
        with tf.compat.v1.variable_scope('temp-block%d' % layers, reuse=reuse):
            output = tf.compat.v1.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.leaky_relu(output)
    with tf.compat.v1.variable_scope('temp-block20', reuse=reuse):
        output = tf.compat.v1.layers.conv2d(output, output_channels, 3, padding='same', use_bias=False)
    return input_middle - output


class ViDeNN(object):
    def __init__(self, sess):
        # imagestream variables
        self.input_img = deque(maxlen = 3)
        self.bridge = CvBridge()
        self.sess = sess
        # build model
        self.Y_ = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='clean_image')
        self.X = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='noisy_image')
        self.Y = SpatialCNN(self.X)
        self.Y_frames = tf.compat.v1.placeholder(tf.float32, [None, None, None, 9], name='clean_frames')
        self.Xframes = tf.compat.v1.placeholder(tf.float32, [None, None, None, 9], name='noisy_frames')
        self.Yframes = Temp3CNN(self.Xframes)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully, ####...")

        tf.compat.v1.global_variables_initializer().run()

        full_path = tf.train.latest_checkpoint(
            '/home/marsians/nasa_src/src2_qual/catkin_ws/src/marsians_sensor_proc/marsians_video_denoise/src/ViDeNN/ckpt_videnn'
            )
        if(full_path is None):
            print('[!] No Temp3-CNN checkpoint!')
            quit()
        vars_to_restore_temp3CNN = {}
        for i in range(len(tf.compat.v1.global_variables())):
            if tf.compat.v1.global_variables()[i].name[0] != 'b':
                a = tf.compat.v1.global_variables()[i].name.split(':')[0]
                vars_to_restore_temp3CNN[a] = tf.compat.v1.global_variables()[i]
        saver_t = tf.compat.v1.train.Saver(var_list=vars_to_restore_temp3CNN)
        saver_t.restore(self.sess, full_path)

        full_path = tf.train.latest_checkpoint(
            '/home/marsians/nasa_src/src2_qual/catkin_ws/src/marsians_sensor_proc/marsians_video_denoise/src/ViDeNN/ckpt_videnn'
            )
        if(full_path is None):
            print('[!] No Spatial-CNN checkpoint!')
            quit()
        vars_to_restore_spatialCNN = {}
        for i in range(len(tf.compat.v1.global_variables())):
            if tf.compat.v1.global_variables()[i].name[0] != 't':
                a = tf.compat.v1.global_variables()[i].name.split(':')[0]
                vars_to_restore_spatialCNN[a] = tf.compat.v1.global_variables()[i]
        saver_s = tf.compat.v1.train.Saver(var_list=vars_to_restore_spatialCNN)
        saver_s.restore(self.sess, full_path)
        # else:
        #     load_model_status, _ = self.load(ckpt_dir)
        print("[*] Model restore successfully!")


        self.pub = rospy.Publisher('denoised_image', Image, queue_size=3)
        print "....publisher initialized"
        rospy.Subscriber('/stereo/left/image_rect', Image, self.callback)
        rospy.spin()

    def callback(self, msg):
        print "### Processing image"
        print msg.header
        # l2_localized = Odometry(
        #     header=msg.header
        #     pose=PoseWithCovariance(Pose(msg.state.pose)) # need msg to give covariance as float64[36]
        #     twist=TwistWithCovariance(Twist(msg.state.velocity))
        # )
        # self.pub(l2_localized)

        # deque of cvMat images
        self.input_img.append(
            cv2.cvtColor(
                self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'), cv2.COLOR_GRAY2RGB
                         )
            )
        if len(self.input_img) > 3:
            self.input_img.popleft()
        elif len(self.input_img) == 3:
            for idx in range(3):

                clean_image = np.zeros((self.input_img[0].shape[0],  self.input_img[0].shape[1], 3), np.uint8)
                if idx == 0:
                    noisy = self.input_img[idx]
                    print "\n Dim for noisy \t"
                    print noisy.shape
                    noisy1 = self.input_img[idx+1]
                    noisy2 = self.input_img[idx+2]

                    noisy = noisy.astype(np.float32) / 255.0
                    noisy1 = noisy1.astype(np.float32) / 255.0
                    noisy2 = noisy2.astype(np.float32) / 255.0

                    noisyin2 = np.zeros((1, noisy.shape[0], noisy.shape[1], 9))
                    current = np.zeros((noisy.shape[0], noisy.shape[1], 3))
                    previous = np.zeros((noisy.shape[0], noisy.shape[1], 3))

                    noisyin = np.zeros((3, noisy.shape[0], noisy.shape[1], 3))
                    noisyin[0] = noisy
                    noisyin[1] = noisy1
                    noisyin[2] = noisy2
                    out = self.sess.run([self.Y], feed_dict={self.X: noisyin})
                    out = np.asarray(out)

                    noisyin2[0, :, :, 0:3] = out[0, 0]
                    noisyin2[0, :, :, 3:6] = out[0, 0]
                    noisyin2[0, :, :, 6:] = out[0, 1]
                    temp_clean_image = self.sess.run([self.Yframes], feed_dict={self.Xframes: noisyin2})

                    temp_clean_image = np.squeeze(np.asarray(temp_clean_image)) # (1, 1, 480, 640, 3)
                    clean_image = temp_clean_image*255

                    print "\n dataslice difference\t"
                    print (noisy[400:403, 400:403, :]*255 - temp_clean_image[400:403, 400:403, :])
                    clean_image = ndimage.rotate(clean_image.astype(np.uint8), 90)

                    #clean_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2GRAY)
                    rosimg_clean_image = self.bridge.cv2_to_imgmsg(clean_image, encoding="rgb8")
                    rosimg_clean_image.header = msg.header
                    # cv2.imwrite(save_dir + '/%04d.png' % idx, temp_clean_image[0, 0]*255)
                    self.pub.publish(rosimg_clean_image)

                    noisyin2[0, :, :, 0:3] = out[0, 0]
                    noisyin2[0, :, :, 3:6] = out[0, 1]
                    noisyin2[0, :, :, 6:] = out[0, 2]
                    current[:, :, :] = out[0, 2, :, :, :]
                    previous[:, :, :] = out[0, 1, :, :, :]
                else:
                    if idx < (len(self.input_img)-2):
                        noisy3 = self.input_img[idx+2]
                        noisy3 = noisy3.astype(np.float32) / 255.0

                        out2 = self.sess.run([self.Y], feed_dict={self.X: np.expand_dims(noisy3, 0)})
                        out2 = np.asarray(out2)
                        noisyin2[0, :, :, 0:3] = previous
                        noisyin2[0, :, :, 3:6] = current
                        noisyin2[0, :, :, 6:] = out2[0, 0]
                        previous = current
                        current = out2[0, 0]
                    else:
                        try:
                            out2
                        except NameError:
                            out2 = np.zeros((out.shape))
                            out2 = out
                            out2[0, 0] = out[0, 2]
                        noisyin2[0, :, :, 0:3] = current
                        noisyin2[0, :, :, 3:6] = out2[0, 0]
                        noisyin2[0, :, :, 6:] = out2[0, 0]
                temp_clean_image = self.sess.run([self.Yframes], feed_dict={self.Xframes: noisyin2})

                temp_clean_image = np.squeeze(np.asarray(temp_clean_image))
                clean_image = temp_clean_image*255

                clean_image = ndimage.rotate(clean_image.astype(np.uint8), 90)

                # clean_image = cv2.cvtColor(clean_image, cv2.COLOR_RGB2GRAY)
                rosimg_clean_image = self.bridge.cv2_to_imgmsg(clean_image, encoding="rgb8")
                rosimg_clean_image.header = msg.header

                # cv2.imwrite(save_dir+ '/%04d.png' % (idx + 1), temp_clean_image[0, 0] * 255)
                self.pub.publish(rosimg_clean_image)

        def load(self, checkpoint_dir):
            print("[*] Reading checkpoint...")
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                full_path = tf.train.latest_checkpoint(checkpoint_dir)
                global_step = int(full_path.split('/')[-1].split('-')[-1])
                saver.restore(self.sess, full_path)
                return True, global_step
            else:
                return False, 0

def main(_):
    devices = tf.config.list_physical_devices('GPU')
    if devices:
        rospy.init_node('node_video_denoise')
        # if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        tf.config.experimental.set_memory_growth(devices[0], True)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            model = ViDeNN(sess)
    else:
        print("CPU\n")
        with tf.device('/cpu:0'):
            with tf.compat.v1.Session() as sess:
                model = ViDeNN(sess)


if __name__ == '__main__':
    tf.compat.v1.app.run()
