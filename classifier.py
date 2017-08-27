import tensorflow as tf
import numpy as np
import os
import glob
import numpy as np
import cv2

sess = tf.Session()
saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")


dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = input('Insert the Path to the File to be Classified')
filename = dir_path + image_path
image_size=128
images = []
image = cv2.imread(filename)

image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = images / 255

x= graph.get_tensor_by_name("x:0")
img_size=128
num_channels=3
img_size_flat = img_size * img_size * num_channels
x_batch = images.reshape(1, img_size_flat)
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 2))
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
print(result)