# coding: utf-8
import tensorflow as tf
import CNN
import os
import shutil
from tensorflow.examples.tutorials.mnist import input_data

# Set the hyper params  设置超参数
num_step = 20000  # number of the training step  训练迭代数
train_batch_size = 50 # batch size for training  训练的batch大小
test_batch_size = 50 # batch size for test  测试所有的batch大小

# If a trained model exists, delete it and train a new model from the beginning
# 如果有已经训练好的模型存在，删除它，从头开始训练
if os.path.exists('saved_model'):
    shutil.rmtree('saved_model')

# Display the tensorflow log
# 显示tensorflow日志
tf.logging.set_verbosity(tf.logging.INFO)

# Get data from MNIST dataset
# 从MNIST数据集中获取数据
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)
train_x = mnist.train.images
train_y = mnist.train.labels
test_x = mnist.test.images
test_y = mnist.test.labels

# =============Training  训练模型=============
# Build the Estimator  创建一个tensorflow estimator
model = tf.estimator.Estimator(CNN.model_fn, model_dir=r'saved_model/')

# Define the input function for training  # 定义训练的数据输入函数
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':train_x}, y=train_y,
    batch_size=train_batch_size, num_epochs=None, shuffle=True
)

# Begin the training  开始训练
model.train(train_input_fn, steps=num_step)

# =============Evaluate  测试评估模型=============
# Define the input function for evaluating  # 定义测试的数据输入函数
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images':test_x}, y=test_y,
    batch_size=test_batch_size, shuffle=False
)

# Use the Estimator 'evaluate' method  开始测试
model.evaluate(test_input_fn)
