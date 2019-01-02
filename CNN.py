# coding: utf-8
import tensorflow as tf


# Define CNN model  定义CNN模型
def conv_net(input_x_dict, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        # 为了适应有多个输入变量的情况，TF Estimator要求输入是一个字典
        input_x = input_x_dict['images']

        # Input layer 输入层 [28*28*1]
        input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])
        # The reason why the first dimension should be -1, is that we don't know the size of input,
        # tf will figure it out automatically
        # -1是因为不知道有多少个图像，tf会帮我们算出来第一个维度的值

        # Convolution Layer 1 [Filter 5*5*32]
        # 卷积层1 [卷积核5*5*32]
        # layers.conv2d parameters  参数如下
        # inputs: a tensor  输入，是一个张量
        # filters: number of the filter  卷积核个数，也就是卷积层的厚度
        # kernel_size: 卷积核的尺寸
        # strides: 扫描步长
        # padding: 边补0，valid不需要补0，same需要补0，为了保证输入输出的尺寸一致,补多少不需要知道
        # activation: 激活函数
        conv1 = tf.layers.conv2d(
            inputs=input_x_images,
            filters=32,
            kernel_size=[5, 5],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )

        # Pooling Layer 1  池化层1
        # inputs 输入：张量必须要有4个维度
        # pool_size：过滤器尺寸
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2
        )

        # Convolution Layer 2  卷积层2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            strides=1,
            padding='same',
            activation=tf.nn.relu
        )

        # Pooling Layer 2  池化层2
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2
        )

        # Flat layer  平坦化
        flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        # densely-connected layers 全连接层
        # tf.layers.dense
        # inputs: 张量
        # units： 神经元的个数
        # activation: 激活函数
        dense = tf.layers.dense(
            inputs=flat,
            units=1024,
            activation=tf.nn.relu
        )

        # Dropout层
        # tf.layers.dropout
        # inputs 张量
        # rate 丢弃率
        # training 是否是在训练的时候丢弃
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.5,
            training=is_training
        )

        # Output Layer, activation is not needed (actually a dense layer)
        # 输出层，不用激活函数（本质就是一个全连接层）
        logits = tf.layers.dense(
            inputs=dropout,
            units=10
        )
        # Output size 输出形状 [?,10]

        return logits


# Define the model function (As TF Estimator required)
def model_fn(features, labels, mode):
    # Build the neural network  建立神经网络
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    # 因为Dropout对于训练和测试/预测有不同的行为，我们需要建立两个独立的网络，但它们共享相同的权重
    logits_train = conv_net(features, reuse=False, is_training=True)  # Net for training  对于训练
    logits_test = conv_net(features, reuse=True, is_training=False)  # Net for evaluation and prediction  对于评估和预测

    # Predictions  预测
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return  如果是预测模式，则提前退出
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': pred_classes,
            'probabilities': pred_probas
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Define loss  定义损失函数
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits_train)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = 0.001   # 学习速率
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Calculate accuracy  计算准确率
    acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=pred_classes)
    # Calculate recall  计算召回率
    rec_op = tf.metrics.recall(labels=tf.argmax(labels, axis=1), predictions=pred_classes)
    eval_metrics = {
        'accuracy': acc_op,
        'recall': rec_op
    }

    # For tensorboard display  用于tensorboard显示
    tf.summary.scalar('accuracy', acc_op[1])
    tf.summary.scalar('recall', rec_op[1])

    # Evaluate the model  评估模型
    if mode == tf.estimator.ModeKeys.EVAL:
        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics
        )
        return estim_specs
