# coding: utf-8
import tensorflow as tf
from PIL import Image
import numpy as np
import CNN


class CNNPredictor:
    # Initialize the saved trained CNN model
    # 初始化保存的训练好的CNN模型
    def __init__(self):
        self.model = tf.estimator.Estimator(CNN.model_fn, model_dir=r'saved_model/')
        print('获取模型')

    # Process the image
    # 处理图片
    def process_img(self, filepath):
        img = Image.open(filepath)  # Open the file  打开文件
        img = img.resize((28, 28))
        img = img.convert('L')  # Transfer the image into a grey image  转换成灰度图
        imgarr = np.array(img, dtype=np.float32)
        imgarr = imgarr.reshape([1, 28*28])/255.0
        return imgarr

    # Do predictions and return the result
    # 进行预测，返回预测结果
    def get_predictions(self, filepath):
        imgarr = self.process_img(filepath)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images':imgarr}, batch_size=1, shuffle=False
        )
        predictions = list(self.model.predict(predict_input_fn))
        return predictions[0]


if __name__ == '__main__':
    predictor = CNNPredictor()
    pre = predictor.get_predictions('r.png')
    print(pre)
