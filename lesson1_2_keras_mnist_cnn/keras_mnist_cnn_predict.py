# coding:utf-8

import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from PIL import Image

model = load_model("conv_model.h5")


class PredictImg(object):
    def __init__(self):
        pass

    # filename为图片名字
    def pred(self, filename):
        image = Image.open(filename)
        image_L = image.convert("L")
        image_L = image_L.resize((28, 28), Image.ANTIALIAS)
        image_L = np.array(image_L)
        image_L = image_L / 255.  # 如果少了这一步，那么返回的prediction的10个值，要么是0要么是1
        image_L = image_L.reshape(-1, 28, 28, 1)
        prediction = model.predict(image_L)
        print(prediction)
        print(prediction[0])
        Final_prediction = np.argmax(prediction)
        a = 0
        for i in prediction[0]:
            print(a)
            print("percent:%.4f" % i)
            a = a + 1
        return Final_prediction


def main():
    Predict = PredictImg()
    res = Predict.pred("test.jpg")
    print("预测结果为：", res)


if __name__ == "__main__":
    main()