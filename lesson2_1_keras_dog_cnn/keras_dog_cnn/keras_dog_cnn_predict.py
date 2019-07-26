from keras.models import load_model
from PIL import Image
import numpy as np
import os


class PredImage(object):
    def __init__(self, Model_name, Test_image):
        self.model_name = Model_name
        self.test_image = Test_image

    def Class_dog(self, prediction):
        Final_prediction = np.argmax(prediction)
        dog_name_list = ["哈士奇", "柯基犬", "藏獒", "金毛"]
        for i in range(4):
            if Final_prediction == i:
                print("输出结果为：", dog_name_list[i])

    def Predict(self):
        model = load_model(self.model_name)
        img = Image.open(self.test_image)
        img_rgb = img.convert("RGB")
        img_resize = img_rgb.resize((100, 100), Image.BILINEAR)
        img_array = np.array(img_resize)
        img_array = img_array / 255.
        img_array = img_array.reshape(-1, 100, 100, 3)
        prediction = model.predict(img_array)
        print(prediction)
        self.Class_dog(prediction)


test_dir = "./test_img"
for img in os.listdir(test_dir):
    PredictImage = PredImage("dog.h5", os.path.join(test_dir, img))
    PredictImage.Predict()
